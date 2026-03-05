"""
Author: Redal
Date: 2025-12-13
Todo: 完成归一化层的底层实现,并添加到norm layer中,并且支持不同数据精度
      以供后续模型架构的model调用
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Union, Tuple, List, Optional


class LayerNorm(nn.Module):
    """计算该样本在所有通道+空间位置上的均值和方差
    LN(x)=gamma·(x-E(x))/sqrt(Var(x)+eps)+bias
    normlized_shape输入张量的形状[-1, -2, ..., -n]
    elementwise_affine是否进行gamma和bias的放射变换"""
    def __init__(self, normalized_shape: int,
                 eps: float=1e-5,
                 elementwise_affine: bool=True,
                 dtype=None,
                 device=None
                 )->None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 判断是否进行仿射变换
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        # 初始化参数weight和bias
        self.init_parameters()
    def init_parameters(self,)->None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f"Input tensor shape {x.shape} is incompatible with "
                             f"normalized_shape {self.normalized_shape}.")
        # 计算公式: LayerNorm(x)=weights·(X-E(x)) / sqrt(Var(x)+eps) + bias
        norm_dim = len(self.normalized_shape)
        assert x.shape[-norm_dim:]==self.normalized_shape, \
               f"输入的x_fp32形状与self.normalized_shape形状不兼容"
        norm_dims = tuple(range(-norm_dim, 0))
        mean = torch.mean(x, dim=norm_dims, keepdim=True)
        var = torch.var(x, dim=norm_dims, unbiased=False, keepdim=True)
        x_norm = (x -mean) / torch.sqrt(var + self.eps)
        # 进行仿射变换
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm
    

class LayerNormFP32(LayerNorm):
    """使用float16/bfloat16进行低精度计算时,层归一化的均值,
    方差计算可能会因精度不足导致数值不稳定"""
    def forward(self, x:torch.Tensor)->torch.Tensor:
        orig_type = x.dtype
        # 确保输入的x是float32
        if x.dtype in [torch.float16, torch.bfloat16]:
            x_fp32 = x.to(torch.float32)
            x_norm = super().forward(x_fp32)
            return x_norm.to(orig_type)
        else: 
            return super().forward(x)


class BatchNorm(nn.Module):
    """计算该通道在所有样本 + 所有空间位置上的均值和方差
    BN(x)=gamma·(X-E(x))/sqrt(Var(x)+eps)+bias
    num_features输入张量通道的数量即C的值 
    momentum更新运行均值和方差时的动量系数
    track_running_stats是否在训练过程中跟踪并更新运行统计量"""
    def __init__(self, num_features: int,
                 eps: float=1e-5,
                 momentum: float=0.1,
                 affine: bool=True,
                 track_running_stats: bool=True,
                 device=None,
                 dtype=None
                 )->None:
        super().__init__()
        factory_kwargs = {'device':device, 'dtype': dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        # 若需要放射变换,则进行参数注册
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=device))
        else :
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.init_parameters()
    def init_parameters(self,)->None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x.shape形状[N, C, H, W]或[N, L, C]
        if x.dim() < 2: raise ValueError(f'[WARNING] 通道的维度需要大于2D,实际{x.dim()}D!')
        shape = [1] * x.dim()
        shape[1] = self.num_features
        # 根据training和track_running_stats参数,进行BN计算
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                # self.num_batches_tracked += 1
                self.num_batches_tracked.add_(1)
        if self.training or not self.track_running_stats:
            var_dim = tuple(range(0, x.dim()))
            var_dim = var_dim[:1] + var_dim[2:] 
            # 排除第1维并且求解mean和var值,即不使用channels进行计算
            mean = x.mean(dim=var_dim, keepdim=True)
            var = x.var(dim=var_dim, unbiased=False, keepdim=True)
            # 更新running stats进行指数平均移动
            if self.training and self.track_running_stats:
                # m = x.numel() / x.size(1)  # 已删除：未使用且易误解
                if self.momentum is None: 
                    factor = 1 / float(self.num_batches_tracked)
                else:
                    factor = self.momentum
                # 使用view(self.num_features)确保形状为 (C,)
                current_mean = mean.view(self.num_features)
                current_var = var.view(self.num_features)
                self.running_mean = (1 - factor) * self.running_mean + factor * current_mean
                self.running_var = (1 - factor) * self.running_var + factor * current_var
        else:
            # 推理阶段:使用running stats
            mean = self.running_mean.view(shape)
            var = self.running_var.view(shape)
        # 进行batch_norm计算并判断是否进行归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_norm = self.weight.view(shape) * x_norm + self.bias.view(shape) 
        return x_norm
    

class BatchNormFp32(BatchNorm):
    """使用float16/bfloat16进行低精度计算时,层归一化的均值,
    方差计算可能会因精度不足导致数值不稳定"""
    def forward(self, x:torch.Tensor)->torch.Tensor:
        orig_type = x.dtype
        if x.dtype in [torch.float16, torch.bfloat16]:
            x_fp32 = x.to(torch.float32)
            x_norm = super().forward(x_fp32)
            return x_norm.to(orig_type)
        else:
            return super().forward(x)


class SyncBatchNorm(nn.Module):
    """同步批归一化,Synchronized Batch Normalization用于分布式训练中多GPU间的
    统计量同步在分布式训练中,每个GPU上的batch统计量会被聚合,确保全局一致性
    BN(x)=gamma·(X-E(x))/sqrt(Var(x)+eps)+bias
    affine是否进行仿射变换
    num_features输入张量通道的数量即C的值 
    momentum更新运行均值和方差时的动量系数
    track_running_stats是否在训练过程中跟踪并更新运行统计量"""
    def __init__(self, num_features: int,
                 eps:float=1e-5,
                 momentum:float=0.1,
                 affine: bool=True,
                 track_running_stats: bool=True,
                 device=None,
                 dtype=None
                 )->None:
        super().__init__()
        factory_kwargs = {'device':device, 'dtype':dtype}
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        # 进行参数注册
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.register_buffer('num_batches_tracked', torch.zeros(num_features, dtype=torch.long, device=device))
        self.init_parameters()
    def init_parameters(self, )->None:
        if self.track_running_stats:
            self.running_mean.zeros_()
            self.running_var.fill_(1)
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.zeros_()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.ones_(self.bias)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x.shape形状[N, C, H, W]或[N, L, C]
        if x.dim() < 2: raise ValueError(f'[WARNING] 通道的维度需要大于2D,实际{x.dim()}D!')
        shape = [1] * x.dim()
        shape[1] = self.num_features
        # 根据training和track_running_stats参数,进行BN计算
        # 便于后续计算factor因子,以更新running_mean和running_var
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
        if self.training or not self.track_running_stats:
            var_dim = tuple(range(0, x.dim()))
            var_dim = var_dim[:1] + var_dim[2:]
            # 排除第1维并且求解mean和var值,即不使用channels进行计算
            mean = x.mean(dim=var_dim, keepdim=True)
            var = x.var(dim=var_dim, unbiased=False, keepdim=True)
            # 同步统计量:在分布式训练中聚合所有GPU的统计量
            if self.training and dist.is_available() and dist.is_initialized():
                # 计算当前GPU的统计量,用于后续同步操作
                count = torch.tensor(x.numel() // x.size(1), dtype=torch.float32, device=self.device)
                local_sum = mean.squeeze(var_dim).view(self.num_features) * count
                local_sqr_sum = (var.squeeze(var_dim).view(self.num_features) + (local_sum / count)**2) *count
                # 统计所有GPU的统计量
                stats = torch.stack([local_sum, local_sqr_sum, count])
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                # 重新计算全局统计量
                global_sum, global_sqr_sum, global_count = stats.unbind(0)
                global_mean = global_sum / global_count
                global_var = global_sqr_sum / global_count - global_mean ** 2
                # 使用全局统计量进行归一化
                mean = global_mean.view(shape)
                var = global_var.view(shape)

                # 更新running stats(使用全局统计量)
                if self.training and self.track_running_stats:
                    if self.momentum is not None:
                        factor = 1 / float(self.num_batches_tracked)
                    else:
                        factor = self.momentum
                    self.running_mean = (1-factor)*self.running_mean + factor * global_mean
                    self.running_var = (1-factor)*self.running_var + factor * global_var
            else: # 非分布式训练或推理模式
                if self.training and self.track_running_stats:
                    current_mean = mean.view(self.num_features)
                    current_var = var.view(self.num_features)
                    if self.momentum is None:
                        factor = 1 / float(self.num_batches_tracked)
                    else:
                        factor = self.momentum
                    self.running_mean = (1-factor)*self.running_mean + factor*current_mean
                    self.running_var = (1-factor)*self.running_var + factor*current_var
        else:
            # 推理阶段:使用running stats
            mean = self.running_mean.view(shape)
            var = self.running_var.view(shape)
        # 根据self.affine进行bn(x)求解
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_norm = self.weight.view(shape) * x_norm + self.bias.view(shape)
        return x_norm


class SyncBatchNormFp32(SyncBatchNorm):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        orig_type = x.dtype
        if x.dtype in [torch.float16, torch.bfloat16]:
            x_fp32 = x.to(torch.float32)
            x_norm = super().forward(x_fp32)
            return x_norm.to(orig_type)
        else:
            return super().forward(x)


class ScaleNorm(nn.Module):
    """ScaleNorm是一种简化的归一化方法,使用可学习的缩放参数进行归一化
    与LayerNorm不同,ScaleNorm不需要学习均值和方差参数,仅使用L2范数进行归一化
    SN(x) = gamma * x / sqrt(sum(x^2, dim=-1, keepdim=True) + eps)
    dim: 归一化的维度,通常为特征维度; eps: 防止除零的小常数
    clamp_min: 可选的最小值限制，防止梯度爆炸"""
    def __init__(self, dim:int,
                 eps: float=1e-5,
                 clamp_min: float=1e-2,
                 device=None,
                 dtype=None
                 )->None:
        super().__init__()


class GroupNorm(nn.Module):
    """"""


