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
from typing import Union, Tuple, List, Optional


NORM_CONFIGS = {'InstanceNorm': '风格迁移、GAN、图像处理',
                'GroupNorm': '小批量训练、目标检测Mask R-CNN',
                'SyncBatchNorm': '多卡分布式训练',
                'BatchRenorm': '小批量场景下的稳定性改进',
                'SwitchableNorm': '多任务学习,结合BN/IN/LN',  
                'AdaIN': '神经风格迁移',
                'Conditional InstanceNorm': '条件生成模型、条件GAN',
                'WeightNorm': '生成模型、RNN训练稳定性',
                'SpectralNorm': 'GAN判别器,权重谱范数约束',
                'RMSNorm': 'Transformer、NLP模型',
                'FRN': '图像分类、分割任务',
                'Local Response Norm': '早期CNNAlexNet',
                'PixelNorm': '生成模型StyleGAN',
                'ScaleNorm': 'Transformer、注意力机制',
                'PowerNorm': '图像分类任务',
                'IterNorm/Decorrelated Norm': '特征去相关性增强',
                'GroupWhitening': '色彩归一化、图像增强',
                'EvoNorm': '图像分类,可演变的归一化' }


class LayerNorm(nn.Module):
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
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 判断是否进行仿射变换
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
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
    """"""


class SyncBatchNorm(nn.Module):
    """"""


class InstanceNorm(nn.Module):
    """"""


class GroupNorm(nn.Module):
    """"""


class ScaleNorm(nn.Module):
    """"""

