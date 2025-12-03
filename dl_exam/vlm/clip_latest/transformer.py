"""
Author: Redal
Date: 2025-12-03
Todo: 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from typing import Callable, Dict, List, Optional
from typing import Sequence, Tuple, Type, Union, Literal
from .utils import to_2tuple, feature_take_indices
from .pos_embed import get_2d_sincos_pos_embed


"""
class LayerNormFp32(nn.Module):
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]],
                 eps: float=1e-5,
                 elementwise_affine:bool=True,
                 device=None,
                 dtype=None):
        super().__init__()
        # 将normalized_shape参数转为tuple类型
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
        # 给weight和bias参数初始化
        self.reset_parameter()
    def reset_parameter(self,)->None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # 转换为torch。float32保证稳定计算
        orig_type = x.dtype
        x_fp32 = x.to(torch.float32)
        norm_dim = len(self.normalized_shape)
        assert x_fp32.shape[-norm_dim:]==self.normalized_shape, \
               f"输入的x_fp32形状与self.normalized_shape形状不兼容"
        # 在标准化维度上计算均值和方差
        norm_dims = tuple(range(-norm_dim, 0))
        mean = torch.mean(x_fp32, dim=norm_dims, keepdim=True)
        var = torch.var(x_fp32, dim=norm_dims, unbiased=True, keepdim=True)
        x_norm = (x_fp32 -mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm.to(orig_type) 
"""
class LayerNormFp32(nn.LayerNorm):
    """继承torch的LayerNorm以处理fp16,通过先转换为float32再转换回来"""
    def forward(self, x:torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """子类化PyTorch的LayerNorm(并转换回输入数据类型)"""
    def forward(self, x:torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    

"""
class GELU(nn.Module):
    def __init__(self, approximate: Literal['none', None]=None):
        super().__init__()
        if approximate not in ('none', None):
            raise NotImplementedError("目前仅实现了精确的GELU(approximate=None)")
        self.approximate = approximate
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # GELU原始公式为 GELU(x)=0.5·x·(1 + erf(x/sqrt(2)))
        if self.approximate is None:
            return x * 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))
        else:
            return x * torch.sigmoid(1.702 * x)
"""  
class QuickGELU(nn.Module):
    """GELU激活函数的快速近似实现
    比官方nn.GELU/nn.SiLU慢,占显存更多,但实现更简洁"""
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
    

class LayerScale(nn.Module):
    """LayerScale层缩放模块,常用于Transformer类模型中LLaMA,PaLM
    核心作用是对特征层进行逐维度的可学习缩放,提升模型训练稳定性和表达能力
    init_values: 缩放因子,默认为1e-5; inplace: 是否使用原地操作,默认为False"""
    def __init__(self, dim:int, 
                 init_values:float=1e-5, 
                 inplace:bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """ PatchDropout机制的PyTorch模块:https://arxiv.org/abs/2212.00794
    对ViT中的图像patch tokens进行随机丢弃,以增强模型泛化能力,属于正则化技术"""
    def __init__(self, prob:float=0.5,
                 exclude_first_token:bool=True)->None:
        assert 0<=prob<1, f'prob:{prob}必须在0到1之间'
        self.prob = prob
        self.exclude_first_token = exclude_first_token
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if not self.training or self.prob==0: return x
        # 判断是否包括first_token
        if self.exclude_first_token: 
            cls_tokens, x = x[:, :1], x[:, 1:]
        else: cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])
        