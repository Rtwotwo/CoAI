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
        # 计算公式: LayerNorm(x)=weights·(X-E(x)) / sqrt(Var(x)+eps) + bias
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
    """PatchDropout机制的PyTorch模块:https://arxiv.org/abs/2212.00794
    对ViT中的图像patch tokens进行随机丢弃,以增强模型泛化能力,属于正则化技术"""
    def __init__(self, prob:float=0.5,
                 exclude_first_token:bool=True)->None:
        assert 0<=prob<1, f'prob:{prob}必须在0到1之间'
        self.prob = prob
        self.exclude_first_token = exclude_first_token
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if not self.training or self.prob==0: return x
        # 判断是否包括first_token,分离cls_token
        if self.exclude_first_token: 
            cls_tokens, x = x[:, :1], x[:, 1:]
        else: cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])
        # 随机丢弃部分patch
        batch = x.size()[0]
        num_tokens = x.size()[1]
        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]
        keep_prob = 1 - self.prob
        num_keep_patches = max(1, int(num_tokens * keep_prob))
        # 随机生成一个概率矩阵以生成保留patch索引
        rand = torch.randn(batch, num_tokens)
        patch_keep_indices = rand.topk(num_keep_patches, dim=-1).indices
        x = x[batch_indices, patch_keep_indices]
        # 判断是否保留first_token,即是cls_token
        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


class Attention(nn.Module):
    """灵活扩展的多头注意力Multi-Head Attention的PyTorch实现,
    整合了多个主流注意力改进技术核心用于Transformer类模型的特征交互"""
    def __init__(self,
                # 初始化基本attention参数
                 dim: int,
                 num_heads: int=8,
                 qkv_bias: bool=True,
                # 控制注意力计算变体参数
                 qk_norm: bool=False,
                 scaled_cosine: bool=False,
                # 缩放以及归一化相关参数
                 scale_heads: bool=False,
                 inner_norm: bool=False,
                 logit_scale_max:float=math.log(1. / 0.01),
                 norm_layer: Type[nn.Module]=LayerNormFp32,
                 attn_drop: float=0.,
                 proj_drop: float=0.,
                 )->None:
        super().__init__()
        assert not (scaled_cosine and qk_norm), f"scaled_cosine:{scaled_cosine}和qk_norm:{qk_norm}不能同时被激活"
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, f"输入维度{dim}不能被{num_heads}整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.use_fsdpa = hasattr(nn.functional, "scaled_dot_product_attention")

        # 保持in_proj的这种形式,而非nn.Linear以匹配原始的权重方案
        self.in_proj_weight = nn.Parameter(torch.randn(3*dim, dim) * self.scale)
        if qkv_bias: self.in_proj_bias = nn.Parameter(torch.zeros(3*dim))
        else: self.in_proj_bias = None
        # qk_norm归一化源自https://arxiv.org/abs/2106.04560,且与其他QK归一化理念相关
        if qk_norm: 
            self.ln_q = norm_layer(self.head_dim)
            self.ln_k = norm_layer(self.head_dim)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        # 缩放余弦注意力(来自Swin TransformerV2,https://arxiv.org/abs/2111.09883)
        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))))
        else: self.logit_scale = None
        self.atte_drop = nn.Dropout(attn_drop)
        # 每头注意力对数概率缩放(源自NormFormer,https://arxiv.org/abs/2110.09456)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((self.num_heads, 1, 1)))
        else: self.scale_heads = None
        # 注意力对数在最终投影前的归一化
        # 其起源可能是(基础Transformer, https://arxiv.org/abs/2210.06423)中的Sub-LN
        if inner_norm: self.ln_inner = norm_layer(dim)
        else: self.ln_inner = nn.Identity()
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        N, L, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.view(N, L, self.num_heads, -1).transpose(1, 2)
        k = k.view(N, L, self.num_heads, -1).transpose(1, 2)
        v = v.view(N, L, self.num_heads, -1).transpose(1, 2)
        
        
        