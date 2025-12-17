# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------
"""
Modified By: Redal
Date: 2025-12-03
Todo: 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import math

from setuptools import Require
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


"""
class Dropout(nn.Module):
    def __init__(self, p:float=0.1,
                 training:bool=True
                 )->None:
        super().__init__()
        assert p<0 or p>=1, f'Dropout的p值应该在[0, 1)之间'
        self.p = p
        self.training = training
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.training:
            # keep_p = 1 - slef.p
            # mask = (torch.rand_like(x) < keep_p).float() / keep_p
            # return x * mask
            mask = torch.rand_like(x) > self.p
            x = (x * mask.float()) / (1 - self.p)
            return x
        return x
"""
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
    def forward(self, x:torch.Tensor, 
                attn_mask:Optional[torch.Tensor]
                )->torch.Tensor:
        N, L, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.view(N, L, self.num_heads, -1).transpose(1, 2)
        k = k.view(N, L, self.num_heads, -1).transpose(1, 2)
        v = v.view(N, L, self.num_heads, -1).transpose(1, 2)

        # 注意力掩码,在计算注意力时只关注有效位置,忽略无效位置
        if attn_mask is not None:
            if attn_mask.ndim == 3:
                # 此模块适用于(L，L)或(N，num_heads，L，L)掩码
                attn_mask = attn_mask.reshape(N, self.num_heads, L, L)
            elif attn_mask == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask = new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            else: attn_mask = attn_mask.to(dtype=q.dtype)

        # 两种不同的注意力机制计算逻辑
        if self.scaled_cosine is not None:
            # 缩放的余弦相似度注意力
            attn = torch.bmm( # 批量计算归一化的注意力矩阵
                F.normalize(q, dim=-1),
                F.normalize(k, dim=-1).transpose(1, 2))
            # 可学习的参数,用于控制注意力分数的缩放
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn * logit_scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            # 不使用logit_scale的注意力计算
            q = self.ln_q(q)
            k = self.ln_k(k)
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p = self.attn_drop.p if self.training else 0.)
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None: attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        # [N, num_heads, L, head_dim]->[N, L, C]
        # 多头注意力MHA模块的输出处理阶段
        if self.head_scale: x = x * self.head_scale
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.ln_inner(x)
        self.out_proj(x)
        self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    """注意力池化层,用于将变长序列转换为固定长度的表示
    通过注意力机制动态聚合序列信息,能更好地捕捉序列中的关键内容"""
    def __init__(self, 
                 d_model: int,
                 context_dim: int,
                 n_head: int=8,
                 n_queries: int=256,
                 norm_layer: Callable=LayerNorm
                 )->None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, 
                                          vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_head: int,
                 mlp_ratio:float=4.0,
                 ls_init_value:float=None,
                 act_layer:Callable=nn.GELU,
                 norm_layer:Callable=LayerNorm,
                 is_cross_attention:bool=False,
                 batch_first:bool=True
                 )->None:
        super().__init__()
        # 交叉注意力层实现
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention: self.ln_1_kv = norm_layer(d_model)
        # MLP层实现
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, mlp_width)),
            ('gelu', act_layer()),
            ('c_proj', nn.Linear(mlp_width, d_model)),]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
    def get_weight_dtype(self)->torch.dtype:
        """获取模型中MLP层的全连接层c_fc的权重数据类型"""
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype
    def attention(self, q_x:torch.Tensor,
                  k_x: Optional[torch.Tensor]=None,
                  v_x: Optional[torch.Tensor]=None,
                  attn_mask: Optional[torch.Tensor]=None,
                  )->torch.Tensor:
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]
    def forward(self, q_x: torch.Tensor,
                k_x: Optional[torch.Tensor]=None,
                v_x: Optional[torch.Tensor]=None,
                attn_mask: Optional[torch.Tensor]=None,
                )->torch.Tensor:
        k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
    

class CustomResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 n_head:int,
                 mlp_ratio:float=4.0,
                 ls_init_value:float=None,
                 act_layer: Type[nn.Module]=nn.GELU,
                 norm_layer: Type[nn.Module]=LayerNorm,
                 qk_norm: bool=False,
                 scale_cosine_attn:bool=False,
                 scale_heads:bool=False,
                 scale_attn_inner:bool=False,
                 scale_attn: bool=False,
                 scale_fc: bool=False,
                 batch_first:bool=False,
                 )->None:
        super().__init__()
        assert batch_first, f'batch_first必须为True,当前的batch_first:{batch_first}'
        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(d_model,
                              n_head,
                              qk_norm=qk_norm,
                              scale_cosine=scale_cosine_attn,
                              scale_heads=scale_heads,
                              inner_norm=scale_attn_inner,
                              norm_layer=norm_layer)
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else None

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, mlp_width)),
            ('gelu', act_layer()),
            # 来自 NormFormer/Foundation Transformers
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ('c_proj', nn.Linear(mlp_width, d_model)),]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else None
    def get_weight_type(self,)->torch.dtype:
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.weight.int8_original_dtype
        return self.mlp.c_fc.weight.dtype
    def forward(self, x:torch.Tensor, 
                attn_mask:Optional[torch.Tensor]=None
                )->torch.Tensor:
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomTransformer(nn.Module):
    def __init__(self,
                 width:int,
                 layers:int,
                 heads:int,
                 mlp_ratio:float=4.0,
                 ls_init_value:float=None,
                 act_layer: Type[nn.Module]=nn.GELU,
                 norm_layer: Type[nn.Module]=LayerNorm,
                 batch_first:bool=True,
                 block_types:Union[str, List[str]]='CustomResidualAttentionBlock',
                 )->None:
        super().__init__()
        self.width = width
        self.layers = layers
        # batch_first优先则transfromer的形状为[N, L, D]
        self.batch_first = batch_first
        self.grad_checkpointing = False
        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers, f"block_types的长度{len(block_types)}必须等于layers的长度{layers}"
        def _create_block(bt: str)->CustomResidualAttentionBlock:
            """根据给定的block类型创建对应的块实例"""
            if bt == "CustomResidualAttentionBlock":
                return CustomResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first)
            else:
                assert False, f'block_type: {bt}不被支持!'
        # 创建blocks
        self.resblocks = nn.ModuleList([_create_block(bt) for bt in block_types])
    def get_cast_dtype(self,)->torch.dtype:
        """获取Transformer中残差块的权重数据类型"""
        return self.resblocks[0].get_weight_type()
    def forward_intermediates(self, x:torch.Tensor,
                              attn_mask: Optional[torch.Tensor]=None,
                              indices: Optional[Union[int, List[int]]]=None,
                              stop_early: bool= False)->torch.Tensor:
        """对输入张量依次通过多个残差块，并根据指定索引收集中间层输出
        最终返回输出结果和所有指定的中间层特征"""
        # 计算需要收集中间结果的索引和最大索引
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices, )
        # 把输入张量从 (N, L, D) 转换为 (L, N, D)，并保证内存连续
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()
        intermediates = []
        if torch.jit.is_scripting() or not stop_early:
            blocks = self.resblocksblocks
        else: blocks = self.blocks[:max_index +1]
        for i, blk in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, attn_mask)
            else: x = blk(x, attn_mask)
            # 如果当前块的索引在需要收集的索引列表中
            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if self.batch_first else x)
        # 形状[L, N, D] -> [N, L, D]
        if not self.batch_first:
            x = x.transpose(0, 1) 
        return x, intermediates
    def prune_intermediate_layers(self, indices:Union[int, List[int]]=1)->None:
        """根据指定的索引裁剪残差块,只保留需要的部分,并返回需要收集的中间层索引"""
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index +1]
        return take_indices
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor]=None
                )->torch.Tensor:
        # 改变形状[N, L,, D] -> [L, N, D]
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()
        for r in self.resblocks:
            # 
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # 如果启用了梯度检查点且未在 TorchScript 模式下，使用 torch.utils.checkpoint对残差块r进行前向计算
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else: x = r(x, attn_mask=attn_mask)
        # 改变形状[L, N, D] -> [N, L,
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()
        return x


class Transformer(nn.Module):
    def __init__(self, 
                 width:int,
                 layers:int,
                 heads: int,
                 mlp_ratio:float=4.0,
                 ls_init_value:float=None,
                 act_layer: Type[nn.Module]=nn.GELU,
                 norm_layer: Type[nn.Module]=LayerNorm,
                 batch_first: bool=False,
                 block_type: Optional[str]=None,
                 qk_norm: bool=False,
                 scaled_cosine_attn: bool=False,
                 scale_heads: bool=False,
                 scale_attn_inner: bool=False,
                 scale_attn: bool=False,
                 scale_fc: bool=False,
                 )->None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False
        # 如果启用了任何自定义功能,则自动选择自定义块
        if block_type is None:
            if any([qk_norm, scaled_cosine_attn, scale_heads, 
                    scale_attn_inner, scale_attn, scale_fc]):
                block_type = 'custom'
            else: block_type = 'default'
        if block_type == 'custom':
            self.resblocks = nn.ModuleList([
                CustomResidualAttentionBlock(
                    d_model = width,
                    n_head = heads,
                    mlp_ratio = mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    qk_norm = qk_norm,
                    scale_cosine_attn=scaled_cosine_attn,
                    scale_heads = scale_heads,
                    scale_attn_inner = scale_attn_inner,
                    scale_attn = scale_attn,
                    scale_fc = scale_fc,
                    batch_first=batch_first,)
                for _ in range(layers)])
        else:
            self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(
                    d_model = width,
                    n_head = heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,)
                for _ in range(layers)])
    def get_cast_dtype(self,)->torch.dtype:
        """获取模型中MLP层的全连接层c_fc的权重数据类型"""
        return self.resblocks[0].get_weight_dtype()
    def forward_intermediates(self, x: torch.Tensor,
                              attn_mask: Optional[torch.Tensor]=None,
                              indices: Optional[Union[int, List[int]]]=None,
                              stop_early: bool=False):
        """支持指定任意层提取中间特征,适配不同的下游任务需求"""
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        if not self.batch_first:
            # 将形状[N, L, D]转换为[L, N, D]
            x = x.transpose(0, 1).contiguous()
        intermediates = []
        if torch.jit.is_scriping or not self.grad_checkpointing:
            blocks = self.resblocks
        else: blocks = self.resblocks[:max_index + 1]
        # 遍历残差块执行前向计算,并提取中间特征
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
            else: 
                x = blk(x, attn_mask)
            # 若当前层索引在take_indices中,提取并保存特征
            if i in take_indices: intermediates.append(x.transpose(0, 1) if not self.batch_first else x)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, intermediates
    def prune_intermediate_layers(self, indices:Union[int, List[int]]=1):
        """修剪指定中间结果不需要的层"""
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index+1]
        return take_indices
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor]=None,
                )->torch.Tensor:
        if not self.batch_first: x = x.transpose(0, 1).contiguous()
        for r in self.resblocks:
            # 以增加计算量为代价,大幅降低训练过程中的显存GPU内存占用
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else: x = r(x, attn_mask=attn_mask)
        if not self.batch_first: x = x.transpose(0, 1)
        return x
    

# --------------------------------------------------------------------------------------
# 下面的代码共同构成了CLIP(Contrastive Language–Image Pretraining)模型的
# 双塔结构(Dual-tower architecture),并扩展支持了多种变体(如CoCa,带Attentional Pooling的ViT)
# --------------------------------------------------------------------------------------
def _expand_token(token, batch_size:int):
    """对输入的张量 token 进行维度扩展,使其适配指定的批量大小"""
    return torch.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]
    def __init__(self, image_size: int,
                 patch_size: int, 
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 ls_init_value: float = None,
                 attentional_pool: bool = False,
                 attn_pooler_queries: int = 256,
                 attn_pooler_heads: int = 8,
                 output_dim: int=512,
                 patch_dropout: float=0.,
                 no_ln_pre: bool=False,
                 pos_embed_type: str="learnable",
                 pool_type: str='tok',
                 final_ln_after_pool: bool=False,
                 act_layer: Callable=nn.GELU,
                 norm_ayer: Callable=LayerNorm,
                 output_tokens: bool=False,
                 block_type: Optional[str]=None,
                 qk_norm: bool=False,
                 scaled_cosine_attn: bool=False,
                 scale_heads: bool=False,
                 scale_attn_inner: bool=False,
                 scale_attn: bool=False,
                 scale_fc: bool=False,
                 ) -> None:
        super().__init__()
        assert pool_type in ['tok', 'avg', 'none']
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grad_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool
        self.output_dim = output_dim
        # 利用卷积的核尺寸=步长特性完成无重叠分块,并同步完成通道维度的映射
        self.conv1 = nn.Conv2d(in_channels=3,
                            out_channels=width,
                            kernel_size=patch_size,
                            stride=patch_size,
                            bias=False)
        # 类别编码以及位置编码class and positional embedding
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grad_szie[0] * self.grad_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # 选择使用sin-cos embedding
            assert self.grad_size[0] == self.grad_size[1], \
                f"sin-cos embedding仅支持方块分块,当前分块大小为{self.grad_size}"
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grad_size[0] * self.grad_size[1] +1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grad_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else: raise ValueError(f"[WARNING] 未知的位置编码类型: {pos_embed_type}")
        # 设置patch_dropout=0.则PatchDropout不生效
        self.patch_dropout = nn.Dropout(patch_dropout) if patch_dropout>0 else nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_ayer(width)
        self.transformer = Transformer(
                            width = width,
                            layers=layers,
                            heads = heads,
                            mlp_ratio = mlp_ratio,
                            ls_init_value = ls_init_value,
                            act_layer = act_layer,
                            norm_ayer = norm_ayer,
                            block_type=block_type,
                            qk_norm = qk_norm,
                            scaled_cosine_attn = scaled_cosine_attn,
                            scale_heads = scale_heads,
                            scale_attn_inner = scale_attn_inner,
                            scale_attn = scale_attn,
                            scale_fc = scale_fc,)
        # 根据传入的attentional_pool参数来决定是否启用,如何配置注意力池化层
        # 并完成后续的归一化和投影层初始化
        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attentional_pool = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    # 初始化主注意力池化层：使用指定的查询数(attn_pooler_queries)
                    self.attn_pool = AttentionalPooler(
                            d_model = output_dim,
                            context_dim = width,
                            n_head = attn_pooler_heads,
                            n_queries = attn_pooler_queries,)
                    # 初始化对比学习用的注意力池化层:固定查询数为1,通常用于生成单向量的对比特征
                    self.attn_pool_contrastive = AttentionalPooler(
                            d_model = output_dim,
                            context_dim = width,
                            n_head = attn_pooler_heads,
                            n_queries = 1,)
                else: assert False, f'[WARNING] 未知的注意力池化类型attentinoal_pool: {attentional_pool}'
            else: # 仅启用基础注意力池化,保留普通池化的配置
                self.attn_pool = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                            d_model = output_dim, 
                            context_dim = width,
                            n_head = attn_pooler_heads,
                            n_queries = attn_pooler_queries,)
                self.attn_pool_contrastive = None
            # 注意力池化的输出维度固定为output_dim
            pool_dim = output_dim 
        # 注意力池化禁用分支(attentional_pool为假False)
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type
        self.ln_post = norm_ayer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))
        self.init_parameters()
    def init_parameters(self, ):
        """初始化模型参数,包括Transformer层、注意力池化层、归一化层和投影层
        TODO OpenAI的CLIP Model并没有初始化VisionTransformer的参数,需要有实验结果好坏决定
        基本使用normal_正态分布进行初始化(均值0,标准差 self.scale)"""
        nn.init.normal_(self.class_embedding, std=self.scale)
        nn.init.normal_(self.positional_embedding, std=self.scale)
        # 计算标准差并初始化Transformer层
        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std  = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=self.scale)
    def lock(self, unlocked_groups: int=0, freeze_bn_stats: bool=False):
        """冻结模型大部分参数的梯度计算(即固定参数不更新),仅解锁指定数量的参数组让其可训练"""
        # 全局冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        if unlocked_groups > 0:
            # 将模型的不同模块拆分为有序的参数组,分组逻辑贴合模型的结构层级
            # 前置模块→transformer中间残差块→transformer末尾模块→输出层
            groups = [  [self.conv1, self.class_embedding, 
                        self.positional_embedding, self.ln_pre],
                        *self.transformer.resblocks[:-1],
                        [self.transformer.resblocks[-1],self.ln_post],
                        self.proj,]
            def _unlock(x):
                """递归解锁参数组x中的所有参数,支持嵌套结构"""
                # 若输入是序列(列表/元组等),递归遍历内部元素
                if isinstance(x, Sequence):
                    for g in x: _unlock(g)
                else:
                    # x是单个模块/参数,则直接开启参数梯度
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    # 若x是nn.Module,则遍历解锁其所有参数
                    else:
                        for p in x.parameters():
                            p.requires_grad = True
            _unlock(groups[-unlocked_groups:])
    @torch.jit.ignore
    def set_grad_checkpointing(self, enabled: bool=True):
        """使能transformer的梯度检查点grad_checkpointing"""
        self.transformer.grad_checkpointing = enabled
    @torch.jit.ignore
    def no_weight_decay(self):
        """返回transformer中所有参数的名称列表,用于配置优化器时排除权重衰减"""
        # 对于timm库,一维参数(如 logit_scale、logit_bias、层归一化/
        # 批归一化的scale参数、各类偏置bias)默认会被排除在权重衰减之外
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd
    def _global_pool(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """pooled全局池化特征, tokens局部特征"""
        if self.pool_type == 'avg': # 通常指的是平均池化
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok': # token池化,通常是 cls_token
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled, tokens = x, x
        return pooled, tokens
    def _embeds(self, x:torch.Tensor)->torch.Tensor:
        """将输入的图像张量转化为适用于Transformer处理的序列张量"""
        # [batch_size, channels, height, width]->[batch_size, dim, grid, grid]
        x = self.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [*, dim, grid*grid]
        x = x.permute(0, 2, 1)
        # 类型和位置编码,x的形状是[*, grid ** 2 + 1, width]
        # 拼接class_embedding初始一维张量[width],需要使用_expand_token扩展
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        # 使用PatchDropout对输入进行随机丢弃
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        return x
    def _pool(self, x: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """对输入的张量x进行不同策略的池化处理,最后返回
        pooled(池化后的核心特征)和tokens(池化过程中保留的特征令牌/序列特征)"""
        if self.attn_pool is not None:
            # 存在对比注意力池化attn_pool_contrastive
            # 但是这是未完全测试的实验性逻辑(WIP)
            if self.attn_pool_contrastive is not None:
                x = self.ln_post(x)
                if self.pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else: 
                    assert self.pool_type == 'cascade', \
                        f'[WARNING] pool_type必须为cascade!'
                    pooled = self.attn_pool_contrastive(x)
            # 原版OpenAI的CoCa的CLIP的实现
            else:
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        # 全局池化(_global_pool)并根据final_ln_after_pool决定层归一化的时机
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        return pooled, tokens
    def forward_intermediates(self, x: torch.Tensor,
                              indices: Optional[Union[int, List[int]]]=None,
                              stop_early: bool = False,
                              normalize_intermediates: bool=False,
                              intermediates_only: bool=False,
                              output_fmt: str="NCHW",
                              output_extra_tokens: bool=False
                              )->Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """indices如果是整数,则取最后n个块;如果是 None,则取所有块;如果是序列,则选择匹配的索引
        stop_early当遇到最后一个所需的中间结果时,停止对块的迭代
        intermediates_only仅返回中间特征;normalize_intermediates对所有中间结果应用最终的归一化层
        output_fmt中间特征输出的形状;output_extra_tokens返回额外的前缀类别标记"""
        assert output_fmt in ["NCHW", "NLC"], f'[WARNING] output_fmt必须是"NCHW"或"NLC"!'
        reshape = output_fmt == 'NLC'
        
    