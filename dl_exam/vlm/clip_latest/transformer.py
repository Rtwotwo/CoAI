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
                 is_init_value:float=None,
                 act_layer:Callable=nn.GELU,
                 norm_layer:Callable=LayerNorm,
                 is_cross_attention:bool=False,
                 batch_first:bool=True
                 )->None:
        super().__init__()
        # 交叉注意力层实现
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, is_init_value) if is_init_value is not None else nn.Identity()
        if is_cross_attention: self.ln_1_kv = norm_layer(d_model)
        # MLP层实现
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, mlp_width)),
            ('gelu', act_layer()),
            ('c_proj', nn.Linear(mlp_width, d_model)),]))
        self.ls_2 = LayerScale(d_model, is_init_value) if is_init_value is not None else nn.Identity()
    def get_weight_dtype(self)->torch.dtype:
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
        K_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
    

class CustomResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 n_head:int,
                 mlp_ratio:float=4.0,
                 is_init_value:float=None,
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
        self.ls_1 = LayerScale(d_model, is_init_value) if is_init_value is not None else None

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, mlp_width)),
            ('gelu', act_layer()),
            # 来自 NormFormer/Foundation Transformers
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ('c_proj', nn.Linear(mlp_width, d_model)),]))
        self.ls_2 = LayerScale(d_model, is_init_value) if is_init_value is not None else None
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
                 is_init_value:float=None,
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
                    is_init_value=is_init_value,
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
                              stop_early: bool= False)_->torch.Tensor:
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
                 attention_pool: bool=False,
                 attention_pool_queries: int=256,
                 attention_pool_head: int=8,
                 output_dim: int=512,
                 patch_dropout: float=0.,
                 is_init_values:float=None,
                 act_layer: Callable=nn.GELU,
                 norm_layer: Callable=LayerNorm,
                 batch_first: bool=False,
                 )->None:
        super().__init__()
        self.width = width
        self.layers = layers
