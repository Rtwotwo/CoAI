"""
Modified By: Redal
Date: 2025-12-03
Todo: 实现一个高度可配置,模块化的CLIP(Contrastive Language-Image Pretraining)模型,
      支持多种视觉主干(如Vision Transformer,Modified ResNet,timm模型)和文本编码器
      (包括原生Transformer和HuggingFace模型).复现OpenAI的原始CLIP架构,还扩展对LiT
      (Locked-image Tuning)等先进训练策略的支持——即冻结图像编码器,仅微调文本编码器
      以实现高效的零样本迁移.
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import copy
import logging
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial

from .utils import to_2tuple
from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import (LayerNormFp32, LayerNorm, 
                          QuickGELU, Attention,
                          VisionTransformer,
                          TextTransformer,
                          text_global_pool,
                          lock_text_tower)


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    # Custom attention block settings
    block_type: Optional[str] = None  # attention block type ('default', 'custom'), auto-selects 'custom' if any below features enabled
    qk_norm: bool = False  # apply layer norm to q and k in attention
    scaled_cosine_attn: bool = False  # use scaled cosine attention
    scale_heads: bool = False  # learnable head-specific scale applied to attention logits
    scale_attn_inner: bool = False  # apply layer norm on attention context, before output projection
    scale_attn: bool = False  # apply layer norm after full attention block
    scale_fc: bool = False  # apply layer norm in MLP block

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass 
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_mode: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    eos_id: int = 2  # only used for when pool_type == 'eos', must match tokenizer eos
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # Custom attention block settings
    block_type: Optional[str] = None  # attention block type ('default', 'custom'), auto-selects 'custom' if any custom features enabled
    qk_norm: bool = False  # apply layer norm to q and k in attention
    scaled_cosine_attn: bool = False  # use scaled cosine attention
    scale_heads: bool = False  # learnable head-specific scale applied to attention logits
    scale_attn_inner: bool = False  # apply layer norm on attention context, before output projection
    scale_attn: bool = False  # apply layer norm after full attention block
    scale_fc: bool = False  # apply layer norm in MLP block

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    """根据输入的精度字符串,返回对应的PyTorch数据类型对象,
    用于后续张量的数据类型转换cast操作"""
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    """根据输入的精度字符串precision,映射返回对应的PyTorch数据类型,
    本质是实现字符串到 PyTorch dtype 的枚举映射逻辑"""
    input_dtype = None
    if precision in ['bf16', 'pure_bf16']:
        input_dtype = torch.bfloat16
    elif precision in ['fp16', 'pure_fp16']:
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(embed_dim: int,
                        vision_cfg: CLIPVisionCfg,
                        quick_gelu: bool=False,
                        cast_dtype: Optional[torch.dtype]=None):
    """为CLIP类模型(图文对比学习模型)搭建视觉特征提取网络支持多种视觉骨干网络的灵活选择与配置
    初始化vision_tower的模块根据实际情况选择timm_model/ModifiedResNet/VisionTransformer"""
    if isinstance(vision_cfg, dict): 
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    # OpenAI使用QuickGELU作为预训练的激活函数,但是原生的GELU更快更好
    # 在timm model中也是经常使用原生的GELU进行处理而非quickGELU
    act_layer = QuickGELU if quick_gelu else nn.GELU
    # 初始化vision_tower的模块根据实际情况选择timm_model/ModifiedResNet/VisionTransformer
    if vision_cfg.timm_model_name:
        visual = TimmModel(
                    vision_cfg.timm_model_name,
                    embed_dim=embed_dim,
                    image_size=vision_cfg.image_size,
                    pool=vision_cfg.timm_pool,
                    proj=vision_cfg.timm_proj,
                    proj_bias=vision_cfg.timm_proj_bias,
                    drop=vision_cfg.timm_drop,
                    drop_path=vision_cfg.timm_drop_path,
                    patch_drop=vision_cfg.timm_drop_path,
                    pretrained=vision_cfg.timm_model_pretrained,)
    elif isinstance(vision_cfg, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
                    layers=vision_cfg.layers,
                    output_dim=embed_dim,
                    heads=vision_heads,
                    image_size=vision_cfg.image_size,
                    width=vision_cfg.width)
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat) else LayerNorm
        # 使用partial避免重复参数注册
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)
        visual = VisionTransformer( 
                    image_size=vision_cfg.image_size,
                    patch_size=vision_cfg.patch_size,
                    width=vision_cfg.width,
                    layers=vision_cfg.layers,
                    heads=vision_heads,
                    mlp_ratio=vision_cfg.mlp_ratio,
                    ls_init_value=vision_cfg.ls_init_value,
                    patch_dropout=vision_cfg.patch_dropout,
                    attentional_pool=vision_cfg.attentional_pool,
                    attn_pooler_queries=vision_cfg.attn_pooler_queries,
                    attn_pooler_heads=vision_cfg.attn_pooler_heads,
                    pos_embed_type=vision_cfg.pos_embed_type,
                    no_ln_pre=vision_cfg.no_ln_pre,
                    final_ln_after_pool=vision_cfg.final_ln_after_pool,
                    pool_type=vision_cfg.pool_type,
                    output_tokens=vision_cfg.output_tokens,
                    output_dim=embed_dim,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    block_type=vision_cfg.block_type,
                    qk_norm=vision_cfg.qk_norm,
                    scaled_cosine_attn=vision_cfg.scaled_cosine_attn,
                    scale_heads=vision_cfg.scale_heads,
                    scale_attn_inner=vision_cfg.scale_attn_inner,
                    scale_attn=vision_cfg.scale_attn,
                    scale_fc=vision_cfg.scale_fc,)
    return visual


def _build_text_tower(embed_dim: int,
                      text_cfg:CLIPTextCfg,
                      quick_gelu: bool=False,
                      cast_dtype: Optional[torch.dtype]=None):
    """构建文本编码塔(Text Encoder Tower)的核心函数
    根据text_cfg决定使用hf_model/TextTransformer"""
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    if text_cfg.hf_model_name:
        text = HFTextEncoder(
                    text_cfg.hf_model_name,
                    output_dim=embed_dim,
                    pooler_type=text_cfg.hf_pooler_type,
                    proj_type=text_cfg.hf_proj_type,
                    pretrained=text_cfg.hf_model_pretrained,
                    output_tokens=text_cfg.output_tokens)
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)
        text = TextTransformer(
                    context_length=text_cfg.context_length,
                    vocab_size=text_cfg.vocab_size,
                    width=text_cfg.width,
                    heads=text_cfg.heads,
                    layers=text_cfg.layers,
                    mlp_ratio=text_cfg.mlp_ratio,
                    ls_init_value=text_cfg.ls_init_value,
                    output_dim=embed_dim,
                    embed_cls=text_cfg.embed_cls,
                    no_causal_mask=text_cfg.no_causal_mask,
                    pad_id=text_cfg.pad_id,
                    eos_id=text_cfg.eos_id,
                    pool_type=text_cfg.pool_type,
                    proj_type=text_cfg.proj_type,
                    proj_bias=text_cfg.proj_bias,
                    output_tokens=text_cfg.output_tokens,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    block_type=text_cfg.block_type,
                    qk_norm=text_cfg.qk_norm,
                    scaled_cosine_attn=text_cfg.scaled_cosine_attn,
                    scale_heads=text_cfg.scale_heads,
                    scale_attn_inner=text_cfg.scale_attn_inner,
                    scale_attn=text_cfg.scale_attn,
                    scale_fc=text_cfg.scale_fc,)
    return text
    

class CLIP(nn.Module):
    """"""
    output_dict: torch.jit.Final[bool]
    def __in__(self, 
               embed_dim:int,
               vision_cfg: CLIPVisionCfg,
               text_cfg: CLIPTextCfg,
               quick_gelu: bool=False,
               init_logit_scale: float=np.log(1/0.07),
               init_logit_bias: Optional[float]=None,
               nonscalar_logit_scale: bool=False,
               cast_dtype: Optional[torch.dtype]=None,
               output_dict: bool=False
               )->None:
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim=embed_dim,
                                          vision_cfg=vision_cfg,
                                          quick_gelu=quick_gelu,
                                          cast_dtype=cast_dtype)
        self.text = _build_text_tower(embed_dim=embed_dim,
                                      text_cfg=text_cfg,
                                      quick_gelu=quick_gelu,
                                      cast_dtype=cast_dtype)
        self.transformer = self.text.transformer
        

