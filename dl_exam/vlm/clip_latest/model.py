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
    """为CLIP类模型(图文对比学习模型)搭建视觉特征提取网络
    支持多种视觉骨干网络的灵活选择与配置"""
    if isinstance(vision_cfg, dict): 
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    # OpenAI使用QuickGELU作为预训练的激活函数,但是原生的GELU更快更好
    # 在timm model中也是经常使用原生的GELU进行处理而非quickGELU
    act_layer = QuickGELU if quick_gelu else nn.GELU
    if vision_cfg.timm_model_name:
        visual = TimmModel
