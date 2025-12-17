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
class CLIPConfig:
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
    