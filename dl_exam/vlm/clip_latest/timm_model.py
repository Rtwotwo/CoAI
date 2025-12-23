"""
Modified By: Redal
Date: 2025-12-03
Todo: 使用timm_model为CLIP封装一个vision tower模型,同时适配
      各类timm原生的模型集成到CLIP模型中
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import logging
from collections import OrderedDict
from typing import Tuple, Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import freeze_batch_norm_2d
try:
    import timm
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d
    from timm.layers import Mlp, to_2tuple
except ImportError as e:
    timm = None


class TimmModel(nn.Module):
    """针对Timm的模型的适配器"""
    def __init__(self, model_name:str,
                 embed_dim: int,
                 image_size: Union[int, Tuple[int, int]]=None,
                 pool:str = 'avg',
                 proj:str = 'linear',
                 proj_bias: bool = False,
                 drop: float = 0.,
                 drop_path: Optional[float] = None,
                 patch_drop: Optional[float] = None,
                 pretrained: bool = False,
                 )->None:
        super().__init__()
        if timm is None: raise ImportError(f'[ERROR] timm is not installed!')
        self.image_size = to_2tuple(image_size)
        # 配置相关模型不常用的参数
        timm_kwargs = {}
        if drop_path is not None: timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None: timm_kwargs['patch_drop_rate'] = patch_drop
        custom_pool = pool in ['abs_attn', 'rot_attn']
        if proj: assert proj in ['linear', 'mlp', 'none'], f'[WARNING] proj must be one of [linear, mlp, none], but got {proj}'
        extra_proj = proj in ['linear', 'mlp']
        if not extra_proj and not custom_pool:
            # 如果没有使用特别的投影或者池化,使用分类器的projection
            # 如果projection是None,则会跳过创建timm_model
            self.trunk = timm.create_model(
                                model_name,
                                num_classes = embed_dim,
                                global_pool = pool,
                                pretrained=pretrained,
                                **timm_kwargs)
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                                model_name,
                                pretrained=pretrained,
                                **timm_kwargs)
            feat_size = self.trunk.patch_embed.num_patches
            feat_dim = 1 if not feat_size else 2
            # 如果自定义池化层
            if custom_pool:
                
        

