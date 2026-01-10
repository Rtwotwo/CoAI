"""
Modified By: Redal
Date: 2025-12-23
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
from utils import freeze_batch_norm_2d
try:
    import timm
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d
    from timm.layers import Mlp, to_2tuple
except ImportError as e:
    timm = None
try:
    from timm.models.helpers import group_parameters, group_modules            
except:
    raise ImportError(f'[ERROR] please run the command `pip install git+https://github.com/rwightman/pytorch-image-models`')


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
        # 配置相关模型不常用的参数,灵活适配timm模型的特征输出形式
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
            feature_ndim = 1 if not feat_size else 2
            # 如果自定义池化层
            if custom_pool:
                # 如果注意力池化使用,则去掉分类和默认池化
                assert feature_ndim == 2 , '[WARNING] 请确保feature_ndim为2!'
                self.trunk.reset_classifier(embed_dim, pool_type=pool)
            else:
                # 如果设置了池化config则关闭global_pool
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features
        # Add custom pooling to the head
        head_layers = OrderedDict()
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim
        self.head = nn.Sequential(head_layers)
    def lock(self, unlocked_groups: int=0,
             freeze_bn_stats: bool=False
             )->None:
        """锁定模块unlocked_groups保留最后n个层组不锁定"""
        if not unlocked_groups:
            for param in self.trunk.parameters():
                param.requires_grad =  False
            if freeze_bn_stats: freeze_batch_norm_2d(self.trunk)
        # NOTE:部分冻结需要最新的timm分支,且可能会发生变化
        else:
            # timm库的group_parameters(按组划分参数)和group_modules(按组划分模块)
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            # 遍历"冻结层组ID范围"内的所有组,获取组内参数并冻结
            for group_idx in range(max_layer_id+1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                # group_modules按组划分模块(反向排序,确保层组ID对应正确)
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool=True)->None:
        """设置梯度检查点以节省显存,但会减慢训练速度"""
        try: 
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning(f'[WARNING] grad checkpointing is not supported by timm model!')
    def forward_intermediates(self, x:torch.Tensor,
                              indices: Optional[Union[int, List[int]]],
                              stop_early: bool=False,
                              normalize_intermediates: bool=False,
                              intermediates_only: bool=False,
                              output_fmt: str='NCHW',
                              output_extra_tokens: bool=False,
                              )->Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """通过主干网络self.trunk获取图像的中间特征和最终特征,并支持可选的额外Token输出"""
        extra_args = {}
        # 若需要输出额外Token,给主干网络的前向传播添加return_prefix_tokens=True参数
        if output_extra_tokens: extra_args['return_extra_tokens'] = True
        trunk_output = self.trunk.forward_intermediates(x, indices=indices,
                                                        intermediates_only=intermediates_only,
                                                        norm=normalize_intermediates,
                                                        output_fmt=output_fmt,
                                                        **extra_args)
        # 将forward_intermediates的输出拆分为核心的intermediates特征和前缀prefix_token
        return_dict = {}
        intermediates = trunk_output if intermediates_only else trunk_output[1]
        if output_extra_tokens and intermediates and isinstance(intermediates[0], tuple):
            intermediates_prefix = [xi[1] for xi in intermediates]
            intermediates = [xi[0] for xi in intermediates]
            return_dict['image_intermediates_prefix'] = intermediates_prefix
        return_dict['image_intermediates'] = intermediates
        if intermediates_only: return return_dict
        # 把forward_intermediates输出的核心的特征以获取最终的图像特征
        image_features = self.trunk.forward_head(trunk_output[0])
        image_features = self.head(image_features)
        return_dict['image_features'] = image_features
        return return_dict
    def set_input_size(self, image_size: Union[int, Tuple[int, int]]):
        """在模型初始化后设置其输入图像尺寸
        image_size输入的新的图像尺寸,可由timm_model进行动态的调整输入尺寸"""
        image_size = to_2tuple(image_size)
        if hasattr(self.trunk, 'set_input_size'):
            self.trunk.set_input_size(image_size)
        else: logging.warning(f'[WARNING] timm model does not support set_input_size!')
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.trunk(x)
        x = self.head(x)
        return x
    