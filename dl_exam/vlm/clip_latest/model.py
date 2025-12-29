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
        # 初始化text的模型参数
        self.transformer = self.text.transformer
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.token_embedding = self.text.token_embedding
        self.positional_embedding = self.text.positional_embedding
        self.ln_final = self.text.ln_final
        self.text_projection = self.text.text_projection
        self.text_pool_type = self.text.pool_type
        self.test_eos_id = self.text.eos_id
        # 注册模型的非可训练缓冲区,不参与梯度更新
        self.register_buffer('attn_mask', self.text.attn_mask, persistent=False)
        # 对比学习对数尺度/偏置参数初始化
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_scale is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else: self.logit_bias = None
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        """按照LiT锁定图像塔-https://arxiv.org/abs/2111.07991"""
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)
    def lock_text_tower(self, unlocked_layers:int=0, freeze_layer_norm:bool=False):
        """冻结text_tower的部分参数"""
        assert freeze_layer_norm, "[WARNING] LayerNorm像其他权重一样处理!"
        lock_text_tower( unlocked_layers=unlocked_layers)
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable:bool=True):
        """激活visual和text的梯度检查点"""
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing(enable)
    @torch.jit.ignore()
    def no_weight_decay(self,):
        no_wd = {'positional_embedding'}
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.'+n)
        return no_wd
    def encode_image(self, image, normalize:bool=False):
        """将图像转换为具有语义信息的固定维度向量"""
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features
    def encode_text(self, text, normalize:bool=False):
        """将文本转换为具有语义信息的固定维度向量"""
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype) # [N, L, C]
        # 将预定义的位置嵌入positional_embedding与词嵌入相加,为每个token注入位置信息
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = text_global_pool(x, text, self.text_pool_type, eos_token_id=getattr(self, 'text_eos_id', None))
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x
    def get_logits(self, image, text):
        """量化图像image和文本text之间的语义相似度,最终
        返回图像到文本、文本到图像的两组匹配得分"""
        # normalize表示对输出的特征向量进行L2归一化,归一化后特征向量的模长为1
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        # image_features[N, D], text_featrues[M, D]得到相似性[N, M]
        # logit_scale做指数运算,作为相似度缩放因子,最终将余弦相似度映射为更高区分度的匹配得分
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            # 基础匹配得分进行全局或局部偏移修正
            image_logits += self.logit_bias
        text_logits = image_logits.T # text_logits[M, N]
        return image_logits, text_logits
    def forward_intermediates(self,image:Optional[torch.Tensor]=None,
                              text:Optional[torch.Tensor]=None,
                              image_indices:Optional[Union[int, List[int]]]=None,
                              text_indices:Optional[Union[int, List[int]]]=None,
                              stop_early:bool=False,
                              normalize:bool=False,
                              normalize_intermediates:bool=False,
                              intermediates_only:bool=False,
                              image_output_fmt:str='NCHW',
                              image_output_extra_tokens:bool=False,
                              text_output_fmt:str='NLC',
                              text_output_extra_tokens:bool=False,
                              output_logits:bool=False,
                              output_logit_scale_bias:bool=False,
                              )->Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """输入图像张量和/或文本张量,不仅能输出最终的图像特征、文本特征
        (及可选的匹配logits),还能返回模型前向传播过程中的中间层特征"""
        output = {}
        if intermediates_only:
            # 只要intermediates则不进行归一化
            normalize = False
            output_logits = False
        if output_logits:
            assert image is not None and text is not None, \
                f'[WARNING] 请注意输入的image和text的张量均需要存在!'
        # 进行image和text的前向传播intermediates,并返回中间层特征
        if image is not None:
            image_output = self.visual.forward_intermediates(image,
                                                             indices=image_indices,
                                                             stop_early=stop_early,
                                                             normalize_intermediates=normalize_intermediates,
                                                             intermediates_only=intermediates_only,
                                                             output_fmt=image_output_fmt,
                                                             output_extra_tokens=image_output_extra_tokens)
            if normalize and 'image_features' in image_output:
                image_output['image_features'] = F.normalize(image_output['image_features'], dim=-1)
            # 将image_out中的key:value批量添加到output中
            output.update(image_output)
        if text is not None:
            cast_dtype = self.transformer.get_cast_dtype()
            x = self.token_embedding(text).to(cast_dtype)
            x = x + self.positional_embedding.to(cast_dtype)
            x, intermediates = self.transformer.forward_intermediates(x,
                                                            attn_mask=self.attn_mask,
                                                            indices=text_indices)
            if normalize_intermediates:
                intermediates = [self.ln_final[xi] for xi in intermediates]
            # NOTE 模型不支持分类的tokens否存在,因此需要将分类token的输出从intermediates中删除
            output['text_intermediates'] = intermediates
            if intermediates_only:
                x = self.ln_final(x) # [N, L, C]
                x = text_global_pool(x, text, self.text_pool_type, eos_token_id=getattr(self, 'text_eos_id', None))
                if self.text_projection is not None:
                    if isinstance(self.text_projection, nn.Linear):
                        x = self.text_projection(x)
                    else:
                        x = x @ self.text_projection
                if normalize: x = F.normalize(x, dim=-1)
                output['text_features'] = x
        # 如果输出对数几率的变量,输出image/text的对数几率和偏置数据
        logit_scale_exp = self.logit_scale_exp() if output_logits or output_logit_scale_bias else None
        if output_logits:
            image_logits = logit_scale_exp * output['image_features'] @ output['text_features']
            if self.logit_bias is not None: image_logits += self.logit_bias # 添加偏置弥补位移差
            text_logits = image_logits.T
            output['image_features'] = image_logits
            output['text_features'] =  text_logits
        if output_logit_scale_bias:
            output['logit_scale'] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias
        return output
    def forward(self, image: Optional[torch.Tensor]=None,
                text:Optional[torch.Tensor]=None):
        """核心是将输入的图像和文本张量进行编码,并返回最终的图像特征和文本特征即:
        image_features, text_features, logit_scale, logit_bias"""
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            out_dict = {
                'image_features': image_features,
                'text_features': text_features,
                'logit_scale': self.logit_scale.exp()}
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()
                
        
class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]
    def __init__(self, 
                 embed_dim:int,
                 vision_cfg:CLIPVisionCfg,
                 text_cfg:CLIPTextCfg,
                 quick_gelu:bool=False,
                 init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[bool]=False,
                 nonscalar_logit_scale: bool=False,
                 cast_dtype: Optional[torch.dtype]=None,
                 output_dict: bool=False
                 )->None:
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text =  _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        # 根据输入参数的状态,动态初始化PyTorch中的可训练参数
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else: None
    def lock_image_tower(self, unlocked_groups:int=0, freeze_bn_stats:bool=True):
        """lock image tower as per LiT - https://arxiv.org/abs/2111.07991"""
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)
    def lock_text_tower(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        """冻结文本编码塔的部分"""
        self.text.lock(unlocked_layers=unlocked_layers, freeze_layer_norm=freeze_layer_norm)
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable=enable)
        self.text.set_grad_checkpointing(enable=enable)
    @torch.jit.ignore
    def no_weight_decay(self,):
        """获取模型中不需要进行权重衰减的参数名称"""
        no_wd = set()
        if hasattr(self.visual, 'no _weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.', n)
        if hasattr(self.text, 'no_weight_decay'):
            for n in self.text.no_weight_decay():
                no_wd.add('text.', n)
        return no_wd
    def encode_image(self, image, normalize:bool=False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else  features
    def encode_text(self, text, normalize:bool=False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features
    def get_logits(self, image, text):
        """获取图像和文本之间的对数几率"""
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits
    def forward_intermediates(self, image: Optional[torch.Tensor]=None,
                              text: Optional[torch.Tensor]=None,
                              image_indices: Optional[Union[int, List[int]]]=None,
                              text_indices: Optional[Union[int, List[int]]]=None,
                              stop_early: bool=False,
                              normalize: bool=True,
                              normalize_intermediates: bool=False,
                              intermediates_only:bool=False,
                              image_output_fmt: str='NCHW',
                              image_output_extra_tokens: bool=False,
                              text_output_fmt: str='NLC',
                              text_output_extra_tokens: bool=False,
                              output_logits: bool=False,
                              output_logit_scale_bias: bool=False
                              )->Dict[str, Union[torch.Tensor,  List[torch.Tensor]]]:
        """indices如果是整数,则取最后n个块;如果是 None,则取所有块;如果是序列,则选择匹配的索引
        stop_early当遇到最后一个所需的中间结果时,停止对块的迭代
        indices/stop_early灵活筛选中间层,避免冗余计算,适用于需要多层特征融合的场景
        intermediates_only仅返回中间特征;normalize_intermediates对所有中间结果应用最终的归一化
        output_fmt中间特征输出的形状;output_extra_tokens返回额外前缀类别标记"""
        output = {}
        if intermediates_only:
            normalize = False
            output_logits = False
        if output_logits: 
            assert image is not None and text is None, \
                f'[WARNING]  output_logits=True requires image and text inputs.'
        # 获取图像image和文本text的特征编码,并且嵌入到output
        if image is not None:
            image_output = self.visual.forward_intermediates(image,
                                                            indices=image_indices,
                                                            stop_early=stop_early,
                                                            normalize_intermediates=normalize_intermediates,
                                                            intermediates_only=intermediates_only,
                                                            output_fmt=image_output_fmt,
                                                            output_extra_tokens=image_output_extra_tokens,)
            if normalize and "image_features" in image_output:
                image_output['image_features'] = F.normalize(image_output['image_features'], dim=-1)
            output.update(image_output)
        if text is not None:
            text_output = self.text.forward_intermediates(text,
                                                        indices=text_indices,
                                                        stop_early=stop_early,
                                                        normalize_intermediates=normalize_intermediates,
                                                        intermediates_only=intermediates_only,
                                                        output_fmt=text_output_fmt,
                                                        output_extra_tokens=text_output_extra_tokens,)
            if normalize and 'text_features' in text_output:
                text_output['text t_features'] = F.normalize(text_output['text_features'], dim=-1)
                output.update(text_output)
        # 获取文本text和图像image之间的对数几率和bias
        logit_scale_exp = self.logit_scale.exp() if self.output_logits or output_logit_scale_bias else None
        if output_logits:
            image_logits = logit_scale_exp * output['image_features'] @ output['text_features'].T
            if self.logit_bias is not None:
                image_logits += self.logit_bias
            text_logits = image_logits.T
            output['image_logits'] = image_logits
            output['text_logits'] = text_logits
        if output_logit_scale_bias:
            output['logit_scale'] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias
        return output
    def forward(self, image:Optional[torch.Tensor]=None,
                text:Optional[torch.Tensor]=None
                )->torch.Tensor:
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            out_dict = {
                'image_features': image_features,
                'text_features': text_features,
                'logit_scale': self.logit_scale.exp()}
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), 
        return image_features, text_features, self.logit_scale.exp()


#  CLIP模型(图文匹配模型)的工具函数集合,核心用于
# 模型的权重转换、加载、兼容适配、优化与配置管理
def convert_weights_to_lp(model:nn.Module, dtype=torch.float16):
    """将CLIP模型中可适用的参数转换为低精度格式
    目的是减少模型显存占用,提升推理/训练速度"""
    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None: l.bias.data = l.bias.data.to(dtype)
        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                          'in_proj_bias', 'bias_k', 'bias_v']:
                tensor = getattr(l, attr, None)
                tensor.data = tensor.data.to(dtype)
        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, 'text_projection', None)
            if attr is not None: attr.data = attr.data.to(dtype)
        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, 'proj', None)
            if attr is not None:
                attr.data = attr.data.to(dtype)
    model.apply(_convert_weights)
# 保证backwards compact的兼容性
convert_weights_to_fp16 = convert_weights_to_lp


def convert_to_custom_text_state_dict(state_dict:dict):
    """维护模型检查点checkpoint的兼容性,解决新旧
    版本模型权重字典state_dict的格式差异"""
    if 'text_projection' in state_dict:
        # 旧版的format输出到text_tower的.text
        # 主要就是将state_dict中的键更换一下名称
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final')):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu:bool= True,
        cast_dtype=torch.float16):
    """从预训练的CLIP模型权重字典state_dict中自动提取
    视觉分支和文本分支的配置参数,构建CLIP模型实例,并加载
    预训练权重返回一个评估模式的CLIP模型"""
    # 视觉分支的参数的参数配置
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith('visual.') 
                             and k.endswith('attn.in_proj_weight')])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grad_size = round((state_dict['visual.positional_embedding'].shape[0]-1) ** 0.5)
        image_size = vision_patch_size * grad_size
    else:
        counts:list=[len(set(k.split('.')[2] for k in state_dict if k.startswith('visual.layer{b}'))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict['visual.attnpool.positional_embedding'].shape[0], \
                    f'[WARNING]  output_width ** 2 + 1 != visual.attnpool.positional_embedding.shape[0]'
        image_size = output_width * 32
    #  文本分支的配置参数
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embeddidng.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.startswith('transformer.resblocks')))
    # 修改CONFIG中的相关配置
    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size)
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers)
    model = CLIP(embed_dim,
                 vision_cfg=vision_cfg,
                 text_cfg=text_cfg,
                 quick_gelu=quick_gelu,
                 cast_dtype=cast_dtype)
    # 模型的参数的配置
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        state_dict.pop(key, None)
    # OpenAI model的参数模型部分被注册为fp16的精度
    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size:int=256, 
                device=torch.device('cpu')):
    """该函数的核心作用是使用 PyTorch 的JIT追踪机制,将输入的
    视觉-文本模型(推测为 CLIP 类模型)转换为静态图形式的TorchScript模块"""
    model.eval()
    image_size = model.image_size
    example_image = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
                    model, 
                    inputs=dict(
                        forward=(example_image, example_text),
                        encode_text = (example_text),
                        encode_image=(example_image),),)
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, 
                     interpolation:str='bicubic', 
                     antialias:bool=False):
    """当加载预训练模型的状态字典(state_dict)时,自动调整视觉模块visual中位置嵌入
    positional embedding的网格尺寸,使其匹配当前模型的输入网格大小"""
    # 获取model的visual部分的grad_size参数
    old_pos_embed = state_dict.get('visual.postional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'): return
    # 拆分原来的positional_embedding为类别和图像token
    grad_size = to_2tuple(grad_size)
    extra_token = 1 # FIXME: 检测不同类型的token标注
    new_seq_len = grad_size[0] * grad_size[1] + extra_token
    if new_seq_len == old_pos_embed.shape[0]: return # 两者相等,不必转换
    if extra_token:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_token], old_pos_embed[extra_token:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed[extra_token:]
    old_grad_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
    logging.info(f'[INFO] 重新调整位置参数的grad_size从{old_grad_size}到{grad_size}')
    # 使用插值进行处理pos_embed_image,注意pos_embed_img的形状的变化
    pos_emb_img = pos_emb_img.reshape(1, old_grad_size[0], old_grad_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
                pos_emb_img,
                size=grad_size,
                mode=interpolation,
                antialias=antialias,
                align_corners=False)
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grad_size[0], grad_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
    

def resize_text_pos_embed(state_dict, model, 
                          interpolation:str='linear',
                          antialias:bool=False):
    """"""
    





