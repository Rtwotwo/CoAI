"""
Modified By: Redal
Date: 2025-12-20
Todo: CoCa(Contrastive Captioner)模型,融合视觉-文本对比学习
      与文本生成的多模态模型,基于CLIP和Transformer架构扩展而来
Original Repo: 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
from .transformer import (LayerNormFp32, 
                          LayerNorm,
                          QuickGELU,
                          MultimodalTransformer)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
try: 
    from transformers import (BeamSearchScorer, # beam search束搜索的评分器
                              LogitsProcessorList, # 日志概率处理器(LogitsProcessorList+各类处理器)
                              TopPLogitsWarper, # 基于Top-P核采样的处理器,只保留累计概率达到P的前N个token
                              TopKLogitsWarper, # 基于Top-K的处理器,只保留概率最高的K个token
                              RepetitionPenaltyLogitsProcessor, # 重复惩罚处理器,对已生成的token降低其logits
                              MinLengthLogitsProcessor, # 最小长度处理器,强制生成文本达到指定最小长度
                              MaxLengthCriteria, # 最大长度条件,强制生成文本达到指定最大长度
                              StopStringCriteria, # 停止字符串条件,当生成文本中出现指定字符串
                              EosTokenCriteria, # EOS(End of Sentence)token条件
                              StoppingCriteriaList) # 停止条件容器,整合多个停止规则,满足任一条件即停止生成
    GENERATION_TYPES = {'top_k': TopKLogitsWarper,
                        'top_p': TopPLogitsWarper,
                        'beam_search': 'beam_search'}
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {'top_k': None,
                        'top_p': None,
                        'beam_search': 'beam_search'}
    _has_transformers = False


@dataclass
class MultimodalCfg(CLIPTextCfg):
    """面向CLIP衍生多模态模型的配置类:继承自文本配置类
    并扩展了多头注意力,特征聚合相关的超参数"""
    mlp_ratio: int=4
    dim_head: int=64
    heads: int=8
    n_queries: int=256
    attn_pooler_heads: int=8


def _build_text_decoder_tower(embed_dim,
                              multimodal_cfg,
                              quick_gelu: bool=False,
                              cast_dtype: Optional[torch.dtype]=None):
    """PyTorch框架下的文本解码器塔构建函数
    根据配置参数实例化并返回一个适配多模态场景的Transformer解码器"""
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU 
    norm_layer = (LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm)
    decoder = MultimodalTransformer(
                  width=multimodal_cfg.width,
                  layers = multimodal_cfg.layers,
                  heads = multimodal_cfg.heads,
                  context_length=multimodal_cfg.context_length,
                  ls_init_value = multimodal_cfg.ls_init_value,
                  output_dim=embed_dim,
                  act_layer=act_layer,
                  norm_layer=norm_layer)
    return decoder


def _token_to_tensor(token_id, device:str='cpu')->torch.Tensor:
      if not isinstance(token_id, torch.Tensor):
            if isinstance(token_id, int):
                  token_id = [token_id]
            token_id = torch.tensor(token_id, device=device)
      return token_id


class CoCa(nn.Module):
     def __init__(self, 
                  embed_dim,
                  multimodal_cfg: MultimodalCfg,
                  text_cfg: CLIPTextCfg,
                  vision_cfg: CLIPVisionCfg,
                  quick_gelu: bool=False,
                  init_logit_scale: float = np.log(1/0.07),
                  
                  )


