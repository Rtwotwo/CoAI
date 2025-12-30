"""
Modified By: Redal
Date: 2025-12-20
Todo: CLIP 模型OpenAI版本的加载与模型列表查询工具,
      核心用于便捷获取OpenAI预训练CLIP模型并完成加载配置
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import os
import warnings
from typing import Dict, List, Optional, Union
import torch
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url
__all__ = ['list_openai_models', 'load_openai_model']



