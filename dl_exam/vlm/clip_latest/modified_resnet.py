"""
Modified By: Redal
Date: 2025-12-21
Todo: 修改版的ResNet架构ModifiedResNet,在传统ResNet基础上结合了
      抗锯齿下采,注意力池化等优化,兼顾了ResNet的局部特征提取能力
      和自注意力的全局建模能力
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from .utils import freeze_batch_norm_2d, feature_take_indices


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes:int,
                 planes: int,
                 stride: int=1
                 )->None:
        super().__init__()
        
