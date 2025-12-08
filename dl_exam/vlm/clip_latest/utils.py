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
Todo: 提BN冻结,线性层替换,INT8推理准备,中间特征索引解析等实用工具
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
from torch import nn
from torch import _assert
from torchvision.ops.misc import FrozenBatchNorm2d
import collections.abc
from itertools import repeat
from typing import List, Optional, Tuple, Union


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """"""
    


def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    """用linear_replacement替换所有线性层
    为包括注意力层和卷积网络在内的其他线性层添加int8支持"""
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            replace_linear(module, )
