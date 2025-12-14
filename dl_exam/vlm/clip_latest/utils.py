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


def feature_take_indices(
        num_features: int, 
        indices: Optional[Union[int, List[int]]]=None,
        as_set: bool = False,
        )->Tuple[List[int], int]:
    """根据输入的索引参数,结合特征总数num_features,筛选出有效的特征索引列表,
    并返回该索引集合或列表以及其中的最大索引值"""
    # 当所有的特征均为None
    if indices is None: indices = num_features
    # 选取特征的最后indices个索引
    if isinstance(indices, int):
        _assert(0<= indices < num_features, f'feature index {indices} is out of range (0 to {num_features - 1})')
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i<0 else i
            _assert(0<= idx < num_features, f'feature index {idx} is out of range (0 to {num_features - 1})')
            take_indices.append(idx)
    # 判断是否在PyTorch脚本编译模式下,该模式不支持集合set
    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)
    return take_indices, max(take_indices)


def _out_indices_as_tuple(x: Union[int, Tuple[int, ...]]
                          )->Tuple[int, ...]:
    if isinstance(x, int):
        # 如果x是int类型,返回最后x个索引
        return tuple(range(-x, 0))
    return tuple(x)