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
    """将所提供模块的所有BatchNorm2d和SyncBatchNorm层转换为FrozenBatchNorm2d
    如果module本身是BatchNorm2d或SyncBatchNorm的实例,它将被转换为FrozenBatchNorm2d并返回
    否则,将递归遍历该模块,并就地转换子模块.
    module (torch.nn.Module): Any PyTorch module.
    module_match (dict): Dictionary of full module names to freeze (all if empty)
    name (str): Full module name (prefix)
    https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762"""
    res = module
    is_match = True
    if module_match: 
        is_match = name in module_match
    # 如果匹配并且module属于batchnorm或者syncbatchnorm则冻结并将参数传递给res
    if is_match and isinstance(module, (nn.module.batchnorm.BatchNorm2d, nn.module.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data.clone().detach()
        res.running_var.data = module.running_var.data.clone().detach()
        res.eps = module.eps
    else:
        # 使用递归方法遍历子模块child_module
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name] if name else child_name)
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# 嵌套函数-生成一个工具函数
# def _ntuple(n:int):
#     """将输入x转换为长度为n的元组,如果x是可迭代对象则直接返回,否则重复xn次"""
#     def parse(x):
#         if isinstance(x, collections.abc.Iterable):
#             return x
#         return tuple(repeat(x, n))
#     return parse
def _ntuple(n:int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            res = tuple(x)[:n]  # 截断为n长度
            return res + (res[-1],) * (n - len(res)) if len(res) < n else res
        return tuple(repeat(x, n))
    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n,x: _ntuple(n)(x)


def replace_linear(model, linear_replacement, 
                   include_modules=['c_fc', 'c_proj'], 
                   copy_weights=True):
    """用linear_replacement替换所有线性层
    为包括注意力层和卷积网络在内的其他线性层添加int8支持
    model: 待替换线性层的目标模型
    linear_replacement: 用于替换原线性层的自定义层类/实例
    copy_weights: 是否将原线性层的权重/偏置拷贝到新层"""
    for name, module in model.named_children():
        # 递归遍历所有的子模块
        if len(list[Any](module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)
        # 判断子模块的类型并进行替换
        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,)
            # 复制原线性层的权重weights/bias偏置到新层
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias.data)
        return model
 

def convert_int8_model_to_inference_mode(model):
    """将INT8模型转换为推理模式,冻结所有BatchNorm层并将线性层转换为INT8"""
    for m in model.modules():
        if hasattr(m, "prepare_for_eval"):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype


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