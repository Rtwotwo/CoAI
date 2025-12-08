"""
Author: Redal
Date: 2025-12-08
Todo: 完成激活函数的底层实现,并添加到activation_layers中,
      以供后续模型架构的model调用.
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch 
from torch import nn
import torch.nn.functional as F


class tanh(nn.Module):
    """计算公式: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    inplace: 是否使用inplace原地张量操作"""
    def __init__(self, inplace:bool=False)->None:
        super().__init__()
    def tanh_(self, x:torch.Tensor)->torch.Tensor:
        x = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        return x if self.inplace else x.clone()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.tanh_(x)
    

class sigmoid(nn.Module):
    """计算公式: sigmoid(x) = 1 / (1 + e^-x)
    inplace: 是否使用inplace原地张量操作"""
    def __init__(self, inplace:bool=False)->None:
        super().__init__()
        self.inplace = inplace
    def sigmoid_(self, x:torch.Tensor)->torch.Tensor:
        one_ = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        x = one_ / (one_ + torch.exp(-x))
        return x if self.inplace else x.clone()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.sigmoid_(x)


class ReLU(nn.Module):
    """计算公式: ReLU(x) = max(x, 0)
    inplace: 是否使用inplace原地张量操作"""
    def __init__(self, inplace:bool=False)->None:
        super().__init__()
        self.inplace = inplace
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.inplace:
            return x.clamp_min_(0)
        else:
            return x.clamp_min(0)


class LeakyReLU(nn.Module):
    """计算公式: LeakyReLU(x) = max(x, alpha*x)
    inplace: 是否使用inplace原地张量操作
    negative_slope: 负斜率, 默认为0.01"""
    def __init__(self, inplace:bool=False,
                 negative_slope:float=0.01)->None:
        super().__init__()
        self.inplace = inplace
        self.negative_slope = negative_slope
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.inplace:
            return x.copy_(torch.where(x >= 0, x, self.negative_slope *x))
        else:
            return torch.where(x >= 0, x, self.negative_slope *x)


class ELU(nn.Module):
    """计算方式: ELU(x) = max(x, alpha*(exp(x)-1))
    inplace: 是否使用inplace原地张量操作
    negative_slope: 超参敏感数斜率, 默认为1.0"""
    def __init__(self, inplace:bool=False,
                 negative_slope:float=1.0)->None:
        super().__init__()
        self.inplace = inplace
        self.negative_slope = negative_slope
    def elu_(self, x:torch.Tensor)->torch.Tensor:
        x = torch.where(x >= 0, x, self.negative_slope * (torch.exp(x) - 1.0))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.inplace:
            return self.elu_(x)
        else:
            return x.copy_(self.elu_(x))
        