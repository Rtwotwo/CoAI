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


class Tanh(nn.Module):
    """计算公式: tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    use_native(bool):是否使用PyTorch原生实现"""
    def __init__(self, use_native: bool = True):
        super().__init__()
        self.use_native = use_native
    def tanh_(self, x: torch.Tensor) -> torch.Tensor:
        # 数值稳定实现：避免直接 exp(x)上溢
        # tanh(x) = sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))
        x_abs = x.abs()
        z = torch.exp(-2.0 * x_abs)
        y = (1.0 - z) / (1.0 + z)
        # 避免 x.sign() 导致梯度在 0 处错误
        return torch.where(x >= 0, y, -y) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.tanh(x) if self.use_native else self.tanh_(x)


class Sigmoid(nn.Module):
    """计算公式: sigmoid(x) = 1 / (1 + e^{-x})
    use_native(bool): 是否使用 PyTorch 原生实现"""
    def __init__(self, use_native: bool = True):
        super().__init__()
        self.use_native = use_native
    def sigmoid_(self, x: torch.Tensor) -> torch.Tensor:
        # 分段避免 exp 上溢
        return torch.where(x >= 0, 1.0 / (1.0 + torch.exp(-x)),
                        torch.exp(x) / (1.0 + torch.exp(x)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(x) if self.use_native else self.sigmoid_(x)


class ReLU(nn.Module):
    """计算公式: ReLU(x) = max(0, x)
    use_native (bool): 是否使用 PyTorch原生实现"""
    def __init__(self, use_native: bool = True):
        super().__init__()
        self.use_native = use_native
    def relu_(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, torch.zeros_like(x))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) if self.use_native else self.relu_(x)


class LeakyReLU(nn.Module):
    """计算公式: LeakyReLU(x) = max(x, alpha * x)
    negative_slope (float): 负斜率,通常称alpha
    use_native (bool): 是否使用 PyTorch 原生实现"""
    def __init__(self, negative_slope: float=0.01, use_native:bool=True):
        super().__init__()
        self.negative_slope = negative_slope
        self.use_native = use_native
    def leakyrelu_(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.negative_slope * x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_native:
            return F.leaky_relu(x, negative_slope=self.negative_slope)
        else:
            return self.leakyrelu_(x)


class ELU(nn.Module):
    """计算公式: ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    alpha (float): 负区缩放因子,negative_slope,但标准叫alpha
    use_native (bool): 是否使用 PyTorch 原生实现"""
    def __init__(self, alpha:float=1.0, use_native:bool=True)->None:
        super().__init__()
        self.alpha = alpha
        self.use_native = use_native
    def elu_(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.alpha * (torch.exp(x) - 1.0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha) if self.use_native else self.elu_(x)
    

class GELU(nn.Module):
    """计算公式: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    use_native (bool): 是否使用 PyTorch 原生实现"""
    def __init__(self, use_native: bool = True)->None:
        super().__init__()
        self.use_native = use_native
    def gelu_(self, x:torch.Tensor)->torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(2.0)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x) if self.use_native else self.gelu_(x)
    

class QuickGELU(nn.Module):
    """计算公式: QuickGELU(x) = x * sigmoid(1.702 * x)
    use_native (bool): 是否使用PyTorch原生实现"""
    def __init__(self, approximate:bool=True, use_native:bool=True):
        super().__init__()
        self.use_native = use_native
        self.approximate = approximate
        self.gelu = GELU(use_native=False)
    def quickgelu_(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate: return self.quickgelu_(x)
        else: return F.gelu(x) if self.use_native else self.gelu(x)


