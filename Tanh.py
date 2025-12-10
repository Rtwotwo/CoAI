import torch
import torch.nn as nn

class Tanh(nn.Module):
    """计算公式：
    $$
    \tanh(x)=\frac{e^x - e^{-x}}{e^x + e^{-x}}
    $$
    数值稳定实现，不使用 torch.tanh 或 nn.functional。
    """
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def tanh_(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 z = exp(-2*|x|) 的稳定形式：tanh(x)=sign(x)*(1 - z)/(1 + z)，其中 z = exp(-2*|x|)
        z = torch.exp(-2.0 * x.abs())
        y = (1.0 - z) / (1.0 + z)
        y = y * x.sign()
        if self.inplace:
            x.copy_(y)
            return x
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_(x)
