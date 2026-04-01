"""
Author: Redal
Date: 2025-10-14
Todo: lstm_pytorch.py for creating lstm and GRU model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 标准LSTM与PyTorch nn.LSTM功能一致,用于教学/自定义扩展
# 包含: 遗忘门, 输入门, 输出门+记忆单元(Cell State); 相关公式包括: 
# f_t = \sigma(W_{xf}·x_t + W_{hf}·h_{t-1} + b_f)
# i_t &= \sigma(W_{xi}·x_t + W_{hi}·h_{t-1} + b_i) 
# \tilde{C}_t &= \tanh(W_{xc}·x_t + W_{hc}·h_{t-1} + b_c) 
# C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t 
# o_t &= \sigma(W_{xo}·x_t + W_{ho}·h_{t-1} + b_o) 
# h_t &= o_t \odot \tanh(C_t)
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_layers=1, bias=True, dropout=0.0, 
                 bidirectional=False):
        super().__init__()
        # 初始化LSTM层所需超参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # 使用PyTorch内置LSTM(高效且支持cuDNN)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(1)
        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, 
                             self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size,
                             self.hidden_size, device=x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return output, (hn, cn)


# 带窥孔连接Peephole Connections的LSTM
# f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + W_{cf} C_{t-1} + b_f)
class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 从输入到各门的权重值
        self.W_xi = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ci = nn.Parameter(torch.randn(hidden_size)) # 窥孔链接cell -> input gate
        
        self.W_xf = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_cf = nn.Parameter(torch.randn(hidden_size)) # 窥孔链接cell -> forget gate

        self.W_xc = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_xo = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_co = nn.Parameter(torch.randn(hidden_size)) # 窥孔链接cell -> output gate
        # 初始参数化输入, 遗忘, 记忆元与输出门的偏置参数bias
        self.bias_i = nn.Parameter(torch.randn(hidden_size))
        self.bias_f = nn.Parameter(torch.randn(hidden_size))
        self.bias_c = nn.Parameter(torch.randn(hidden_size))
        self.bias_o = nn.Parameter(torch.randn(hidden_size))
    def forward(self, x, h_prev=None, c_prev=None):
        # x is expected to be (seq_len, batch, input_size)
        seq_len, batch_size, _ = x.shape
        # Transpose to (batch, seq_len, input_size) for loop processing
        x = x.transpose(0, 1)  # now (batch, seq_len, input_size)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        # PeeholeLSTM网络的核心计算过程
        # 创建outputs列表存储每个时间步的输出, h_t, c_t继承前时间步的状态
        outputs = []
        h_t, c_t = h_prev, c_prev
        for t in range(seq_len):
            # x形状[batch_size, input_size]
            x_t = x[:, t, :]
            # 输入门, 遗忘门, 候选记忆元, 更新cell -> 使用sigmoid函数将值压缩到0-1之间
            # 计算输入门控信号-决定多少新信息进入细胞状态
            # 计算遗忘门控信号-决定要保留多少之前的细胞状态
            # 计算新的候选记忆内容-使用tanh函数将值压缩到-1到1之间
            i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h_t) + self.W_ci * c_t + self.bias_i)
            f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h_t) + self.W_cf * c_t + self.bias_f)
            c_hat = torch.tanh(self.W_xc(x_t) + self.W_hc(h_t) + self.bias_c)
            c_t = f_t * c_t + i_t * c_hat
            # 输出门, 更新隐藏状态
            o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_t) + self.W_co * c_t + self.bias_o)
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t.unsqueeze(1))
        # 输出output形状[Batch_size, seq_len, hidden_size]
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden)
        output = output.transpose(0, 1)     # (seq_len, batch, hidden) to match PyTorch convention
        return output, (h_t, c_t)

        
# 耦合输入-遗忘门, 即遗忘门和输入门耦合LSTM: f_t = 1-i_t
# C_t = (1 - i_t) \odot C_{t-1} + i_t \odot \tilde{C}_t
class CoupledLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 从输入到各门的权重值, 遗忘门强制被耦合进输入门, 因此不单独设置权重和偏置参数
        self.W_xi = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_i = nn.Parameter(torch.randn(hidden_size))

        self.W_xc = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_c = nn.Parameter(torch.randn(hidden_size))

        self.W_xo = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_o = nn.Parameter(torch.randn(hidden_size))
    def forward(self, x, h_prev=None, c_prev=None):
        # x is (seq_len, batch, input_size)
        seq_len, batch_size, _ = x.shape
        x = x.transpose(0, 1)  # (batch, seq_len, input_size)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        h_t, c_t = h_prev, c_prev
        for t in range(seq_len):
            # x: [batch_size, seq_len, input_size]
            # 单独取出当前时间步的状态数据
            x_t = x[:, t, :]
            # 耦合门: i_t 控制写入的f_t = 1-i_t
            i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h_t) + self.bias_i)
            f_t = 1 - i_t
            c_hat = torch.tanh(self.W_xc(x_t) + self.W_hc(h_t) + self.bias_c)
            c_t = f_t * c_t + i_t * c_hat
            # 输出门进行计算, 并且更行隐藏状态
            o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_t) + self.bias_o)
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t.unsqueeze(1))
        # 输出output形状[Batch_size, seq_len, hidden_size]
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden)
        output = output.transpose(0, 1)     # (seq_len, batch, hidden)
        return output, (h_t, c_t)
    

# GRU作为LSTM的轻量替代(非LSTM, 但常被归为同类)
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        # 使用torch.nn提供的GRU实例化
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)
    def forward(self, x, h0=None):
        batch_size = x.size(1)  # ← FIXED: was x.size(0)
        num_directions = 2 if self.bidirectional else 1
        if h0 is None:
            h0 = torch.zeros(self.gru.num_layers * num_directions, batch_size, 
                             self.gru.hidden_size, device=x.device)
        output, hn = self.gru(x, h0)
        return output, hn    
    

# 统一创建LSTM实例化函数, 根据输入参数选择LSTM实例
# 根据variant返回对应LSTM模型,支持:'vanilla','peephole','coupled','gru'
def build_lstm(variant='vanilla', **kwargs):
    input_size = kwargs.get('input_size')
    hidden_size = kwargs.get('hidden_size')
    assert input_size is not None and hidden_size is not None, "must specify input_size and hidden_size"
    if variant == 'vanilla':
        return VanillaLSTM(**kwargs)
    elif variant == 'peephole':
        assert kwargs.get('num_layers', 1) == 1 and not kwargs.get('bidirectional', False), 'PeepholeLSTM only support single directinal layer'
        return PeepholeLSTM(input_size, hidden_size)
    elif variant == 'coupled':
        assert kwargs.get('num_layers', 1)==1 and not kwargs.get('bidirectional', False), 'CoupledLSTM only support single directinal layer'
        return CoupledLSTM(input_size, hidden_size)
    elif variant == 'gru':
        return GRU(**kwargs)
    else: raise ValueError('Invalid variant: {}'.format(variant))


if __name__ == '__main__':
    # 依次B-批次大小, T-序列长度或时间步数, D-输入特征的维度, H-LSTM隐藏层的大小
    # 对应batch_size, seq_len, input_size, hidden_size参数
    B, T, D, H = 2, 5, 10, 20
    x = torch.randn(T, B, D)
    # 实例化LSTM模型并测试代码
    vanilla_lstm = build_lstm('vanilla', input_size=D, hidden_size=H, num_layers=2, bidirectional=True)
    out, _ = vanilla_lstm(x)
    print(vanilla_lstm, f"Vanilla LSTM output shape: {out.shape}")
    peephole_lstm = build_lstm('peephole', input_size=D, hidden_size=H)
    out, _ = peephole_lstm(x)
    print(peephole_lstm, f"Peephole LSTM output shape: {out.shape}")
    coupled_lstm = build_lstm('coupled', input_size=D, hidden_size=H)
    out, _ = coupled_lstm(x)
    print(coupled_lstm, f"Coupled LSTM output shape: {out.shape}")
    gru_lstm = build_lstm('gru', input_size=D, hidden_size=H, num_layers=2)
    out, _ = gru_lstm(x)
    print(gru_lstm, f"GRU output shape: {out.shape}")