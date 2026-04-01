"""
Author: Redal
Date: 2025-10-16
Description: Pure Python implementation of multi-layer bidirectional LSTM with dropout,
             fully compatible with torch.nn.LSTM (batch_first=True).
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """Single LSTM cell implementing standard equations (no peephole connections)."""
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
    def reset_parameters(self):
        k = 1.0 / self.hidden_size
        for weight in self.parameters():
            nn.init.uniform_(weight, -k, k)
    def forward(self, x, h, c):
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h, self.weight_hh, self.bias_hh)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ManualLSTM(nn.Module):
    """Pure Python implementation of multi-layer, bidirectional LSTM with dropout.
    Fully compatible with torch.nn.LSTM when batch_first=True."""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # Create LSTMCell for each layer and direction
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            for _ in range(self.num_directions):
                self.layers.append(LSTMCell(layer_input_size, hidden_size, bias))
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 and num_layers > 1 else None
    def forward(self, x, hx=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        num_states = self.num_layers * self.num_directions
        if hx is None:
            h0 = torch.zeros(num_states, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(num_states, batch_size, self.hidden_size, device=device)
        else:
            h0, c0 = hx
        # Reorganize hx into list: h_list[layer][direction] -> (batch, hidden)
        h_list = []
        c_list = []
        idx = 0
        for layer in range(self.num_layers):
            h_layer = []
            c_layer = []
            for direction in range(self.num_directions):
                h_layer.append(h0[idx])
                c_layer.append(c0[idx])
                idx += 1
            h_list.append(h_layer)
            c_list.append(c_layer)
        final_h = []
        final_c = []
        input_seq = x  # (B, T, D)
        for layer in range(self.num_layers):
            # Forward direction
            h_f, c_f = h_list[layer][0], c_list[layer][0]
            output_f = []
            for t in range(seq_len):
                h_f, c_f = self.layers[layer * self.num_directions + 0](input_seq[:, t, :], h_f, c_f)
                output_f.append(h_f)
            output_f = torch.stack(output_f, dim=1)  # (B, T, H)
            if self.bidirectional:
                # Backward direction
                h_b, c_b = h_list[layer][1], c_list[layer][1]
                output_b = []
                for t in range(seq_len - 1, -1, -1):
                    h_b, c_b = self.layers[layer * self.num_directions + 1](input_seq[:, t, :], h_b, c_b)
                    output_b.insert(0, h_b)
                output_b = torch.stack(output_b, dim=1)  # (B, T, H)
                output = torch.cat([output_f, output_b], dim=-1)  # (B, T, 2H)
                final_h.extend([h_f, h_b])
                final_c.extend([c_f, c_b])
            else:
                output = output_f  # (B, T, H)
                final_h.append(h_f)
                final_c.append(c_f)
            # Apply dropout except on last layer
            if self.dropout_layer is not None and layer != self.num_layers - 1:
                output = self.dropout_layer(output)
            input_seq = output
        output = input_seq  # (B, T, H * num_directions)
        hn = torch.stack(final_h, dim=0)  # (num_layers * num_directions, B, H)
        cn = torch.stack(final_c, dim=0)  # (num_layers * num_directions, B, H)
        return output, (hn, cn)


def copy_weights_from_torch_lstm(torch_lstm, manual_lstm):
    """Copy weights from torch.nn.LSTM to ManualLSTM."""
    with torch.no_grad():
        for i in range(manual_lstm.num_layers):
            for d in range(manual_lstm.num_directions):
                cell = manual_lstm.layers[i * manual_lstm.num_directions + d]
                # Determine parameter names
                if d == 0:  # forward
                    w_ih_name = f'weight_ih_l{i}'
                    w_hh_name = f'weight_hh_l{i}'
                    b_ih_name = f'bias_ih_l{i}'
                    b_hh_name = f'bias_hh_l{i}'
                else:  # backward
                    w_ih_name = f'weight_ih_l{i}_reverse'
                    w_hh_name = f'weight_hh_l{i}_reverse'
                    b_ih_name = f'bias_ih_l{i}_reverse'
                    b_hh_name = f'bias_hh_l{i}_reverse'

                # Copy weights
                cell.weight_ih.copy_(getattr(torch_lstm, w_ih_name))
                cell.weight_hh.copy_(getattr(torch_lstm, w_hh_name))
                if torch_lstm.bias:
                    cell.bias_ih.copy_(getattr(torch_lstm, b_ih_name))
                    cell.bias_hh.copy_(getattr(torch_lstm, b_hh_name))


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    B, T, D, H = 2, 5, 10, 20
    x = torch.randn(B, T, D)
    # PyTorch official LSTM
    lstm_torch = nn.LSTM(D, H, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2, bias=True)
    out_torch, (hn_torch, cn_torch) = lstm_torch(x)
    # Manual implementation
    lstm_manual = ManualLSTM(D, H, num_layers=2, bidirectional=True, dropout=0.2, bias=True)
    # Copy weights to ensure identical behavior
    copy_weights_from_torch_lstm(lstm_torch, lstm_manual)
    out_manual, (hn_manual, cn_manual) = lstm_manual(x)
    print(lstm_manual)