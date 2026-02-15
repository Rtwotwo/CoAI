"""
Author: Redal
Date: 2025-10-14
Todo: transformer.py created for transformer model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import math
import torch
import torch.nn as nn


def dropout(x, p=0.1, training=True):
    if not training or p==0.0:
        return x
    keep_prob = 1-p
    mask = (torch.rand_like(x) < keep_prob).float() / keep_prob
    return x * mask


def gelu(x):
    # GELU(x) = x·Φ(x), 其中Φ(x)是标准正态分布的累积分布函数(CDF)
    # GELU(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
    return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = normalized_shape
        self.eps = eps
        # 必须初始化参数为Parameter
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
    def __call__(self, x):
        # x: (..., *normalized_shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
    

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        # 使用Xavier近似初始化
        bound = 1 / math.sqrt(in_features)
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))
        else:
            self.bias = None
    def __call__(self, x):
        # x: (*, in_features) -> (*, out_features)
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output
        

# 创建transformer架构组件
class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout=0.1):
        assert d_model % num_heads == 0, "d_model nust be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        # 创建q, k, v, out权重矩阵
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        # 创建缩放因子, 使用head_dim控制缩放
        self.scale = math.sqrt(self.head_dim)
    def __call__(self, query, key, value, mask=None, training=True):
        # query, key, value形状均为[batch_size, seq_len, d_model]
        # batch_size批次大小, seq_len序列长度, d_model模型特征维度
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        # 计算Q, K, V线性投影, 使用Linear层处理
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        # 参数空间映射到更适合注意力计算的空间
        # 让每个注意力头关注不同的表示子空间, 注意V同样使用seq_len_k
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力得分，使用Q, K进行矩阵乘法 atten_scores = (Q @ K.t) / sqrt(d)
        # 除以scale（√head_dim）进行缩放，防止点积过大导致梯度问题
        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # 支持 bool mask 或 0/1 mask
            if mask.dtype == torch.bool:
                atten_scores = atten_scores.masked_fill(~mask, float('-inf'))
            else:
                atten_scores = atten_scores.masked_fill(mask==0, float('-inf'))
        
        # 计算注意力权重 atten_weights = softmax(atten_scores)
        # 计算注意力输出 atten_output = atten_weights @ V
        atten_weights = torch.softmax(atten_scores, dim=-1)
        atten_weights = dropout(atten_weights, p=self.dropout_p, training=training)
        atten_output = torch.matmul(atten_weights, V)
        # 将多头注意力输出拼接在一起, 并进行线性变换
        atten_output = atten_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(atten_output)
        return output


class PositionalEncoding:
    def __init__(self, d_model, max_len=512):
        # 固定正弦位置编码是预先计算好、不可训练的, 效果有限
        # 使用可训练的pos_embedding, 可以在训练过程中更新位置编码
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
    def __call__(self, x):
        seq_len = x.shape[1]
        return x + self.pos_embedding[:, :seq_len, :]


class FeedForward:
    def __init__(self, d_model, dff, dropout=0.1):
        self.w1 = Linear(d_model, dff)
        self.w2 = Linear(dff, d_model)
        self.dropout_p = dropout
    def __call__(self, x, training=True):
        x = self.w1(x)
        x = gelu(x)
        x = dropout(x, p=self.dropout_p, training=training)
        x = self.w2(x)
        return x


class Embedding:
    def __init__(self, vocab_size, d_model):
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, d_model))
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
    def __call__(self, input_ids):
        # 查表操作: 直接使用索引查找对应的嵌入向量
        # 当输入一个词的索引时，它会返回对应的向量
        return self.weight[input_ids]
    

# 构造Transformer架构Encoder和Decoder两块组件
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        # 编码器Encoder部分的单个层实现
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dff, dropout=dropout)
        self.dropout_p = dropout
    def __call__(self, x, mask=None, training=True):
        # 层归一化 -> 多头注意力 -> 残差连接
        attn_input = self.norm1(x)
        attn_out = self.attn(attn_input, attn_input, attn_input, mask=mask, training=training)
        x = x + dropout(attn_out, p=self.dropout_p, training=training)
        # 层归一化 -> 前向传播 -> 残差连接
        ffn_input = self.norm2(x)
        ffn_out = self.ffn(ffn_input,training=training)
        x = x + dropout(ffn_out, p=self.dropout_p, training=training)
        return x
    

class TransformerDecoderLayer:
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        # 解码器Decoder部分的单个层实现
        self.norm1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm3 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dff, dropout=dropout)
        self.dropout_p = dropout
    def __call__(self, x, enc_output, tgt_mask=None, src_mask=None, training=True):
        # enc_output: [batch_size, src_seq_len, d_model], 编码器Encoder处理输入序列后产生的输出
        # tgt_mask: [batch_size, 1, 1, src_seq_len]或[batch_size, src_seq_len], 处理源序列的掩码
        # src_mask: [[batch_size, tgt_seq_len, tgt_seq_len], 用于处理目标序列的掩码
        # 层归一化 -> 自注意力 -> 残差连接 Self-attention(causal)
        self_attn_input = self.norm1(x)
        self_attn_out = self.self_attn(self_attn_input, self_attn_input, self_attn_input, mask=tgt_mask, training=training)
        x = x + dropout(self_attn_out, p=self.dropout_p, training=training)
        # 层归一化 -> 多头注意力 -> 残差连接 Cross-attention
        cross_attn_input = self.norm2(x)
        cross_attn_out = self.cross_attn(cross_attn_input, enc_output, enc_output, mask=src_mask, training=training)
        x = x + dropout(cross_attn_out, p=self.dropout_p, training=training)
        # 层归一化 -> 前向传播 -> 残差连接 FFN
        ffn_input = self.norm3(x)
        ffn_out = self.ffn(ffn_input, training=training)
        x = x + dropout(ffn_out, p=self.dropout_p, training=training)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 注册嵌入层Embedding, 位置编码层PositionalEncoding
        self.embedding = Embedding(config['vocab_size'], config['d_model'])
        self.pos_encoding = PositionalEncoding(config['d_model'], config['max_seq_len'])
        # 注册多个编码器层TransformerEncoderLayer
        self.layers = []
        for _ in range(config['num_layers']):
            layer = TransformerEncoderLayer(
                d_model = config['d_model'],
                num_heads=config['num_heads'],
                dff=config['dff'],
                dropout=config['dropout'])
            self.layers.append(layer)
        self._register_parameters()
    def _register_parameters(self):
        # 递归收集对象中的所有Parameter参数
        # 并将它们注册到当前模块params字典中
        params = {}
        def collect(obj, prefix=""):
            if hasattr(obj, '__dict__'):
                for k, v in obj.__dict__.items():
                    # 三种情况遇到的参数类型分别是Parameter, list, dict
                    name = f'{prefix}.{k}' if prefix else k
                    if isinstance(v, torch.nn.Parameter):
                        params[name] = v
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            collect(item, f"{name}.{i}")
                    else:
                        collect(v, name)
        collect(self)
        for name, param in params.items():
            # register_parameter属于selfs属性, 区分_register_parameters()
            self.register_parameter(name.replace('.', '_'), param)
    def forward(self, input_ids, attention_mask=None, training=True):
        # input_ids输入的词索引序列，形状为[batch_size, seq_len]
        # attention_mask可选的注意力掩码，用于屏蔽某些位置
        x = self.embedding(input_ids) * math.sqrt(self.config['d_model'])
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=attention_mask, training=training)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 注册嵌入层Embedding, 位置编码层PositionalEncoding
        self.embedding = Embedding(config['vocab_size'], config['d_model'])
        self.pos_encoding = PositionalEncoding(config['d_model'], config['max_seq_len'])
        # 注册多个解码器层TransformerDecoderLayer
        self.layers = []
        for _ in range(config['num_layers']):
            layer = TransformerDecoderLayer(
                d_model = config['d_model'],
                num_heads = config['num_heads'],
                dff = config['dff'],
                dropout = config['dropout'])
            self.layers.append(layer)
        self.final_norm = LayerNorm(config['d_model'])
        self._register_parameters()
    def _register_parameters(self):
        # 递归收集对象中的所有Parameter参数
        # 并将它们注册到当前模块params字典中
        params = {}
        def collect(obj, prefix=""):
            if hasattr(obj, '__dict__'):
                for k, v in obj.__dict__.items():
                    # 三种情况遇到的参数类型分别是Parameter, list, dict
                    name = f'{prefix}.{k}' if prefix else k
                    if isinstance(v, torch.nn.Parameter):
                        params[name] = v
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            collect(item, f'{name}.{i}')
                    else:
                        collect(v, name)
        collect(self)
        for name, param in params.items():
            # register_parameter属于selfs属性, 区分_register_parameters()
            self.register_parameter(name.replace('.', '_'), param)
    def forward(self, tgt_ids, enc_output, tgt_mask=None, src_mask=None, training=True):
        # tgt_ids目标的词索引序列，形状为[batch_size, seq_len]
        # enc_output编码器的输出，形状为[batch_size, seq_len, d_model]
        # tgt_mask可选的解码器注意力掩码，用于屏蔽某些位置
        # src_mask可选的编码器注意力掩码，用于屏蔽某些位置
        x = self.embedding(tgt_ids) * math.sqrt(self.config['d_model'])
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask, training=training)
        x = self.final_norm(x)
        return x


# 完整Transformer架构
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        # 语言模型的输出层, 主要有两个作用: 将隐藏状态转换为下一个词的预测概率
        # 是生成任务（如文本生成、机器翻译）的最后一环
        self.lm_head = Linear(config['d_model'], config['vocab_size'], bias=False)
        # weight tying权重共享
        self.lm_head.weight = self.decoder.embedding.weight
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, training=True):
        # src_ids源语言的词索引序列，形状为[batch_size, src_seq_len]
        # tgt_ids目标语言的词索引序列，形状为[batch_size, tgt_seq_len]
        # src_mask可选的源语言注意力掩码，用于屏蔽某些位置
        # tgt_mask可选的目标语言注意力掩码，用于屏蔽某些位置
        enc_output = self.encoder(src_ids, attention_mask=src_mask, training=training)
        dec_output = self.decoder(tgt_ids, enc_output, tgt_mask=tgt_mask, src_mask=src_mask, training=training)
        # 将解码器的输出通过线性层转换为下一个词的预测概率
        logits = self.lm_head(dec_output)
        return logits
    

# 工具函数: 用于构造相关掩码
def generate_causal_mask(seq_len, device=None):
    # 生成因果掩码(Causal Mask)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    # 保证mask形状为[1, 1, seq_len, seq_len]
    return mask.unsqueeze(0).unsqueeze(0)


# 设置transformer不同尺寸的经验性设定模型参数
MODEL_CONFIGS = {
    'tiny': {'num_layers': 2, 'd_model': 128, 'num_heads': 2, 'dff': 512, 
             'vocab_size': 30522, 'max_seq_len': 512, 'dropout': 0.1},
    'small': {'num_layers': 4, 'd_model': 256, 'num_heads': 4, 'dff': 1024,
        'vocab_size': 30522, 'max_seq_len': 512, 'dropout': 0.1},
    'medium': {'num_layers': 6, 'd_model': 512, 'num_heads': 8, 'dff': 2048,
        "vocab_size": 30522, 'max_seq_len': 512, 'dropout': 0.1},
    'large': {'num_layers': 12, 'd_model': 768, 'num_heads': 12, 'dff': 3072,
        'vocab_size': 30522, 'max_seq_len': 512, 'dropout': 0.1},
    'huge': {'num_layers': 24, 'd_model': 1024, 'num_heads': 16, 'dff': 4096,
        'vocab_size': 30522, 'max_seq_len': 512,'dropout': 0.1},}


# 创建构建函数, 辅助实例化Transformer类, 注意: 代码中使用了字典来存储不同尺寸的模型配置
# build_transformer仅包含编码器Encoder-only, 适用于如 BERT,分类,序列标注等任务
# build_transformer_full包含完整Transformer的Encoder+Decoder, 适用于如机器翻译, 文本生成等seq2seq任务
def build_transformer(size: str)-> torch.nn.Module:
    assert size in MODEL_CONFIGS, f'Size must be one of {list(MODEL_CONFIGS.keys())}'
    config = MODEL_CONFIGS[size]
    return TransformerEncoder(config)
def build_transformer_full(size: str)-> torch.nn.Module:
    assert size in MODEL_CONFIGS, f'Size must be one of {list(MODEL_CONFIGS.keys())}'
    config = MODEL_CONFIGS[size]
    return Transformer(config)


if __name__ == '__main__':
    print("=== 测试 Encoder-only ===")
    model_enc = build_transformer('huge')
    print(model_enc)
    input_ids = torch.randint(0, 30522, (2, 64))
    output = model_enc(input_ids, training=True)
    print(f"Encoder 输出: {output.shape}")
    print(f"Encoder 参数量: {sum(p.numel() for p in model_enc.parameters()) / 1e3:.1f}K")

    print("\n=== 测试完整 Transformer (Encoder + Decoder) ===")
    model_full = build_transformer_full('huge')
    print(model_full)
    src_ids = torch.randint(0, 30522, (2, 32))
    tgt_ids = torch.randint(0, 30522, (2, 20))
    causal_mask = generate_causal_mask(tgt_ids.shape[1], device=tgt_ids.device)
    logits = model_full(src_ids, tgt_ids, tgt_mask=causal_mask, training=True)
    print(f"Src: {src_ids.shape}, Tgt: {tgt_ids.shape}, Logits: {logits.shape}")
    print(f"完整模型参数量: {sum(p.numel() for p in model_full.parameters()) / 1e3:.1f}K")