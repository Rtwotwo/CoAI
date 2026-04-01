"""
Author: Redal
Date: 2026-01-28
Todo: Implements the TransFuser Backbone module 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import copy 
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.agents.transfuser.transfuser_config import TransfuserConfig


class TransfuserBackbone(nn.Module):
    """Multi-Scale Fusion Transformer for Image-Lidar Feature Fusion"""
    def __init__(self, config: TransfuserConfig):
        super().__init__()
        self.config = config
        # self.image_encoder = timm.create_model(config.image_architecture, 
        #                                        pretrained=True,
        #                                        features_only=True)
        # image encoder and layers for image features
        self.image_encoder = timm.create_model(config.image_architecture, 
                                               pretrained=True, 
                                               features_only=True,
                                               pretrained_cfg_overlay=dict(file=config.bkb_path))
        # Determine the number of LiDAR input channels based on whether the ground plane is used
        if config.use_ground_plane: in_channels = 2 * config.lidar_seq_len
        else: in_channels = config.lidar_seq_len
        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn((1, in_channels, config.lidar_resolution_width, 
                            config.lidar_resolution_height), requires_grad=True))
        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, 
                                                 self.config.img_horz_anchors))
        # Lidar encoder to extract lidar bev features
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans = in_channels,
            features_only=True,)
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, 
                                                  self.config.lidar_horz_anchors))
        lidar_time_frames = [1, 1, 1, 1]
        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        # create stem layer for some networks
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.transformers = nn.ModuleList(
            [
                GPT( 
                    n_embd = self.image_encoder.feature_info.info[start_index + i]['num_chs'],
                    config = config,
                    # lidar_video = self.lidar_video,
                    lidar_time_frames = lidar_time_frames[i]
                )
                for i in range(4)
            ]
        )


class GPT(nn.Module):
    """The full GPT language backbone, with a context size of block_size"""
    


class SelfAttention(nn.Module):
    """Self-Attention module"""
    def __init__(self, n_embd, n_head, attn_drop, resi_drop):
        super().__init__()
        # query, key, value projections for all heads
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # attention output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        # Dropout layers for regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.resi_drop = nn.Dropout(resi_drop)
    def forward(self, x):
        b, t, c = x.size()
        # calculate query, key, value for all heads in batch 
        # and move head forward to be the batch dim
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        # calculate attention (b, n_head, t, hs) -> (b, n_head, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (b, n_head, t, t)@(b, n_head, t, hs) -> (b, n_head, t, hs)
        scores = att @ v
        scores = scores.transpose(1, 2).contiguous().view(b, t, c)
        # output projection
        scores = self.proj(scores)
        scores = self.resi_drop(scores)
        return scores


class MultiHeadAttentionWithAttention(nn.Module):
    """Multi-Head Attention module"""
    def __init__(self, n_head, n_embd, pdrop):
        super().__init__()
        # query, key, value projections for all heads
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # attention dropout layer for regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resi_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
    def forward(self, q, k, v):
        b, t, c = q.size()
        _, num_t, _ = k.size()
        # calculate query, key, value for all heads in batch 
        # and move head forward to be the batch dim
        # (b, t, c) -> (b, n_head, t, c // n_head)
        q = self.query(q).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = self.key(k).view(b, num_t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(v).view(b, num_t, self.n_head, c // self.n_head).transpose(1, 2)
        # calculate attention scores (b, n_head, t, num_t) -> (b, n_head, t, num_t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1) 
        att = self.attn_drop(att)
        # (b, n_head, t, num_t)@(b, n_head, num_t, hs) -> (b, n_head, t, hs)
        score = att @ v
        # (b, n_head, t, hs) -> (b, t, c)
        score = score.transpose(1, 2).contiguous().view(b, t, c)

        # multihead output projection
        score = self.proj(score)
        score = self.resi_drop(score)
        # average attention over all heads for visualization
        attention = torch.mean(att, dim=1)
        return score, attention


class Block(nn.Module):
    """Transformer Encoder Block with self-attention module"""
    def __init__(self, n_embd, n_head, block_exp, 
                 attn_drop, resi_drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_drop, resi_drop)
        # mlp for transformer block ffp
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLu()
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resi_drop))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerDecoderLayerWithAttention(nn.Module):
    """Transformer Decoder Block that returns attention weights"""
    def __init__(self, d_model, n_head, 
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5):
        super().__init__()
        self.sekf_attn = MultiHeadAttentionWithAttention(n_head, d_model, dropout)
        self.multihead_attn = MultiHeadAttentionWithAttention(n_head, d_model, dropout)
        # Transformer FFP linear layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # normal layer for pre-attention, post-attention and post-ffn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # activation function for FFP
        self.activation = activation
    def forward(self, tgt, memory):
        x = tgt
        # self-attention with pre-normalization and dropout layer
        tmp, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(tmp))
        # multi-head attention with post-normalization and dropout layer
        tmp, attention = self.multihead_attn(x, memory, memory)
        x = self.norm2(x + self.dropout2(tmp))
        # FFP with post-normalization and dropout layer
        tmp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(tmp))
        return x, attention


class TransformerDecoderWithAttention(nn.Module):
    """Transformer Decoder with attention weights returned"""
    def __init__(self, layers, num_layers, norm=None,):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = nn.ModuleList(
            copy.deepcopy(layers) for _ in range(num_layers))
    def forward(self, queries, memory):
        output = queries
        attentions = []
        for mod in self.layers:
            output, attention = mod(output, memory)
            attentions.append(attention)
        if self.norm is not None:
            output = self.norm(output)
        # return average attention over all layers
        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        return output, avg_attention
        
