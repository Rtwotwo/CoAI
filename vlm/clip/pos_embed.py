# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------
"""
Modified By: Redal
Date: 2025-12-07
Todo: 实现和处理2D正弦-余弦位置编码(sine-cosine positional embedding),
      并支持在不同分辨率之间插值位置编码,以适应图像尺寸变化.
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import numpy as np


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """生成二维正弦余弦位置编码Positional Embedding"""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # 这里w先,对应图像列方向变化更快
    grid = np.meshgrid(grid_w, grid_h) 
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # 添加cls_token形状为[1+grid_size**2, embed_dim]
    if cls_token: pos_embed = np.concatenate([np.zeros([1, embed_dim], pos_embed)], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """使用get_1d_sincos_pos_embed_from_grid生成二维正弦余弦位置编码
    通过将二维网格分解为高度和宽度两个维度,分别生成1D正弦余弦编码,
    再拼接得到完整的二维位置编码"""
    assert embed_dim % 2 == 0, f'embed_dim必须是偶数'
    # 使用一半的维度来编码 grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) # (H*W, D/2)
    # 再进行拼接(H*W, D)
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """生成1D正弦余弦位置编码,Transformer常用的位置编码方式
    为每个位置生成唯一的编码，同时能捕捉不同位置间的相对关系
    让模型理解序列的时序特性
    输出(M, D),M是位置数量,D是嵌入维度"""
    assert embed_dim % 2 == 0, f'保证embed_dim为偶数'
    # W_k = 1 / (10000^(2i / embed_dim))
    # PE(pos, 2i) = sin(pos / (10000^(2i/embed_dim)))
    # PE(pos, 2i+1) = cos(pos / (10000^(2i/embed_dim)))
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega # (D/2, )
    # 更加直接的omega的实现方式
    # i = np.arange(embed_dim // 2, dtype=float)
    # omega = 1.0 / (10000.0 ** (2 * i / embed_dim))
    # 计算位置与频率外积(M, D/2)的矩阵
    pos = pos.reshape(-1) # (M, )
    out = np.einsum('m,d->md', pos, omega) # (M, D/2)
    # 生成正弦 / 余弦编码
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1) # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    """处理预训练模型checkpoint和当前模型model之间位置嵌入positional 
    embedding的尺寸不匹配问题,通过插值方法调整位置嵌入的大小,使其适配当前模型"""
    if 'pos_embed' in checkpoint_model:
        # 获取当前模型checkpoint的位置嵌入的尺寸
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # 计算原始与新型图像的边长
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
        new_size = int(num_patches**0.5)
        if orig_size != new_size:
            print(f'位置嵌入：从{orig_size}x{orig_size}到{new_size}x{new_size}!')
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # 位置嵌入插值:双三次插值(bicubic)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
