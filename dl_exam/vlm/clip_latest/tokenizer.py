"""
Modified By: Redal
Date: 2025-12-20
Todo: 将自然语言文本转换为模型可识别的数值token序列,
      同时包含反向解码token 转文本和多种适配CLIP输入要求的扩展功能
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import gzip 
import html
import os
import random
import string
import warnings
import ftfy
import numpy as np
import regex as re
import torch
from functools import lru_cache, partial
from typing import Callable, List, Optional, Union, Dict
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
_nltk_init = False
DEFAULT_CONTEXT_LENGTH=77


@lru_cache()
def default_bpe():
    """核心作用是实现函数结果缓存,避免重复计算带来的性能损耗"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "bpe_simple_vocab_16e6.text.gz")


@lru_cache()
def bytes_to_unicode():
    """返回utf-8字节列表和相应的unicode字符串列表。
    可逆的bpe代码适用于unicode字符串
    这意味着如果您想避免UNK未登录词,您的词汇表中需要大量的unicode字符
    当您处理大约10B token的数据集时,为了获得不错的覆盖率,最终大约需要5K个unicode字符
    这在您通常的(比如32K的bpe词汇表)中占相当大的比例
    为了避免这种情况,我们需要utf-8字节和unicode字符串之间的查找表
    并且避免映射到bpe代码无法处理的空白/控制字符"""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            

