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
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), 
            ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """该函数提取单词中相邻字符的所有配对"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return  pairs


def basic_clean(text):
    """文本基础清洗工具核心作用是修复文本中的常见格式异常"""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """去除文本中的多余空白字符"""
    text = ' '.join(text.split())
    return text.strip()


def _clean_canonicalize(x):
    """基础(处理),去除空格,去除标点,小写(转换)"""
    return  canonicalize_text(basic_clean(x))


def _clean_lower(x):
    """将文本转换为小写"""
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
     """去除文本中的多余空白字符"""
     return whitespace_clean(basic_clean(x))


def canonicalize_text(text, *, keep_punctuation_exact_string=None,
                      trans_punctuation:dict = str.maketrans("", "", string.punctuation)):
    """keep_punctuation_exact_string如果提供了该参数,
    那么这个精确的字符串会被保留,提供'{}'将保留所有出现的
    '{}'但仍会移除单独出现的'{'和'}'"""
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        # 保留'{}'符号进行符号处理
        text = keep_punctuation_exact_string.join(
                    part.translate(trans_punctuation) 
                    for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate( trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def get_clean_fn(type: str):
    """获取文本清理函数"""
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return  _clean_whitespace
    else:
        assert False, f'[WARNING] Invalid clean function type: {type}!'


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str=default_bpe(),
                 additional_special_tokens: Optional[List[str]]=None,
                 context_length: Optional[int]=DEFAULT_CONTEXT_LENGTH,
                 clean: str = 'lower',
                 reduction_mask: str=''):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
        # 处理bpe_simple_vocab_16e6.txt.gz文件数据
        merges = gzip.open(bpe_path).read().decode('utf-8').split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        # 添加额外的token用于标注或后续其他任务
        special_tokens = ['<start_of_text>', '<end_of_text>']
        if additional_special_tokens: 
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.bpe_tokens = dict(zip(merges,  range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        # 缓存特殊token(避免重复处理，提升效率)
        special = "|".join(special_tokens)
        # 正则表达式模式,用于文本分词预处理
        self.pat = re.compile(special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                              re.IGNORECASE)
        self.vocab_szie = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.reduction_fn = get_reduction_mask_fn(reduction_mask) if reduction_mask else None
_tokenizer =  SimpleTokenizer()
        

def decode(output_ids: torch.Tensor):
    """将token序列解码为原始文本
    output_ids模型输出的token序列,需要由索引转换成文本"""
    output_ids = output_ids.cpu().numpy()
    return _tokenizer.decode(output_ids)


def tokenize(texts: Union[str, List[str]],
             context_length: int=DEFAULT_CONTEXT_LENGTH,
             )->torch.Tensor:
    """将文本转换为token序列
    texts: 输入的文本,可以是单个文本字符串或文本列表
    context_length: 模型输入的序列长度"""
    return _tokenizer(texts,  context_length=context_length)


def random_mask_tokenize(texts: Union[str, List[str]],
                         context_length: int,
                         sot_token_id: int,
                         eot_token_id: int,
                         encode_fn: Callable,
                         shuffle: bool=False):
    """随机掩码填充文本"""
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        # context_length-2留给sot和eot token
        if num_tokens > context_length-2: 
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                 indices = indices.msort()
            tokens =  tokens[indices]
            num_tokens = num_keep
        result[i, 0] = sot_token_id
        result[i, 1:num_tokens+1] = tokens
        result[i, num_tokens+1] = eot_token_id
    return  result


def get_reduction_mask_fn():
    """获取用于减少掩码的函数"""