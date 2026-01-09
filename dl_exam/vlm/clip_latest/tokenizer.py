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


# ----------------------------------------------------------------------
# SimpleTokenizer:
# 基于BPE(字节对编码)的轻量级文本分词器(SimpleTokenizer),核心用于将自然语言
# 文本转换为模型可处理的token序列,适配大语言模型的输入格式
# ----------------------------------------------------------------------
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
        # encoder和decoder的设置目的是将获取到的bpe编码转成token ID
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
    def bpe(self, token):
        """BPE字节对编码算法的核心实现代码,用于将输入token拆分为BPE子词
        功能:输入一个原始 token如单词apple,输出其BPE编码后的子词序列
        (如 “app le</w>”，</w>是词尾标记),同时用cache缓存结果避免重复计算"""
        if token in self.cache: return self.cache[token]
        word = tuple(token[-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)
        if not pairs: return token+'</w>'
        # BPE合并循环-预训练的bigram优先级字典
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_tokens.get(pair, float('inf')))
            if bigram not in self.bpe_tokens: break
            first, second = bigram
            new_word = []
            i = 0
            while i<len(word):
                # 将word中的词汇进行筛选出不存在bpe词汇表的情况
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break
                # 按照优先级的情况来合并相邻的两个token
                if word[i]==first and i<len(word)-1 and word[i+1]==second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word)==1: break
            else: pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
    def encode(self, text):
        bpe_tokens = []
        # 1.将原始纯文本全部转成小写字符
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            # 2.将token转换为utf-8编码的字节序列
            token = ''.join(self.byte_encoder(b) for b in token.encode('utf-8'))
            # 3.对token进行BPE编码,并转换为token ID
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens
    def decode(self, tokens):
        text = ''.join(self.decoder[token] for token in tokens)
        text = bytearray([self.byte_decoder(b) for b in text]).decode('utf-8', errors='replace').replace('</w>', ' ')
        return text
    def __call__(self, texts: Union[str, List[str]],
                 context_length: Optional[int] = None,
                 )->torch.LongTensor:
        if isinstance(texts, str): texts = [texts]
        context_length = context_length or self.context_length
        assert context_length, f"[ERROR] 请设置合理的context_length的值!"
        if self.reduction_fn is not None:
            return self.reduction_fn(texts, 
                                     context_length=context_length,
                                     sot_token_id=self.sot_token_id,
                                     eot_token_id=self.eot_token_id,
                                     encode_fn=self.encode)
        # 对tokens进行正常的文本->文本索引->bpe编码->token
        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, token in enumerate(all_tokens):
            if len(token) > context_length:
                token = token[:context_length] #truncate
                token[-1] = self.eot_token_id
            result[i, len(token)] = torch.tensor(token)
        return result
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
    """带随机掩码/截断的文本tokenize工具,核心作用是将文本编码为固定长度的
    token序列,适配模型输入格式,同时支持超长文本的随机截断(可选有序保留)"""
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        # context_length-2留给sot和eot token
        if num_tokens > context_length-2: 
            num_keep = context_length - 2
            # 对超出的序列进行截断并且打乱排序
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                 indices = indices.msort()
            tokens =  tokens[indices]
            num_tokens = num_keep
        # 将正常的序列放入context_length中
        result[i, 0] = sot_token_id
        result[i, 1:num_tokens+1] = tokens
        result[i, num_tokens+1] = eot_token_id
    return  result


def simple_mask_tokenize(texts: Union[str, List[str]],
                         context_length: int,
                         sot_token_id: int, 
                         eot_token_id: int,
                         encode_fn:  Callable):
    """文本的简单掩码/填充预处理工具,核心作用是将输入文本
    (单条或多条)编码后,统一处理成固定长度的token序列"""
    all_tokens =  [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        if num_tokens > context_length-2:
            num_keep = context_length -2
            # 掩码并且进行截断
            start_index = random.randint(0, num_tokens - num_keep)
            tokens = tokens[start_index: start_index + num_keep]
        tokens = [sot_token_id] + tokens + [eot_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def syntax_mask_tokenize(texts: Union[str, List[str]],
                         context_length: int,
                         sot_token_id: int,
                         eot_token_id: int,
                         encode_fn: Callable
                         )->torch.LongTensor:
    """带语法掩码的文本tokenize工具,核心作用是将文本编码为固定长度的
    token序列,适配模型输入格式,同时支持超长文本的随机截断(可选有序保留)"""
    import nltk
    global _nltk_init
    if not _nltk_init:
        # 如果首次运行则下载相应数据
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        _nltk_init = True
    def get_order(x):
        """根据字符串前缀分类并返回对应优先级序号"""
        if x.startswith('NN'): return 1
        elif x.startswith('JJ'): return 2
        elif x.startswith('VB'): return 3
        else: return 4
    # 对输入文本集合texts进行词性优先级采样
    new_texts = []
    for text in texts:
        # 分词+词性标注
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        # 词性优先级排序与筛选
        order_list = [get_order(tag) for _, tag in pos_tags]
        sorted_ids = np.argsort(np.array(order_list))
        sample_ids = sorted(sorted_ids[:context_length-2]) # need 2 for sot and eot
        # 重构文本,np.take根据筛选后的索引,从原分词列表中提取关键tokens
        sample_tokens = np.take(np.array(list_tokens), sample_ids, axis=0)
        new_text = ''
        for token in sample_tokens:
            new_text = new_text + str(token) + ' '
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts
    # 进行常规的tokenize处理
    all_tokens = [[sot_token_id]+encode_fn(text)+[eot_token_id] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        # 仍然需要需要第一阶段,有些词汇会出现两个token
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot_token_id
        result[i,:len(tokens)] = torch.tensor(tokens)
    return result


def get_reduction_mask_fn(type: str):
    """获取用于减少掩码的函数"""
    assert type in ('simple', 'random', 'shuffle', 'syntax'), \
            f'[WARNING] Invalid reduction mask type: {type}!'
    if type == 'simple': return simple_mask_tokenize
    elif type == 'random': return random_mask_tokenize
    elif type == 'shuffle': return partial(random_mask_tokenize, shuffle=True)
    elif type == 'syntax': return syntax_mask_tokenize
    else: assert False, f'[ERROR] Unkonwn type {type}!'


# ----------------------------------------------------------------------
# HFTokenizer:
# 基于HuggingFace Transformers库的文本分词器,用于将自然语言文本转换为模型可处理的token序列
# ----------------------------------------------------------------------
class HFTokenizer:
    def __init__(self, tokenizer_name:str,
                 context_length: Optional[int]=DEFAULT_CONTEXT_LENGTH,
                 clean: str='whitespace',
                 strip_sep_token: bool=False,
                 language: Optional[str]=None,
                 cache_dir: Optional[str]=None,
                 tokenizer_mode: Optional[str]=None, # None, 'clips'
                 **kwargs)->None:
        # 初始化相关参数
        self.tokenizer_name = tokenizer_name or ''
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token
        # NOTE: 下列代码为自定义的tokenizer并且初始化
        if self.tokenizer_name == 'bert_clips':
            self.special_tokens = {
                'bos_token': 1,
                'eos_token': 2,
                'cls_token': 101,
                'pad_token': 0}
            # 对于Bert的CLIP的模式使用vocab文件
            from tokenizers import BertWordPieceTokenizer
            if tokenizer_mode.startswith('hf-hub'):
                # 构建下载文件链接 
                from huggingface_hub import hf_hub_download
                repo_url = tokenizer_name[7:]
                parts = repo_url.split('/')
                filename = parts[-1]
                repo_id = '/'.join(parts[:-1])
                # 下载需要的vocab file文件至指定的文件缓存文件夹
                vocab_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
                self.tokenizer = BertWordPieceTokenizer(lowercase=True)
                self.tokenizer = self.tokenizer.from_files(vocab_file)
            else:
                # 确保vocab文件存在于本地文件夹的地址
                # 此时tokenizer_name是本地路径地址,而非hugging_face链接
                self.tokenizer = BertWordPieceTokenizer(lowercase=True)
                self.tokenizer = self.tokenizer.from_files(tokenizer_name)
        # 下列标准的HuggingFace tokenizer初始化
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
        # Set language function if available
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)
    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)
    def __call__(self, texts: Union[str, List[str]],
                 context_length: Optional[int]=None
                 )->torch.Tensor:
        """"""
        if isinstance(texts, str): texts = [texts]
        context_length = context_length or self.context_length
        assert context_length, f'[ERROR] 请设置合理的context_length!'
        # 清洗文本: 主要通过删除文本之间的空白
        texts = [self.clean_fn(text) for text in texts]
        # 处理不同tokenization的模式
        if self.tokenizer_name == 'clips':
            return self._clips_tokenize(texts, context_length)
        else:
            # 标准的tokenization处理流程
            input_ids = self.tokenizer.batch_encode_plus(
                texts, 
                context_length = context_length,
                add_special_tokens = False,
                padding=False,
                truncation=False,
                return_tensors=None).input_ids
            if self.strip_sep_token:
                input_ids = torch.where(
                        input_ids == self.tokenizer.strip_sep_token_id,
                        torch.zeros_like(input_ids),
                        input_ids)
            return input_ids
    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn(f'[WARNING]  {self.tokenizer_name} does not support set_language()!')

    def _clips_tokenize(self, texts: List[str], 
                        context_length:int
                        )->torch.Tensor:
        """专门用于HFTokenzier里面针对CLIP模型的tokenizer分词器
        提供CLIP和Standard的两种标准的分词模式"""
        # 基础分词(无特殊token)
        encoded_outputs = self.tokenizer.batch_encode_plus(
                        texts,
                        add_special_tokens=False, # 不自动加bos/eos/cls/pad
                        padding=False,
                        truncation=False,    # 进行截断处理
                        return_tensors=None) # 返回原生list,并非tensor
        encoded = []
        for tokens in encoded_outputs['input_ids']:
            tokens = tokens[:context_length - 3]
            tokens = [self.tokenizer.bos_token_id] + tokens + [self.eos_token_id]
            encoded.append(tokens)
        # 创建输出的结果tensor并且处理padding+cls token
        result = torch.zeros(len(encoded), context_length, dtype=torch.long)
        for i, tokens in enumerate(encoded):
            # 同时对tokens的序列进行padding和cls标记处理
            padded_tokens = self._pad_and_cls_token(
                        tokens, 
                        max_length = context_length,
                        pad_token_id = self.tokenizer.pad_token_id,
                        cls_token_id= self.tokenizer.cls_token_id,)
            result[i, :len(padded_tokens)] = torch.tensor(padded_tokens)
        return result
    def _pad_and_cls_token(self, tokens: List[int], 
                           max_length: int,
                           pad_token_id: int=0,
                           cls_token_id: int=101
                           )->List[int]:
        """针对提出的文本序列token化进行tokens的padding和clstoken处理"""
        if len(tokens) > max_length -1:
            tokens = tokens[:max_length - 1]
        # 添加padding直到tokens的max_length -1的位置
        if len(tokens) < max_length - 1:
            tokens = tokens + [pad_token_id]*(max_length - 1 -len(tokens))
        # 在添加cls token在tokens的末尾
        tokens = tokens + [cls_token_id]
        return tokens 


class SigLipToeknizer:
    # 给定指定的vocab_files进行下载
    VOCAB_FILES = { # english, vocab_size=32_000
                    "c4-en": "http://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",
                    # used in multilingual models (mT5, PaLI), vocab_size=250_000
                    "mc4": "http://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
                    # used in SigLIP2 models, vocab_size=256000
                    "gemma": "http://storage.googleapis.com/big_vision/gemma_tokenizer.model",}
    def __init__(self, tokenizer_name: str,
                 context_length: int=64):
        """"""