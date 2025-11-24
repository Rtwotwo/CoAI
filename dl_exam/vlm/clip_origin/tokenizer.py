"""
Author: Redal
Date: 2025-11-04
Todo: tokenizer.py to map text to tokens
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import os
import gzip
import html
import ftfy
import regex as re
from functools import lru_cache


@lru_cache()
def default_bpe():
    """获取bpe_simple_vocab_16e6.txt.gz的文件路径"""
    bpe_dir = os.path.dirname(os.path.abspath(__file__))
    bpe_path = os.path.join(bpe_dir, "bpe_simple_vocab_16e6.txt.gz")
    return bpe_path


@lru_cache()
def bytes_to_unicode():
    """返回utf-8字节列表和对应的unicode字符串列表,可逆的bpe编码适用于unicode字符串,
    这意味着,如果你想避免UNK未登录词,词汇表中需要包含大量的unicode字符,
    当你处理大约100亿个令牌的数据集时,为了达到不错的覆盖率,
    这在你常规的bpe词汇表(比如说32000个)中占比相当大,
    为了避免这种情况,我们需要utf-8字节和unicode字符串之间的查找表,
    并且要避免映射到bpe编码会处理出错的空白字符/控制字符"""
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("i"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
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
    """返回一个单词中的符号对集合,
    单词以符号元组的形式表示(符号是可变长度的字符串)"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """用于对文本进行基础清洗处理"""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """清理文本中的空白字符,使文本格式更规范"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class Tokenizer(object):
    def __init__(self, bpe_path:str=default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)
        if not pairs:
            return token+'</w>'
        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens
    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
