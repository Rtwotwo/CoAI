"""
Modified By: Redal
Date: 2025-12-03
Todo: HuggingFace transformers模型用于CLIP model的text tower构建,
      让HuggingFace文本模型能"融入 CLIP",解决"兼容性问题"
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import re
import torch
import torch.nn as nn
from torch import TensorType
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
                        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e: 
    transformers = None
    class BaseModelOutput: pass
    class PretrainedConfig: pass
from hf_configs import arch_dict
# 全局类别注册器
_POOLERS = {}


def _camel2snake(s:str):
    """将驼峰命名法(CamelCase)的字符串转换为蛇形命名法(snake_case)
    (?<!^)负向逆序环视,(?=[A-Z])正向顺序环视,将上述匹配到的位置替换为下划线_
    example: CamelCase -> Camel_Case -> camel_case"""
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def register_pooler(cls):
    """将被装饰的池化器pooler类注册到一个全局的字典_POOLERS"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """均值池化器:模型输出的序列级隐藏状态token-level压缩为句子/文本级的向量sentence-level
    last_hidden_state: [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len], Return [batch_size, hidden_size]"""
    def forward(self, x:BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keep_dim=True)
    

@register_pooler
class MaxPooler(nn.Module):
    """最大池化器:保留每个特征维度上的最大值,能捕捉序列中最显著的特征,但对噪声较敏感
    last_hidden_state: [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len], Return [batch_size, hidden_size]"""
    def forward(self, x:BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsequeeze(-1), -torch.inf)
        return masked_output.max(dim=1).values
    

@register_pooler
class ClsPooler(nn.Module):
    """类别token池化器:从模型输出中提取CLS token对应的特征
    Return: [batch_size, hidden_dim]"""
    def __init__(self, use_pooler_output: bool = True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output
    def forward(self, x:BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and isinstance(x, (BaseModelOutputWithPooling, 
            BaseModelOutputWithPoolingAndCrossAttentions)) and (x.pooler_output is not None)):
            return x.pooler_output
        return x.last_hidden_state[:, self.cls_token_position, :]
    

@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling:基于CLS Token的池化层
    这与上面的ClsPooler(use_pooler_output=False)等效
    returen: [batch_size, hidden_dim] """
    def __init__(self,):
        super().__init__()
        self.cls_token_position = 0
    def forward(self, x:BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:,self.cls_token_position,:]
    

class HFTextEncoder(nn.Module):
    """把任意HuggingFace预训练文本编码器(如BERT,RoBERTa,DeBERTa,XLM-R等)
    封装成一个统一接口的模块,用于多模态模型(比如CLIP,OpenCLIP)中的文本编码部分"""
    output_tokens: torch.jit.Final[bool]
    def __init__(self, model_name_or_path: str,
                 output_dim: int, 
                 config: PretrainedConfig = None,
                 pooler_type: str=None,
                 proj_type: str=None,
                 pretrained: bool=True,
                 output_tokens: bool=False
                 )->None:
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        uses_transformer_pooler = (pooler_type == 'cls_pooler')
        if transformers is None:
            raise RuntimeError("[WARNING] 请使用'pip install transformers'来使用预训练的huggingface模型!")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (
                AutoModel.from_config, self.config)
            # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
        if pooler_type is None:  # get default arch pooler
            pooler_type = (arch_dict[self.config.model_type]["pooler"])

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),)
    def foward(self, x: TensorType) -> TensorType:
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :] 
            if type(self.pooler) == ClsPooler 
            else out.last_hidden_state)
        if self.output_tokens:
            return projected, tokens
        return projected
    def lock(self, unlocked_layers: int=0, freeze_layer_norm: bool=True):
        """冻结部分参数,默认冻结所有参数,但保留CLS token的参数"""
        if not unlocked_layers:
            for name,param in self.named_parameters():
                param.requires_grad = (not freeze_layer_norm) if 'LayerNorm' in name.split('.') else False
            return 
        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]['config_names']['layer_attr'])
        print(f'[INFO] unlocked {unlocked_layers}/{len(layer_list)+1} of the hf model')
        embeddings = getattr(self.transformer, arch_dict[self.config.model_type['config_names']['token_embeddings_attr']])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # 释放层
        for module in modules:
            for name, param in module.named_parameters():
                param.requires_grad = (not freeze_layer_norm) if 'LayerNorm' in name.split('.') else False
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """激活梯度检查点的grad_checkpointing以节省显存,但会减慢训练速度"""
        self.transformer.set_grad_checkpointing(enable)
    def init_parameters(self,):
        """TODO: 完成对HF Model模型参数初始化"""
        pass