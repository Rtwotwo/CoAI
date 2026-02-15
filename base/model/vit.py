"""
Author: Redal
Date: 2025-10-14
Todo: vit.py for creating vision of transformer model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import os
import math
import torch
import torch.nn as nn
from functools import partial
from typing import Union, Type, List, Dict, Any
from typing import Optional, Tuple, Callable, Set


# 实现不同尺寸的Vision Transformer模型架构,保证模型可以被调用
# 并且支持目前ImageNet数据集预训练的模型权重导入, 不是最基础的ViT模型架构!!
# 注意: 该VIT包含DropPath, LayerScale, qk_norm, LayerNorm, 复杂的权重加载逻辑, 多种预训练配置等
class DropPath(nn.Module):
    """随机丢弃路径的实现
    drop_prob(float): 丢弃概率"""
    def __init__(self, drop_prob:float=0.1,
                 eps:float=1e-6,
                 training:bool=True)->None:
        self.drop_prob = drop_prob
        self.eps = eps
        self.training = training
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if not self.training or self.drop_prob==0.0:
            return x
        # 使用随机mask完成随机路径的筛选
        keep_prob = 1.0 -self.drop_prob
        mask = (torch.rand_like(x) < keep_prob).float(x) / keep_prob
        return x * mask


class LayerNorm(nn.Module):
    """Layer Normalization基础的LayerNorm层实现
    normalized_shape (int或tuple): 需要归一化的维度形状
    eps (float): 用于数值稳定的小常数，防止除零错误
    elementwise_affine (bool): 是否使用可学习的缩放参数gamma/weight和偏移参数beta/bias"""
    def __init__(self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        # 将 normalized_shape统一转为tuple格式
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 初始化可学习的参数：weight即gamma用于尺度缩放和bias即beta用于偏移
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            # 注册为None参数(PyTorch模块标准做法)
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查输入张量的最后若干维是否与 normalized_shape 匹配
        if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f'输入张量的最后维度{x.shape[-len(self.normalized_shape):]}与归一化形状{self.normalized_shape} 不匹配')
        ndim = x.ndim
        norm_dims = tuple(range(ndim - len(self.normalized_shape), ndim))
        # 计算均值和方差(注意LayerNorm使用有偏方差, 即unbiased=False)
        # 归一化: (x - mean) / sqrt(var + eps)
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = x.var(dim=norm_dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 如果启用仿射变换, 则应用可学习的缩放和偏移
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm
    

class MLP(nn.Module):
    """使用1x1卷积的多层感知机MLP层的实现-替代FeedForward层
    输入张量的维度和输出维度均为[B, C, H, W]"""
    def __init__(self, in_features:int,
                 hidden_features: Optional[int]=None,
                 out_features:Optional[int]=None,
                 act_layer:Type[nn.Module]=nn.GELU,
                 norm_layer:Type[nn.Module]=nn.BatchNorm2d,
                 drop:float=0.,
                 device=None,
                 dtype=None)->None:
        super().__init__() 
        # 当out_features和hidden_features为None或未被赋值时
        # 会自动使用in_features的值作为默认值
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = norm_layer(in_features)
        # 设置基础两层全连接MLP
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.norm1(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class PatchEmbed(nn.Module):
    """将输入图像划分为多个非重叠的patch并线性嵌入到高维空间
    img_size:输入图像的尺寸;
    patch_size:patch的尺寸;
    in_chans:输入图像的通道数;
    embed_dim:嵌入维度;
    norm_layer:归一化层;
    flatten:是否将patch展平为一维向量"""
    def __init__(self, img_size:int=224, 
                 patch_size:int=16, 
                 in_chans:int=3,
                 embed_dim:int=768)->None:
        super().__init__()
        # 计算patch数量
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 卷积层划分输入图像为patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        # 检查输入图像的尺寸是否与预设尺寸匹配
        B, C, H, W = x.shape
        assert H==self.img_size and W==self.img_size, \
            f'输入的图像尺寸{H}x{W}与预设尺寸{self.img_size}不匹配!!'
        # x:[B, C, H, W]->[B, embed_dim, H//p, W//p]->[B, embed_dim, H//p*W//p]
        # transpose(1,2)则是改变张量的维度顺序为[B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1,2)
        return x
    

class Attention(nn.Module):
    """Transformer的注意力机制实现
    dim:输入通道数; num_heads:注意力头数;
    qkv_bias:如果为True, 给查询、键、值添加一个可学习的偏置;
    qk_norm:如果为True, 对查询和键应用归一化;
    proj_bias:如果为True,给输出投影添加偏置"""
    def __init__(self, dim:int,
                 num_heads:int,
                 qkv_bias:bool=False,
                 qk_norm:bool=False,
                 attn_drop:float=0.,
                 proj_drop:float=0.,
                 norm_layer:Type[nn.Module]=LayerNorm)->None:
        super().__init__()
        # 计算每个注意力头的维度
        assert dim%num_heads==0, f'输入通道数{dim}不能被注意力头数{num_heads}整除!!'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # 初始化查询、键、值映射层
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # 输入张量的维度为[B, N, C]即[B, num_patches, embed_dim]
        B, N, C = x.shape
        # [B, N, C]->[B, N, 3*C]->[B, N, 3, num_heads, head_dim]->[3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # 张量按照第 0 维度(最外层维度)进行拆分
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        # 乘以值并求和输出得到注意力的实际得分
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    """采用预归一化的Transformer-Block块的实现
        dim:输入通道数; num_heads:注意力头数;
        mlp_ratio:MLP 隐藏维度与嵌入维度的比率;
        qkv_bias:如果为True, 给查询、键、值添加一个可学习的偏置;
        qk_norm:如果为True, 对查询和键应用归一化;
        proj_bias:如果为True,给输出投影添加偏置;
        proj_drop:投影 dropout 率;
        attn_drop:注意力 dropout 率;
        init_values:层缩放的初始值;
        drop_path:随机深度率;act_layer:激活层;
        norm_layer:归一化层;mlp_layer:MLP 层"""
    def __init__(self, dim:int, 
                 num_heads:int, 
                 mlp_ratio:float=4.0,
                 qkv_bias:bool=False, 
                 qk_norm:bool=False,
                 scale_attn_norm:bool=False,
                 scale_mlp_norm:bool=False, 
                 proj_bias:bool=True, 
                 proj_drop:float=0.,
                 attn_drop: float=0., 
                 init_values:Optional[float]=None,
                 drop_path: float=0.,
                 act_layer:Type[nn.Module]=nn.GELU,
                 norm_layer:Type[nn.Module]=LayerNorm,
                 mlp_layer: Type[nn.Module]=MLP,
                 device=None,
                 dtype=None)->None:
        super().__init__()
        # 设置Transfo0rmer-Block块,主要涵盖Encoder部分
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm,
                            attn_drop, proj_drop, norm_layer)
        # LayerScale对神经网络层进行自适应缩放
        self.ls1 = nn.Parameter(init_values*torch.ones(dim)) if init_values else None
        self.drop_path1 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), act_layer=act_layer)
        self.ls2 = nn.Parameter(init_values*torch.ones(dim)) if init_values else None
        self.drop_path2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # 注意力机制层同时决定是否使用LayerScale
        if self.ls1 is not None:
            x = x + self.drop_path1(self.ls1 * self.attn(self.norm1(x)))
        else: x = x + self.drop_path1(self.attn(self.norm1(x)))
        # MLP作为VIT-Block块的前馈层并使用LayerScale
        if self.ls2 is not None:
            x = x +self.drop_path2(self.ls2*self.mlp(self.norm2(x)))
        else: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformer(nn.Module):
    """Vision Transformer的实现
    img_size:输入图片尺寸; patch_size:patch尺寸; in_chans:输入通道数;
    embed_dim:嵌入维度; depth:Transformer块数; num_heads:注意力头数; 
    mlp_ratio:MLP  hidden dim 与 embed_dim 的比率; qkv_bias:给查询、键、值添加一个可学习的偏置; 
    qk_norm:对查询和键应用归一化; init_values:层缩放的初始值; class_token:是否使用类标记; 
    no_embed_class:是否不使用嵌入类标记; drop_path_rate:随机drop丢弃"""
    def __init__(self, img_size:int=224,
                 patch_size:int=16,
                 in_chans:int=3,
                 num_classes:int=1000, 
                 embed_dim:int=768,
                 depth:int=12,
                 num_heads:int=12,
                 mlp_ratio:float=4.0,
                 qkv_bias:bool=True,
                 qk_norm:bool=False, 
                 init_values:int=None,
                 class_token:bool=True,
                 no_embed_class:bool=False,
                 drop_rate:float=0.,
                 attn_drop_rate:float=0.,
                 drop_path_rate:float=0.,
                 norm_layer:Type[nn.Module]=LayerNorm,
                 act_layer:Type[nn.Module]=nn.GELU,
                 block_fn:Type[nn.Module]=Block)->None:
        super().__init__()
        # partial(nn.LayerNorm, eps=1e-6)预设置eps=1e-6参数的LayerNorm层构造器
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans,embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # 以匹配PatchEmbed输出形状[B, num_patches, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        # 位置嵌入层设计
        embed_len = num_patches if no_embed_class else num_patches+(1 if class_token else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_norm=qk_norm, proj_drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=dpr[i], 
                     norm_layer=norm_layer, act_layer=act_layer, init_values=init_values)
                    for i in range(depth)])
        # 归一化并且设置分类头self.head(注意该head并非Attention中的注意力头)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes>0 else nn.Identity()
        if self.cls_token is not None:
            # 用于将张量初始化为符合正态分布的随机值
            nn.init.normal_(self.cls_token, std=1e-6)
        # trunc_normal_初始化对位置嵌入pos_embed使用截断正态分布初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    def _init_weights(self, m:nn.Module)->None:
        """初始化权重: 批量初始化自定义网络层"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # constant_专用于偏置初始为常数
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward_features(self, x:torch.Tensor)->torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            # 扩展后类别令牌的形状将从[1, 1, C]变为[B, 1, C]
            # 以满足PatchEmbed输出的形状进行拼接[B, N, C]
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.norm(self.blocks(x))
        return x[:, 0] if self.cls_token is not None else x.mean(dim=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# 预训练模型配置以及相关的函数实现
# 包括vit的tiny, small, base, large, huge等版本
VIT_CONFIGS = {
    'vit_tiny_patch16_224':{'embed_dim':192, 'depth':12, 'num_heads':3, 'mlp_ratio':4,
                            'url':'https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'},
    'vit_small_patch16_224':{'embed_dim':384, 'depth':12, 'num_heads':6, 'mlp_ratio':4,
                             'url':'https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'},
    'vit_base_patch16_224':{'embed_dim':768, 'depth':12, 'num_heads':12, 'mlp_ratio':4,
                            'url':'https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'},
    'vit_large_patch16_224':{'embed_dim':1024, 'depth':24, 'num_heads':16, 'mlp_ratio':4,
                             'url':'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'},
    'vit_huge_patch16_224':{'embed_dim':1280, 'depth':32, 'num_heads':16, 'mlp_ratio':4, 'patch_size':14, 
                            'url':'https://storage.googleapis.com/vit_models/augreg/H_14-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'}}


def load_weights_from_npz(model, url_or_path, num_classes=1000):
    """权重加载工具,支持.npz格式的权重模型"""
    try:
        import numpy as np
        from urllib.request import urlopen
        from io import BytesIO
    except ImportError:
        raise ImportError('加载预训练模型需要安装numpy!!')
    if url_or_path.startswith('http'):
        print(f'正在从{url_or_path}下载预训练模型权重...')
        data = urlopen(url_or_path).read()
        weights = np.load(BytesIO(data))
    else:
        weights = np.load(url_or_path)
    # npz权重读取完毕,加载进入模型中
    state_dict = {}
    for k,v in weights.items():
        if k=='cls': continue
        k = k.replace('Transformer/posembed_input/pos_embedding', 'pos_embed')
        k = k.replace('embedding', 'patch_embed.proj')
        k = k.replace('cls', 'cls_token')
        k = k.replace('Transformer/encoderblock_', 'blocks.')
        k = k.replace('LayerNorm_0', 'norm1')
        k = k.replace('LayerNorm_2', 'norm2')
        k = k.replace('MlpBlock_3/Dense_0', 'mlp.fc1')
        k = k.replace('MlpBlock_3/Dense_1', 'mlp.fc2')
        k = k.replace('MultiHeadDotProductAttention_1/query', 'attn.qkv')
        k = k.replace('MultiHeadDotProductAttention_1/key', 'attn.qkv')
        k = k.replace('MultiHeadDotProductAttention_1/value', 'attn.qkv')
        k = k.replace('MultiHeadDotProductAttention_1/out', 'attn.proj')
        k = k.replace('head', 'head')
        k = k.replace('/', '.')
        # 若预训练权重分类头与自定义不匹配, 则跳过
        if 'head' in k and v.shape[0]!=num_classes: continue
        # 处理qkv权重形状
        if 'attn.qkv.weight' in k:
            if k not in state_dict:
                state_dict[k] = torch.from_numpy(v).float()
            else: 
                state_dict[k] = torch.cat([state_dict[k], torch.from_numpy(v).float()], dim=0)
            continue
        if 'attn.qkv.bias' in k:
            if k not in state_dict:
                state_dict[k] = torch.from_numpy(v).float()
            else:
                state_dict[k] = torch.cat([state_dict[k], torch.from_numpy(v).float()], dim=0)
                continue
        state_dict[k]=torch.from_numpy(v).float()
    # 针对qkv的特殊处理
    new_state_dict = {}
    for k,v in state_dict.items():
        if 'attn.qkv' in k:
            if k.endswith('weight'):
                base_k = k.replace('.weight', '')
                qkv_w = torch.cat([
                    state_dict[base_k + '.weight'][:v.shape[0]//3],
                    state_dict[base_k + '.weight'][v.shape[0]//3:2*v.shape[0]//3],
                    state_dict[base_k + '.weight'][2*v.shape[0]//3:]
                ], dim=0)
                new_state_dict[base_k + '.weight'] = qkv_w
            elif k.endswith('bias'):
                base_k = k.replace('.bias', '')
                qkv_b = torch.cat([
                    state_dict[base_k + '.bias'][:v.shape[0]//3],
                    state_dict[base_k + '.bias'][v.shape[0]//3:2*v.shape[0]//3],
                    state_dict[base_k + '.bias'][2*v.shape[0]//3:]
                ], dim=0)
                new_state_dict[base_k + '.bias'] = qkv_b
        else:
            new_state_dict[k] = v
    # 去除多余的键值
    model_state = model.state_dict()
    matched_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(matched_state_dict, strict=False)
    print(f'从预训练权重中加载{len(matched_state_dict)}层网络')


def create_vit(model_name:str, 
               pretrained:bool=False, 
               num_classes:int=1000,
               img_size:int=224,
               in_chans:int=3,
               **kwargs:Dict[str, Any])->VisionTransformer:
    if model_name not in VIT_CONFIGS:
        raise ValueError(f'模型{model_name}不接受支持,请参照VIT_CONFIGS选择!!\n{list(VIT_CONFIGS.keys())}')
    config = VIT_CONFIGS[model_name].copy()
    # 尝试从CONFIGS取值, 若不存在则使用默认值
    patch_size = config.pop('patch_size', 16)
    url = config.pop('url', None)
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        **config,
        **kwargs)
    if pretrained:
        if url is None: raise ValueError(f'没有预训练权重支持该模型{model_name}')
        load_weights_from_npz(model, url, num_classes=num_classes)
    return model
    

# 创建快速模型实例化函数
# 主要包括vit的tiny, small, base, large, huge版本
def vit_tiny(pretrained:str=False, **kwargs:Dict[str, Any]):
    return create_vit('vit_tiny_patch16_224', pretrained=pretrained, **kwargs)
def vit_small(pretrained:str=False, **kwargs:Dict[str, Any]):
    return create_vit('vit_small_patch16_224', pretrained=pretrained, **kwargs)
def vit_base(pretrained:str=False, **kwargs:Dict[str, Any]):
    return create_vit('vit_base_patch16_224', pretrained=pretrained, **kwargs)
def vit_large(pretrained:str=False, **kwargs:Dict[str, Any]):
    return create_vit('vit_large_patch16_224', pretrained=pretrained, **kwargs)
def vit_huge(pretrained:str=False, **kwargs:Dict[str, Any]):
    return create_vit('vit_huge_patch16_224', pretrained=pretrained, **kwargs)


if __name__ == '__main__':
    # 创建自定义模型
    vit_tiny = vit_tiny(pretrained=True)
    # vit_small = vit_small()
    # vit_base = vit_base()
    # vit_large = vit_large()
    # vit_huge = vit_huge()
    print(vit_huge)
