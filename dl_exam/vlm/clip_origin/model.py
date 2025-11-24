"""
Author: Redal
Date: 2025-11-04
Todo: __init__.py for vlm tasks
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, Union, List


class Bottleneck(nn.Module):
    # 定义ResNet的BottleNeck结构的扩展系数
    expansion = 4
    def __init__(self, 
                 inplanes:int, 
                 planes:int, 
                 stride:int=1
                 )->None:
        # 所有卷积层的步进均为1
        # 当步进stride>1时, 会设置AvgPool2d层作为下采样
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride>1 else nn.Identity()

        self.conv3 = nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True) # 用于残差连接再激活
        # 初始化downsample和stride参数配置
        self.downsample = None
        self.stride = stride
        # 当条件成立时, 配置下采样层进行降采样, 用于残差连接的identity张量
        if self.stride>1 or inplanes!=planes*Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ('-1', nn.AvgPool2d(stride)),
                ('0', nn.Conv2d(inplanes, planes*self.expansion)),
                ('1', nn.BatchNorm2d(planes*self.expansion))]))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity =self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out
    

class AttentionPool2d(nn.Module):
    """基于注意力机制的2D特征池化操作-直接分割后的patch:
    通过自注意力机制,让类别token学习对所有空间位置特征的加权聚合,
    从而实现具有注意力机制的池化操作，比传统的平均/最大池化更具判别性
    spacial_dim: 输入特征图的空间维度,可能是H/W"""
    def __init__(self, 
                 spacial_dim:int,
                 embed_dim:int,
                 num_heads:int,
                 output_dim:int=None,
                 )->None:
        super().__init__()
        # spacial_dim²表示空间位置数量, +1通常用于添加一个特殊的类别token, embed_dim表示嵌入维度并作出归一化
        # 注意special_dim**2针对裁剪的patch块做的处理,转为[patch_size*patch_size+cls_token, embed_dim]的形状
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x.flatten(start_dim=2).permute(2, 0, 1) # Adjust NCHW to (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) # Add class token (HW+1)NC
        # [:, None, :]通过None增加一个维度,调整位置嵌入的形状,使其能与x的维度匹配
        x = x + self.positional_embedding[:, None, :].to(x.dtype) # Maintain shape (HW+1)NC
        # 直接调用torch提供的MHA的Forward函数
        # x[:1]是为了实现类别cls_token对所有空间特征的注意力聚合操作
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x.squeeze(0)
    

class ModifiedResNet(nn.Module):
    """与torchvision中的ResNet类相似但包含以下更改的ResNet类:
    现在有3个主干卷积,而非1个,并且使用平均池化而非最大池化,
    执行抗锯齿跨步卷积,其中在跨步大于1的卷积前添加一个平均池化,
    最终的池化层是QKV注意力机制AttentionPool2d,而非平均池化"""
    def __init__(self,
                 layers: List[int],
                 output_dim:int,
                 heads:int,
                 input_resolution:int=224,
                 width:int=64)->None:
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        # 3层卷积层的主干卷积
        self.conv1 = nn.Conv2d(3, width//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width//2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width//2, width//2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width//2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # 残差模块层
        self._inplanes = width # 在构建过程中使用的可变变量
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32 # ResNet特征维度->layer4输出的512//32=16为patch_size
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
    def _make_layer(self, planes:int, blocks:int, stride:int=1):
        """注册每层的ResNet的BottleNeck的层数由blocks参数决定
            因此layer1-4使用的BottleNeck的层数不会相同"""
        layers = [Bottleneck(self.inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        def stem(x:torch.Tensor)->torch.Tensor:
            # 首先完成3层主干卷积
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    """子类化torch的LayerNorm以处理fp16
    确保模块内部计算使用特定精度float32,同时保持输入输出数据类型一致,
    避免类型不匹配导致的错误,常见于需要控制数值精度的场景"""
    def forward(self, x:torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """相比标准GELU计算更高效, 同时能保持类似的性能
    torch.sigmoid(1.702*x)计算Sigmoid函数,其中1.702是经验常数"""
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    """Transformer架构中的基本构建块残差注意力块
    Residual Attention Block需要的MHA+add&norm, MLP+add&norm"""
    def __init__(self, 
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor=None
                 )->None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, d_model * 4)),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    def attention(self, x:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """仅设置Vision Transformer架构中的Encoder模块累积层数
    后续文本和图像的token通过Encoder模块进行编码, 输出为CLS_token"""
    def __init__(self, 
                 width: int,
                 layers: int, 
                 heads: int,
                 attn_mask: torch.Tensor=None
                 )->None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
    def forward(self, x:torch.Tensor)->torch.Tensor:
        self.resblocks(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, 
                 input_resolution:int,
                 patch_size:int,
                 width:int,
                 layers:int,
                 heads:int,
                 output_dim:int
                 )->None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = self.output_dim
        # 使用stride=patch_size进行分块编码
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, 
                    kernel_size=patch_size, stride=patch_size,bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x) # 形状是[*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1) # # 形状是[*, width, grid*grid]
        x = x.permute(0, 2, 1)
        # 形状为[*, grid ** 2 + 1, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + 
                torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.type, 
                device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                # 视觉变量
                image_resolution: int,
                vision_layers: Union[Tuple[int, int, int, int], int],
                vision_width: int, 
                vision_patch_size: int,
                # 文本变量
                context_length: int,
                vocab_size: int,
                transformer_width: int,
                transformer_heads: int,
                transformer_layers: int
                )->None:
        super().__init__()
        self.context_length = context_length
        # 这里通过vision_layers的类型来选择使用ModifiedResNet
        # 还是VisionTranformer作为CLIP的编码器部分
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                        layers=vision_layers,
                        output_dim=embed_dim,
                        heads=vision_heads,
                        input_resolution=image_resolution,
                        width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.toekn_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
    def initialize_parameters(self)->None:
        nn.init.normal_(self.toekn_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            # ResNet模型中特定的BatchNorm层参数进行初始化, 有针对得初始化BN层得权重
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    def build_attention_mask(self):
        # 使用因果掩码机制-惰性创建因果注意力掩码, 视觉标记之间采用完全注意力机制
        # PyTorch使用加性注意力掩码; 用-inf无穷大填充
        # 最终得到的掩码矩阵中，对于位置 i 的元素，只能看到 i 及之前位置的元素（对应掩码为 0 的区域），无法看到 i 之后的元素（对应掩码为 - inf 的区域），实现了 "因果注意力" 机制。
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1) # 将下三角部分置零
        return mask
    @property
    def dtype(self):
        # 获取视觉visual部分的conv1的weight数据类型float32/64
        return self.visual.conv1.weight.dtype
    def encode_image(self, image):
        # 将输入图像的数据类型转换为与模型权重相同的类型
        # 编码图像前确保输入数据类型与模型权重一致, 避免类型不匹配错误
        return self.visual(image.type(self.dtype))
    def encode_text(self, text):
        # [batch_size, n_ctx, d_model]即NLD
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x形状为[batch_size, n_ctx, transformer.width]
        # 从eot嵌入中提取特征(eot_token是每个序列中最大的数字)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # 特征归一化: 用特征向量除以自身按行的L2范数, 将所有特征向量标准化到单位球面上
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # 将余弦相似性作为概率输出
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # 此时image和text的形状是[gobal_batch_size, gloabl_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model:nn.Module):
    """将模型的参数转为fp16精度"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']], 'in_proj_bias', 'bias_k', 'bias_v']:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", 'proj']:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


def build_model(state_dict:dict):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f'visual.layer{b}'))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width =  state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64 
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswidth("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        # 视觉部分visual
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        # 文本变量context
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers)
    for key in ["input_resolution", "context_length"]:
        if key in state_dict:
            del state_dict[key]
    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

