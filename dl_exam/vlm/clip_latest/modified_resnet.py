"""
Modified By: Redal
Date: 2025-12-21
Todo: 修改版的ResNet架构ModifiedResNet,在传统ResNet基础上结合了
      抗锯齿下采,注意力池化等优化,兼顾了ResNet的局部特征提取能力
      和自注意力的全局建模能力
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Union
import torch
from torch import nn
from torch.nn import functional as F
from utils import freeze_batch_norm_2d, feature_take_indices


class Bottleneck(nn.Module):
      expansion = 4
      def __init__(self, inplanes:int,
                   planes: int,
                   stride: int=1
                   )->None:
            super().__init__()
            # 所有的卷积层均使用步进stride=1的层并且
            # 在第二层卷积后添加avgpool if stride>1
            self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.act1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.act2 = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

            self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.act3 = nn.ReLU(inplace=True)

            self.downsample = None
            self.stride = stride
            if stride > 1 or inplanes != planes * self.expansion:
                  # 下采样层前接一个平均池化层,后续的卷积层步长为1
                  self.downsample = nn.Sequential(OrderedDict([
                        ('-1', nn.AvgPool2d(stride)),
                        ('0', nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=1, bias=False)),
                        ('1', nn.BatchNorm2d(planes * self.expansion)),]))
      def forward(self, x:torch.Tensor)->torch.Tensor:
            identity = x
            out = self.act1(self.bn1(self.conv1(x)))
            out = self.act2(self.bn2(self.conv2(out)))
            out = self.avgpool(out)
            out = self.bn3(self.conv3(out))
            # 判断是否需要下采样
            if self.downsample is not None:
                  identity = self.downsample(x)
            out += identity
            out = self.act3(out)
            return out


class AttentionPool2d(nn.Module):
      def __init__(self, spacial_dim: int,
                   embed_dim: int,
                   num_heads: int,
                   output_dim: int = None
                   )->None:
            super().__init__()
            self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) * embed_dim ** -0.5)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
            self.num_heads = num_heads
      def forward(self, x:torch.Tensor)->torch.Tensor:
            # N, C, H, W = x.shape
            # x = x.view(N, C, H*W).permute(2, 0, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(2, 0, 1) # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) # (HW+1)NC
            x = x + self.positional_embedding[:, None, :].to(x.dtype)
            x, _ = F.multi_head_attention_forward(
                  query=x, key=x, value=x,
                  embed_dim_to_check=x.shape[-1],
                  num_heads=self.num_heads,
                  q_proj_weight=self.q_proj.weight,
                  k_proj_weight=self.k_proj.weight,
                  v_proj_weight=self.v_proj.weight,
                  in_proj_weight=None,
                  in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],),
                  bias_k=None,
                  bias_v=None,
                  add_zero_attn=False,
                  dropout_p=0.,
                  out_proj_weight=self.c_proj.weight,
                  out_proj_bias=self.c_proj.bias,
                  use_separate_proj_weight=True,
                  training=self.training,
                  need_weights=False,)
            return x[0]


class ModifiedResNet(nn.Module):
      """一个与torchvision中的ResNet类相似但包含以下修改的ResNet类:
      现在有3个“主干”卷积,而非1个,并且使用平均池化代替最大池化
      执行抗锯齿跨步卷积,其中在跨步大于1的卷积前添加一个平均池化
      最后的池化层是QKV注意力机制,而非平均池化"""
      def __init__(self, layers: List[int],
                   output_dim: int,
                   heads: int,
                   image_size, int=224,
                   width: int =64
                   )->None:
            super().__init__()
            self.output_dim = output_dim
            self.image_size = image_size
            # 3层stem卷积
            self.conv1 = nn.Conv2d(3, width//2, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(width//2)
            self.act1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(width//2, width//2, kernel_size=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(width//2)
            self.act2 = nn.ReLU(inplace=True)
            self.conv3 =nn.Conv2d(width//2, width, kernel_size=3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(width)
            self.act3 = nn.ReLU(inplane=True)
            self.avgpool = nn.AvgPool2d(width)
            # 残差层通过_make_layer方法进行构建
            self._inplanes = width
            self.layer1 = self._make_layer(width, layers[0])
            self.layer2 = self._make_layer(width*2,layers[1], stride=2)
            self.layer3 = self._make_layer(width*4, layers[2], stride=2)
            self.layer4 = self._make_layer(width*8, layers[3], stride=2)
            # 计算embed_dim的输出结果
            embed_dim = width * 32 # ResNet标准输出的维度
            self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)
            self.init_parameters()
      def stem(self, x:torch.Tensor)->torch.Tensor:
            """首先完成stem主干卷积的前向传播"""
            x = self.act1(self.bn1(self.conv1(x)))
            x = self.act2(self.bn2(self.conv2(x)))
            x = self.act3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
      def _make_layer(self, planes: int,
                      blocks: int,
                      stride: int =1):
            layers = [Bottleneck(self, self._inplanes, planes, stride)]
            self._inplanes = planes * Bottleneck.expansion
            for _ in range(1, blocks):
                  layers.append(Bottleneck(self._inplanes, planes, stride))
            return nn.Sequential(*layers)
      def init_parameters(self,):
            if self.attnpool is not None:
                  std = self.attnpool.c_proj.in_features ** -0.5
                  nn.init.normal_(self.attnpool.q_proj.weight, std=std)
                  nn.init.normal_(self.attnpool.k_proj.weight, std=std)
                  nn.init.normal_(self.attnpool.v_proj.weight, std=std)
                  nn.init.normal_(self.attnpool.c_proj.weight, std=std)
            for resblock in [self.layer1, self.layer2, self.layer3, self.layer4]:
                  for name, params in resblock.named_parameters():
                        if name.endswith('bn3.weight'):
                              nn.init.zeros_(params)
      def lock(self, unlocked_groups=0, freeze_bn_stats=False):
            """冻结模型大部分参数的梯度计算(即固定参数不更新)
            仅解锁指定数量的参数组让其可训练,TODO:添加可实现部分参数冻结的功能"""
            assert unlocked_groups==0, f'[WARNING] 暂不支持冻结部分参数组,请将unlocked_groups设置为0!'
            for param in self.parameters():
                  param.requires_grad = False
            # 可选冻结BN层的统计量
            if freeze_bn_stats:
                  freeze_batch_norm_2d(self)
      @torch.jit.ignore
      def set_grad_checkpointing(self, enable=True):
            """激活梯度检查点以节省显存,但会减慢训练速度
            TODO: 添加对ModifiedResNet/transformer的梯度检查点"""
            pass
      def forward_intermediates(self, x: torch.Tensor,
                                indices: Optional[Union[int, List[int]]]=None,
                                stop_early:bool=False,
                                normalize_intermediates: bool=False,
                                intermediates_only:bool=False,
                                output_fmt: str='NCHW',
                                output_extra_tokens: bool=False
                                )->Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
            """TODO: 添加对normalize_intermediates和output_extra_tokens的支持"""
            assert output_fmt in ['NCHW'], f'[WARNING] 确保输出的格式必须为NCHW!'
            take_indices, max_index = feature_take_indices(5, indices)
            output = {}
            intermediates = []
            blocks = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
            if torch.jit.is_scripting() or not stop_early:
                  blocks = blocks[:max_index+1]
            for i, block in enumerate(blocks):
                  x = block(x)
                  if i in take_indices:
                        intermediates.append(x)
            output['image_intermediates'] = intermediates
            # 选择输出形式intermediates_only
            if intermediates_only: return output
            pooled = self.attnpool(x)
            output['image_features'] = pooled
            return output
      def forward(self, x:torch.Tensor)->torch.Tensor:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.attnpool(x)
            return x


