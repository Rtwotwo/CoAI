"""
Author: Redal
Date: 2025-10-14
Todo: resnet.py for creating ResNet model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, List, Dict, Any

# 预定义各类尺寸的ResNet版本, 包括18, 34, 50, 101, 152五种尺寸
__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """定义3x3带padding的卷积层
    inplanes输入特征图的通道数
    out_planes输出特征图的通道数
    stride卷积核在输入特征图上滑动的步长
    groups控制输入和输出通道之间的连接方式,groups=1所有输入通道都与所有输出通道连接
    dilation空洞率/膨胀率用于空洞卷积,卷积核元素之间会有间隔,增大感受野而不增加参数量"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """定义1x1带padding的卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 定义ResNet基本卷积块
class BasicBlock(nn.Module):
    # 定义通道扩展系数expansion
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, 
                 downsample=None, groups=1, base_width=64, 
                 dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None: norm_layer == nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        # 当步长(stride)不等于1时,self.conv1层和self.downsample层都会对输入进行下采样
        # 标准卷积+BN+激活层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # 初始化downsample和stride
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        # 保存原始输入,用于后续的残差连接
        identity = x
        # 残差卷积块标准实现
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 如果需要下采样(调整维度或步长),对输入进行处理
        if self.downsample is not None:
            identity  =self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ResNet中的瓶颈(Bottleneck)模块的实现
class Bottleneck(nn.Module):
    # expansion定义了输出通道数相对于输入通道数的扩展比例
    expansion = 4
    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) *groups
        # 当步长不等于1时,self.conv2层和self.downsample层都会对输入进行下采样
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 判断是否残差链接
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    

# 定义完整的ResNet网络模型
class ResNet(nn.Module):
    def __init__(self, block: Union[Type[BasicBlock], Type[Bottleneck]], layers: List[int], 
                 num_classes=1000, zero_init_residual: bool=False,groups:int=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 元组中的每个元素都表示是否应该用膨胀卷积替代 2x2 的步长卷积
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or " \
            "a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group
        # 定义初始卷积层和池化层, 已解决输入的3通道的图像
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 定义ResNet的4个残差模块,每个模块包含多个残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """用于构建ResNet网络层的一系列残差块
        block使用的块类型Bottleneck或BasicBlock
        planes该层的基准通道数
        blocks该层包含的块数量
        stride第一个块的步长
        dilate是否使用膨胀卷积"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # 判断是否进行膨胀卷积->更新膨胀率
        if dilate:
            self.dilation *= stride
            stride = 1
        # 判断是否进行下采样->以匹配输入通道数量
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes*block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)
    def forward(self, x):
        # 前向传播过程中注意使用的池化层的区别：
        # 第一层conv1直接maxpool, 后续残差结束avgpool
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        # 添加ResNet18的残差连接
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # 特征图降低维度输出
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 规定ResNet尺寸，便于加载目前主流的ResNet预训练模型
def _resnet(arch: str, block: Union[Type[BasicBlock], Type[Bottleneck]], layers: List[int], 
            pretrained: bool, progress: bool, **kwargs: Dict[str, Any]):
    """工厂函数主要用于torch官网上下载预训练模型权重"""
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # 手动定义model_urls, 避免ImportError兼容新版本torchvision
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',}
        from torch.hub import load_state_dict_from_url
        assert arch in model_urls, f"No checkpoint URL available for {arch}"
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model
"""ResNet-18模型源自《深度残差学习用于图像识别》https://arxiv.org/pdf/1512.03385.pdf
pretrained: 如果为 True, 则返回在 ImageNet 上预训练的模型
progress: 如果为 True, 则在标准错误输出上显示下载进度条"""
def resnet18(pretrained:bool=False, progress:bool=True, **kwargs:Dict[str, Any]):
    """加载ResNet18模型网络架构"""
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
def resnet34(pretrained:bool=False, progress:bool=True, **kwargs:Dict[str, Any]):
    """加载ResNet34模型网络架构"""
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)
def resnet50(pretrained:bool=False, progress:bool=True, **kwargs:Dict[str, Any]):
    """加载ResNet50模型网络架构"""
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
def resnet101(pretrained:bool=False, progress:bool=True, **kwargs:Dict[str, Any]):
    """加载ResNet101模型网络架构"""
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
def resnet152(pretrained:bool=False, progress:bool=True, **kwargs:Dict[str, Any]):
    """加载ResNet152模型网络架构"""
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    # 实例化ResNet18模型示例
    resnet_custom_18 = ResNet(BasicBlock, [2, 2, 2, 2])
    print(resnet_custom_18)
    resnet_custom_34 = ResNet(BasicBlock, [2, 2, 2, 2])
    print(resnet_custom_34)
    resnet_custom_50 = ResNet(BasicBlock, [2, 2, 2, 2])
    print(resnet_custom_50)
    resnet_custom_101 = ResNet(BasicBlock, [2, 2, 2, 2])
    print(resnet_custom_101)
    resnet_custom_152 = ResNet(BasicBlock, [2, 2, 2, 2])
    print(resnet_custom_152)
    # 加载预训练模型权重导入
    resnet_18 = resnet18(pretrained=True)
    print(resnet_18)