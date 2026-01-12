"""
Modified By: Redal
Date: 2025-12-03
Todo: 通过拉近匹配的图像-文本对的特征表示,推开不匹配的对,
      从而学习到对齐的跨模态语义空间,用于对比学习(Contrastive Learning)
      的损失函数,主要用于图文多模态预训练模型(如CLIP/CoCa/SigLIP等)
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = False
except ImportError:
    has_distributed = False
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(image_features,
                    text_features,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1,
                    use_horovod=False):
    """在分布式训练场景下收集所有GPU/进程上的图像特征
    image_features和文本特征text_features,参数设置:
    local_loss: 是否仅计算本地损失，决定是否保留本地特征的梯度
    gather_with_grad: 收集特征时是否保留梯度
    rank/world_size: 当前进程编号 / 总进程数"""
    assert has_distributed, '[INFO] 注意安装torch.distributed.nn!'
    if use_horovod:
        assert hvd is not None, '[INFO] 注意安装horovod!'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # 当所有其他GPU的的所有all_*特征没有梯度而在本地local_gpu存在梯度
                # 将完整的图像/文本特征按照指定的维度进行切分，并转为列表
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # 将所有gpus的特征集中到一块
        if gather_with_grad:
            all_image_features = torch.distributed.nn.all_gather(image_features, dim=0)
            all_text_features = torch.distributed.nn.all_gather(text_features, dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(self, local_loss=False,
                 gather_with_grad=False,
                 cache_labels=False,
                 rank=0,
                 world_size=1,
                 use_horovod=False):
        super().__init__()
        # 初始化CLIP loss相关技术参数
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        # 创建缓存相关参数
        self.prev_num_logits = 0
        self.labels = {}
    def get_ground_truth(self,):
        
    def get_logits(self, ):
        """"""
    def forward(self, ):
        """"""
        
    