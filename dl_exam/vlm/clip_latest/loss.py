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
    def get_ground_truth(self, device, 
                         num_logits
                         )->torch.Tensor:
        """生成/缓存并返回指定设备、指定长度的标签张量,适配分布式训练场景"""
        # 触发重新生成标签的条件:本次需要的num_logits标签长度和上一次缓存的prev_num_logits不一致
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            # 分布式训练的总进程数大于1且使用本地损失时,为不同进程生成不重叠的标签,避免标签冲突
            if self.world_size > 1 and self.local_loss:
                labels = labels + self.rank * num_logits
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    def get_logits(self, image_features, 
                   text_features, 
                   logit_scale, 
                   logit_bias=None):
        """计算图像-文本匹配对数概率:输入图像特征image_features,文本特征text_features,
        结合温度系数logit_scale和可选偏置logit_bias,计算两组匹配得分"""
        if self.world_size > 1:
            # 在分布式场景下搜集所有进程和GPU的相关特征图
            all_image_features, all_text_features = gather_features(
                        image_features,
                        text_features,
                        local_loss=self.local_loss,
                        gather_with_grad=self.gather_with_grad,
                        rank=self.rank,
                        world_size=self.world_size,
                        use_horovod=self.use_horovod,)
            # 仅用当前卡特征和全局特征计算,而非全局特征互乘
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        # 添加可选偏置logit_bias参数进行求解
        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text -= logit_bias
        return logits_per_image, logits_per_text
    def forward(self, image_features,
                text_features,
                logit_scale,
                logit_bias=None,
                output_dict=False):
        """对比学习Contrastive Learning中图文匹配任务的前向传播逻辑,核心目标是
        计算图文之间的对比损失Contrastive Loss,让匹配的图文特征更相似,不匹配的更疏远"""
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, 
                                                            text_features,
                                                            logit_scale, 
                                                            logit_bias=logit_bias)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return {'contrastive_loss': total_loss} if output_dict else total_loss
    

class CoCaLoss(ClipLoss):
    def __init__(self, ):
        """"""
        
        
    