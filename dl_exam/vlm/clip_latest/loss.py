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


# -----------------------------------------------------------------------
# 实现了CLIP/CoCa模型的损失计算逻辑,核心包含对比损失(Contrastive Loss)、图文生成
# 损失(Caption Loss)、蒸馏损失(Distillation Loss)三类损失,同时适配多卡分布式训练场景
# -----------------------------------------------------------------------
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
    """CoCaLoss类是对ClipLoss(CLIP 模型的对比损失)的扩展,融合了
    CLIP对比损失和文本生成的交叉熵损失,是CoCa模型的损失函数核心实现
    clip_loss_weight针对clip模型的对比损失函数的权重
    caption_loss_weight针对文本生成的交叉熵损失权重因子"""
    def __init__(self, caption_loss_weight, 
                 clip_loss_weight,
                 pad_id=0, # pad_token for open_clip custom tokenizer
                 local_loss=False,
                 gather_with_grad=False,
                 cache_labels=False,
                 rank=0,
                 world_size=1,
                 use_horovod=False):
        super().__init__(local_loss=local_loss,
                         gather_with_grad=gather_with_grad,
                         cache_labels=cache_labels,
                         rank=rank,
                         world_size=world_size,
                         use_horovod=use_horovod)
        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
    def forward(self, image_features, 
                text_features, 
                logits, 
                labels, 
                logit_scale, 
                output_dict=False):
        """根据CLIPLoss的forward方法,实现caption_loss和clip_loss"""
        # 利用CLIPLoss计算对比损失函数的输出值
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = clip_loss * self.clip_loss_weight
        else:
            clip_loss = torch.tensor(0, device=logits.device)
        # 计算文本生成损失,实际使用的是nn.CrossEntropyLoss,需要注意的是
        # 交叉熵损失要求输入形状为[batch, vocab_size, seq_len],需将生成logits的[batch, seq_len, vocab_size]转置
        caption_loss = self.caption_loss(logits.permute(0, 2, 1), labels)
        caption_loss = caption_loss * self.caption_loss_weight
        # 若output_dict=True,返回包含两个损失的字典
        if output_dict: 
            return {'contrastive_loss':clip_loss, 'caption_loss':caption_loss}
        return clip_loss, caption_loss

        
class DistillClipLoss(ClipLoss):
    """基于CLIP的对比损失Contrastive Loss扩展的知识蒸馏损失实现,
    核心是让学生模型模仿教师模型的输出分布,同时保留CLIP原本的图文对比损失"""
    def dist_loss(self, teacher_logits, student_logits):
        # 蒸馏损失核心:本质是KL 散度的简化形式(也叫交叉熵损失/负对数似然)
        # 衡量学生模型输出分布与教师模型输出分布的差异
        # 教师logits先softmax归一化为概率分布; 学生模型通过先取log再softmax避免数值不稳定
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
    def forward(self, image_features, 
                text_features, 
                logit_scale, 
                dist_image_features, 
                dist_text_features, 
                dist_logit_scale, 
                output_dict=False):
        """整体损失计算-输入分为两组特征/缩放因子"""
        # 教师端: image_features/text_features/logit_scale
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        # 学生端: dist_image_features/dist_text_features/dist_logit_scale
        dist_logits_per_image, dist_logits_per_text = self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)
        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])
        contrastive_loss = (F.cross_entropy(logits_per_image, labels) + \
                            F.cross_entropy(logits_per_text, labels)) / 2
        distill_loss = (self.dist_loss(dist_logits_per_image, logits_per_image) + \
                        self.dist_loss(dist_logits_per_text, logits_per_text)) / 2
        if output_dict:
            return {'contrastive_loss': contrastive_loss, 'distill_loss': distill_loss}
        return contrastive_loss, distill_loss
        
        
# -----------------------------------------------------------------------
# SigLIP 损失函数(Sigmoid Loss for Language-Image Pre-Training),并针对
# 分布式训练场景多卡/多节点做了优化,主要包含分布式张量交换、损失计算两大核心模块
# -----------------------------------------------------------------------
def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    """基于PyTorch分布式通信的点对点P2P邻居张量交换函数
    实现两个进程rank之间异步发送/接收张量,最终返回从目标进程接收的张量
    group 参数指定通信进程组,默认是全局进程组,仅在该组内的进程可通信"""
    tensor_recv = torch.zeros_like(tensor)
    # 封装非阻塞发送isend操作,目标进程为to_rank
    send_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor,
                to_rank,
                group=group,)
    # 封装非阻塞发送irecv操作,目标进程为from_rank
    recv_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv,
                from_rank,
                group=group,)
    # 批量执行非阻塞收发,batch_isend_irecv批量提交发送和接收操作
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs: req.wait()
    return tensor_recv

    
def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    """实现双向邻居交换功能:同时向左右两个相邻进程发送张量,并接收来自它们的张量,
    通过异步发送/接收操作,将本地的tensor_to_left发送给左邻居,tensor_to_right发
    送给右邻居,同时接收来自左右邻居的对应张量,最后返回接收到的数据"""
    tensor_from_right = torch.zeros_like(tensor_to_right)
    tensor_from_left = torch.zeros_like(tensor_to_left)
    # 封装非阻塞发送isend/irecv操作
    send_op_left = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_to_left,
                left_rank,
                group=group,)
    send_op_right = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_to_right,
                right_rank,
                group=group,)
    recv_op_left = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_from_left,
                left_rank,
                group=group,)
    recv_op_right = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_from_right,
                right_rank,
                group=group,)
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, 
                                                recv_op_right, recv_op_left])
    for req in reqs: req.wait()
    return tensor_from_right, tensor_from_left

    
class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        """用于执行邻居节点间的张量交换操作,方法接收发送方排名、接收方排名、通信组和张量作为参数"""
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group)
    @staticmethod
    def backward(ctx, grad_output):
        """在反向传播时,前三个参数to_rank、from_rank、group不需要梯度,返回None;
        第四个参数grad_output通过NeighbourExchange.apply进行梯度交换操作,实现分布式训练中的梯度传递"""
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output), )


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        """用于执行双向邻居节点间的张量交换操作,方法接收左邻居排名、右邻居排名、通信组和张量作为参数"""
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)
    @staticmethod
    def backward(ctx, *grad_outputs):
        """NeighbourExchangeBidir自定义反向传播函数的backward方法"""
        return (None, None, None) + NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)
    
def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    """用于在分布式训练中进行邻居节点间的数据交换,该函数接收发送方排名、接收方排名、
    张量数据和通信组作为参数,并通过调用NeighbourExchange类的apply方法来执行具体的交换操作"""
    return NeighbourExchange.apply(from_rank, to_rank, tensor, group=None)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    """用于在分布式训练中进行邻居节点间的数据交换,该函数接收发送方排名、接收方排名、
    张量数据和通信组作为参数,并通过调用NeighbourExchangeBidir类的apply方法来执行具体的交换操作"""
    return NeighbourExchangeBidir.apply(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None)


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343
    @article{zhai2023sigmoid,
        title={Sigmoid loss for language image pre-training},
        author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
        journal={arXiv preprint arXiv:2303.15343},
        year={2023}}"""
    def __init__(self, cached_labels: bool=False,
                 rank: int=0,
                 world_size: int=1,
                 dist_impl:Optional[str]=None):
        super().__init__()
        self.cached_labels = cached_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')
        # cache state FIXME cache not currently used
        self.prev_num_logits = 0
        self.labels = {}
    def get_groud_truth(self, device, dtype, num_logits, negative_only=False):
        """形成对角线为+1,其余为-1的标签矩阵,用于对比学习中的正负样本标记"""
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only: 
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels
    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        """计算图像和文本特征的相似度得分:通过矩阵乘法计算图像特征与文本特征的余弦相似度
        然后乘以温度缩放参数logit_scale进行缩放,如果提供了偏置参数logit_bias,则将其加到
        结果上,最后返回计算得到的logits"""
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None: logits += logit_bias
        return logits
    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        """调用get_ground_truth和get_logits方法计算图像和文本特征之间的相似度得分,
        然后计算损失函数,返回计算得到的损失值, 参数保持不变"""
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_groud_truth(image_features.device, 
                                      image_features.dtype, 
                                      image_features.shape[0],
                                      negative_only=negative_only)
        loss = -F.logsigmoid(labels*logits).sum() / image_features.shape[0]
        return loss
    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)
        # 设计不同的分布式策略双向交换/轮转/归约/收集来获取其他进程的文本特征
        # 与本地图像特征进行对比学习,增强模型训练效果
        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                # bidir双向环形更高效,最高效的通信策略之一
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                # 每次双向通信能收到来自两个方向的文本,因此仅需要(N-1)//2轮次通信
                num_bidir, remainder = divmod(self.world_size-1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank, 
                        right_rank,
                        text_features_to_left,
                        text_features_to_right)
                    for f in text_features_recv:
                        loss += self.loss(image_features,
                                          f, 
                                          logit_scale, 
                                          logit_bias, 
                                          negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv
                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, 
                        right_rank,
                        text_features_to_right)
                    loss += self.loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True)        
            elif self.dist_impl == 'shift':
                
            elif self.dist_impl == 'reduce':

            elif self.dist_impl == 'gather':
        
        return loss