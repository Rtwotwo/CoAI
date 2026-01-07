"""
Modified By: Redal
Date: 2025-12-20
Todo: 将类别名称通过文本模板转化为CLIP的文本嵌入,再通过聚合和
      归一化得到零样本分类的权重,并提供了批量和逐样本两种实现方式
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn.functional as F
from functools import partial
from itertools import islice
from typing import Callable, Optional, List, Sequence, Union
import torch
import torch.nn.functional as F


def batched(iterable, n):
      """将一个可迭代对象iterable按指定批次大小n进行分批处理
      返回一个个批次列表,最后一个批次如果不足n个元素,会以实际
      剩余元素数量返回"""
      it = iter(iterable)
      while True:
            batch = list(islice(it, n))
            if not batch: break
            # 生成器函数:在每次迭代时,惰性生成一个批次batch
            yield batch


def build_zero_shot_classifier(model,
                               tokenizer,
                               classnames: Sequence[str],
                               templates: Sequence[Union[Callable, str]],
                               num_classes_per_batch: Optional[int]=10,
                               device: Union[str, torch.device]='cpu',
                               use_tqdm: bool=False):
      """为零样本学习Zero-Shot Learning任务生成类别嵌入class embeddings,最终输出零样本权重
      model: CLIP模型实例,用于文本和图像的编码
      tokenizer: CLIP分词器实例,用于将文本转换为token
      classnames: 类别（标签）名称的序列，如["cat", "dog", "bird"]
      templates: 模板的序列，可以是字符串格式化模板或可调用函数，如["a photo of a {}", "a picture of {}"]
      num_classes_per_batch: 每个批次处理的类别数量,默认为10用于控制内存使用
      device: 计算设备，如'cpu'或'cuda'
      use_tqdm: 是否显示进度条，便于观察处理进度"""
      assert isinstance(templates, Sequence) and len(templates)>0
      assert isinstance(classnames, Sequence) and len(classnames)>0
      use_format = isinstance(templates[0], str)
      num_templates = len(templates)
      num_classes  =len(classnames)
      if use_tqdm:
            import tqdm
            num_iter = 1 if num_classes_per_batch is None else ((num_classes-1) // num_classes_per_batch + 1)
            iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
      else: 
            iter_wrap = iter
      def _process_batch(batch_classnames):
            """处理每个批次的文本数据并获取类别嵌入"""
            num_batch_classes = len(batch_classnames)
            texts = [template.format(c) if use_format else templates(c) for c in batch_classnames for template in templates]
            texts = tokenizer(texts).to(device)
            class_embeddings = model.encode_text(texts, normalize=True)
            class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
            class_embeddings = class_embeddings.T
            return class_embeddings
      with torch.no_grad():
            if num_classes_per_batch:
                  batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
                  zeroshot_weights = torch.cat(batched_embeds, dim=1)
            else:
                  zeroshot_weights = _process_batch(classnames)
      return zeroshot_weights
                       

def build_zero_shot_classifier_legacy(model, 
                                      tokenizer,
                                      classnames: Sequence[str],
                                      templates: Sequence[Union[Callable, str]],
                                      device: Union[str, torch.device]='cpu',
                                      use_tqdm: bool=False):
      assert isinstance(templates, Sequence) and len(templates)>0
      assert isinstance(classnames, Sequence) and len(classnames)>0
      if use_tqdm:
            import tqdm
            iter_wrap = tqdm.tqdm
      else:
            iter_wrap = iter
      use_format = isinstance(templates[0], str)
      # 获取类别嵌入,注意启动零嵌入分类器
      with torch.no_grad():
            zeroshot_weights = []
            for classname in iter_wrap(classnames):
                  texts = [templates.format(classname) if use_format else templates(classname) for template in templates]
                  texts = tokenizer(texts).to(device)
                  class_embedding = model.encode_text(texts)
                  class_embedding = F.normalize(class_embedding, dim=-1).mean(dim=0)
                  class_embedding /= class_embedding.norm()
                  zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
      return zeroshot_weights
