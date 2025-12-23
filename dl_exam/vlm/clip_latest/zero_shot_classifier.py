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


