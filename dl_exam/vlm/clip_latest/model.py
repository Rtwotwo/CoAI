"""
Author: Redal
Date: 2025-12-03
Todo: 实现一个高度可配置,模块化的CLIP(Contrastive Language-Image Pretraining)模型,
      支持多种视觉主干(如Vision Transformer,Modified ResNet,timm模型)和文本编码器
      (包括原生Transformer和HuggingFace模型).复现OpenAI的原始CLIP架构,还扩展对LiT
      (Locked-image Tuning)等先进训练策略的支持——即冻结图像编码器,仅微调文本编码器
      以实现高效的零样本迁移.
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
