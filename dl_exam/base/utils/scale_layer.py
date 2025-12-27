"""
Author: Redal
Date: 2025-12-13
Todo: 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as  nn


SCALE_CONFIGS = {'MinMaxScaler': '数据归一化到[0,1]区间、分类任务、特征量纲统一',
                 'StandardScaler': '数据标准化(均值0方差1)、回归任务、深度学习预处理',
                 'MaxAbsScaler': '稀疏数据缩放、不改变稀疏性、文本特征处理',
                 'RobustScaler': '抗异常值干扰、离群点较多的场景、金融数据预处理',
                 'Normalizer': '样本级归一化、文本分类、聚类任务、基于L1/L2范数',
                 'QuantileTransformer': '非线性缩放、将数据映射到均匀分布、处理偏态数据',
                 'PowerTransformer': '正态分布转换、Box-Cox变换、Yeo-Johnson变换、回归任务',
                 'StandardScalerWithMean': '保留均值信息的标准化、多模态数据预处理、特征对齐',
                 'MinMaxScalerWithClip': '限制极值的归一化、防止异常值溢出、图像像素预处理',
                 'ScalerWithRobustStd': '基于中位数和四分位距的标准化、工业数据预处理、抗噪声',
                 'FeatureScaler': '多特征混合缩放、机器学习流水线、自动匹配特征类型',
                 'BatchScaler': '批量数据在线缩放、流式数据处理、实时预测系统',
                 'ChannelScaler': '图像通道级缩放、计算机视觉、CNN输入预处理',
                 'SparseScaler': '高维稀疏特征、推荐系统、点击率预测任务',
                 'LogScaler': '对数变换缩放、处理指数分布数据、金融收益数据预处理',
                 'ZScoreScaler': 'Z分数标准化、统计分析、异常值检测、医疗数据处理',
                 'UnitVectorScaler': '单位向量缩放、相似度计算、聚类任务、文本嵌入处理',
                 'RangeScaler': '自定义区间缩放、特定业务需求、特征约束到[a,b]区间',
                 'PercentileScaler': '基于百分位数的缩放、抗极端值、气象数据预处理',
                 'NormalizeScaler': 'L2范数归一化、向量空间模型、自然语言处理',
                 'LayerScale': 'Transformer架构、大语言模型、视觉Transformer(ViT)、缓解深度网络训练不稳定性'}


