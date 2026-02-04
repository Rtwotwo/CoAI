"""
Author: Redal
Date: 2026-01-28
Todo: Implements the TransFuser Backbone module 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import copy 
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.agents.transfuser.transfuser_config import TransfuserConfig


class TransfuserBackbone(nn.Module):
    """Multi-Scale Fusion Transformer for Image-Lidar Feature Fusion"""
    def __init__(self, config: TransfuserConfig):
        super().__init__()
        self.config = config
        # self.image_encoder = timm.create_model(config.image_architecture, 
        #                                        pretrained=True,
        #                                        features_only=True)
        self.image_encoder = timm.create_model(config.image_architecture, 
                                               pretrained=True, 
                                               features_only=True,
                                               pretrained_cfg_overlay=dict(file=config.bkb_path))
        # Determine the number of LiDAR input channels based on whether the ground plane is used
        if config.use_ground_plane: in_channels = 2 * config.lidar_seq_len
        else: in_channels = config.lidar_seq_len
        
        
