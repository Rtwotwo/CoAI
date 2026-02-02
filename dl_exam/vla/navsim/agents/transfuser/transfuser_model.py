"""
Author: Redal
Date: 2026-01-28
Todo: 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import numpy as np
import torch.nn as nn
from typing import Dict
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index


class TransfuserModel(nn.Module):
    """Build the Transfuser model for autonomous driving."""
    def __init__(self, trajectory_sampling: TrajectorySampling,
                 config: TransfuserConfig):
        """Initializes the Transfuser torch module
        trajectory_sampling: trajectory sampling specification
        config: global config dataclass of TransFuser"""
        super().__init__()
        # Define query segmentation configuration: 
        # basic query and number of bounding boxes
        self._query_splits = [1, config.num_bounding_boxes,]
        self._config = config
        self._backbone = TransfuserBackbone(config)
        # Key-Value Embedding Layer: 8x8 Feature Grid with Trajectory Encoding
        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        # Query embedding layer: total number of queries encoded
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # BEV feature dimension reduction convolution layer and state encoding layer
        # and usually the BEV features are variable in size
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        # BEV Semantic Segmentation Head: For Semantic Prediction in BEV Space
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(config.bev_features_channels, 
                      config.bev_features_channels,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=(1, 1),
                      bias=True),
            nn.ReLu(inplace=True),
            nn.Conv2d(config.bev_features_channels,
                      config.num_bev_classes,
                      kernel_size=(1, 1),
                      stride=1,
                      padding=0,
                      bias=True),
            nn.Upsample(size=(config.lidar_resolution_height // 2, 
                              config.lidar_resolution_width),
                        mode='bilinear',
                        align_corners=False),)
        # Transformer decoder layer configuration
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,)
        # Transformer decoder, agent head, and trajectory head
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead()
        self._trajectory_head = TrajectoryHead()
    def forward(self, features: Dict[str, torch.Tensor]
                )->Dict[str, torch.Tensor]:
        """Transfuser model forward pass module"""
        camera_features:torch.Tensor = features['camera_features']
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features['lidar_features']
        status_feature: torch.Tensor = features['status_features']
        # Extract the last element from the BEV features 
        if isinstance(status_feature, list):
            status_feature = status_feature[-1]
        if isinstance(camera_features, list):
            camera_features = camera_features[-1]
        batch_size = status_feature.shape[0]
        # Process camera and lidar features through backbone network to get BEV features
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_features, lidar_feature)
        


# class AgentHead(nn.Module):
#     """Agent head for Transfuser model: Bounding Box prediction Head"""
#     def __init__(self, num_agents:int, 
#                  d_ffn: int,
#                  d_model: int):
#         """Initialize the agent head
#         num_agents: maximum number of agents to predict
#         d_ffn: feed-forward network dimension
#         d_model: input features dimension"""
#         super(AgentHead, self).__init__()
#         self._num_agents = num_agents
#         self._d_ffn = d_ffn
#         self._d_model = d_model

#         self._mlp_states = nn.Sequential(
#             nn.Linear(self._d_model, self._d_ffn),
#             nn.ReLU(),
#             nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),)
        

        

