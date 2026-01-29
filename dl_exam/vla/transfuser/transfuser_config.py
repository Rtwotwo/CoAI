"""
Author: Redal
Date: 2026-01-28
Todo: 为自动驾驶场景设计的配置类,整合了多模态感知(摄像头+激光雷达),
      BEV特征提取,目标检测等功能所需的所有超参数,为基于Transformer
      的传感器融合感知系统提供统一的参数管理.
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class TransfuserConfig:
    """"""
    image_architecture: str='resnet34'
    lidar_architecture: str='resnet34'
    # the bkb_path and ckpt_path should be set by user
    bkb_path: str='your_dataset_path/models/resnet34_model.bin'
    ckpt_path: str=None

    latent: bool=True
    latent_rad_thresh: float=4 * np.pi / 9

    max_height_lidar: float=100.0
    pixels_per_meter: float=4.0
    hist_max_per_pixel: int=5

    lidar_min_x: float=-32
    lidar_max_x: float=32
    lidar_min_y: float=-32
    lidar_max_y: float=32
    lidar_split_height: float=0.2
    use_ground_plane: bool=True

    # add new parameters for camera and lidar settings
    lidar_seq_len: int=1
    camera_width: int=1024
    canera_height: int=256
    lidar_resolution_width: int=256
    lidar_resolution_height: int=256

    img_vert_anchors: int=256//32
    img_horz_anchors: int=256//32
    lidar_vert_anchors: int=256//32
    lidar_horz_anchors: int=256//32

    block_exp: int=4
    n_layer: int=2 # number of transformer layers used in version backbone
    n_head: int=4
    n_scale: int=4
    embed_pdrop: float=0.1
    resid_pdrop: float=0.1
    attn_pdrop: float=0.1
    # mean of the normal distribution initialization for linear layers in the gpt
    gpt_linear_layer_init_mean: float=0.0
    # std of the normal distribution initialization for linear layers in the gpt
    gpt_linear_layer_init_std: float=0.02
    # initial weight of the layer norms in the gpt
    gpt_layer_norm_init_weight: float=1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer related parameters
    tf_d_model: int=256
    tf_d_ffn: int=1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0
    # detection 
    num_bounding_boxes: int=30
    # loss weight
    trajectory_weight: float=10.0
    agent_class_weight: float=10.0
    agent_box_weight: float=1.0
    bev_semantic_weight: float=10.0

    # BEV mapping 
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSCTION]), # roads
        2: ("polygon", [SemanticMapLayer.WALKWAY]), # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),
        4: ("box", [TrackedObjectType.CZONE_SIGN, # static objects
                    TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT,]),
        5: ("box", [TrackedObjectType.VEHICLE]), # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),} # pedestrians
    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25
    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
    @property
    def bev_semantic_frame(self,)->Tuple[int, int]:
        """Get BEV semantic frame size and returns 
        the tuple containing the size of the frame"""
        return (self.bev_pixel_height, self.bev_pixel_width)
    @property
    def bev_radius(self,)->float:
        """Compute the radius of the BEV map, The radius value of 
        the bird's-eye view, which is the maximum absolute value 
        of the LiDAR boundary coordinates"""
        values = [ self.lidar_max_x, self.lidar_max_y, 
                  self.lidar_min_x, self.lidar_min_y]
        return max([abs(value) for value in values])