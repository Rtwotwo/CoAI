"""
Author: Redal
Date: 2026-01-28
Todo: 实现TransFuser自动驾驶模型的输入特征构建器TransfuserFeatureBuilder:
      a.拼接并缩放前视三摄像头图像(左、前、右)为一张宽幅RGB图;b.将LiDAR点云
      按高度分层,投影到鸟瞰图BEV生成2D直方图作为LiDAR特征;c.拼接自车当前状
      态(驾驶指令、速度、加速度)作为状态特征.
      输出目标构建器TransfuserTargetBuilder:a.提取未来轨迹用于自车规划;b.在
      BEV空间中提取邻近交通参与者车辆的2D有向包围框及其存在标签;c.渲染包含车
      道线、道路边界、交通参与者等语义信息的 多类别BEV语义地图,用于辅助感知与规划.
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
from enum import IntEnum
from typing import Any, Dict, List, Tuple
import cv2
import torch
import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, MapObject, SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import affinity
from shapely.geometry import LineString, Polygon
from torchvision import transforms
# Import local files from customized package
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TransfuserFeatureBuilder(AbstractFeatureBuilder):
    """Input features builder for Transfuser agent"""
    def __init__(self, config: TransfuserConfig):
        """Initializes feature builder
        :param config: global config dataclass of TransFuser"""
        self._config = config
    def get_unique_name(self,)->str:
        """Inherited, see superclass"""
        return "transfuser_feature"
    def compute_features(self, agent_input: AgentInput
                         )->Dict[str, torch.Tensor]:
        """Build the input features for camera/lidar/status features
        for TransFuser agent to processing data and extracting targets"""
        features = {}
        features["camera_feature"] = self._get_camera_feature(agent_input)
        if not self._config.use_lidar: 
            features["lidar_feature"] = self._get_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate([
            torch.tensor(agent_input.ego_status[-1].driving_command, dtype=torch.float32),
            torch.tensor(agent_input.ego_status[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(agent_input.ego_status[-1].ego_acceleration, dtype=torch.float32)])
        return features
    def _get_camera_feature(self, agent_input: AgentInput)->torch.Tensor:
        """Extract stitched camera from AgentInput
        agent_input: input dataclass, stitched front view image as torch tensor"""
        cameras = agent_input.cameras[-1]
        # crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]
        # stitch images l0, r0, f0 together
        stitched_image = np.concatenate([l0, f0, r0],axis=1)
        # resize the stitched image and convert to torch tensor
        resized_image = cv2.resize(stitched_image, (2048, 512))
        tensor_image = transforms.ToTensor()(resized_image)
        return tensor_image
    def _get_lidar_feature(self, agent_input: AgentInput)->torch.Tensor:
        """Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors"""
        # for lidar, only consider (x, y, z) & swap axes for (N, 3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T
        # 
