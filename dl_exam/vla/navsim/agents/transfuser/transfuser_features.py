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
        """"""
        return "transfuser_feature"
