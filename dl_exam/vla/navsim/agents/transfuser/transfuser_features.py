"""
Author: Redal
Date: 2026-01-28
Todo: 
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
