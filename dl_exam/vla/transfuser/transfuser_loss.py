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
