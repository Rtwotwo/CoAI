"""
Author: Redal
Date: 2026-01-28
Todo: 
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import os
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Dict, Any, List, Union, Optional
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


