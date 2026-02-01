from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np
import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class AbstractAgent(torch.nn.Module, ABC):
    def __init__(self, trajectory_sampling: TrajectorySampling,
                 requeires_scene: bool = False):
        super().__init__()
        self.requires_scene = requeires_scene
        self._trajectory_sampling = trajectory_sampling
    @abstractmethod
    def name(self,)->str:
        """Returns the name of str agent"""
    @abstractmethod
    def get_sensor_config(self
                )->SensorConfig:
        """return sensor config for lidar and cameras"""
    @abstractmethod
    def initialize(self,)->None:
        """Initialize agent for parameters"""
    def forward(self, features:Dict[str, torch.Tensor]
                )->Dict[str, torch.Tensor]:
        """forward pass of the agent"""
        raise NotImplementedError('[ERROR] No foward features function')
    def get_feature_builders(self,
        ) -> List[AbstractFeatureBuilder]:
        """return list of feature builders"""
        raise NotImplementedError("[ERROR] No feature builders. Agent does not support training")
    def get_target_builders(self,
        )->List[AbstractTargetBuilder]:
        """return list of target builders"""
        raise NotImplementedError("[ERROR] No target builders. Agent does not support training")
    def compute_trajectory(self, agent_input:AgentInput, 
                           dp_proposals=None, 
                           device='cpu'
                           )->Trajectory:
        """Compute the ego trajectory based on the agent input
        agengt_input: AgentInput from cameras, lidars and ego status
        dp_proposals: dynamic programming proposals for possible trajectories evaluation
        device: device to run the computation"""
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features agent_input
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch_size dimension
        # features = {k:v.unsqueeze(0) for k, v in features.items()}
        for k, v in features.items():
            if isinstance(v, list):
                v = [i.unsqueeze(0).to(device) for i in v]
            elif isinstance(v, np.ndarray):
                v = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
            elif isinstance(v, torch.Tensor):
                v = v.unsqueeze(0).to(device)
            features[k] = v
        # forward pass to get predictions
        with torch.no_grad():
            if dp_proposals is not None:
                # use dynamic programming proposals to evaluate best trajectory
                predictions = self.evaluate_dp_proposals(features, dp_proposals)
            else:
                predictions = self.forward(features)
            poses = predictions['trajectory'].squeeze(0).cpu().numpy()
            predictions['trajectory'] = Trajectory(poses, self._trajectory_sampling)
        # extract trajectory and return Trajectory(poses, self._trajectory_sampling)
        return predictions
    def compute_loss(self, features: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     predictions: Dict[str, torch.Tensor]
                     )->torch.Tensor:
        """compute the loss based on the features, targets and predictions"""
        raise NotImplementedError('[ERROR] No compute loss function')
    def get_optimizer(self,)->Union[torch.optim.Optimizer, 
        Dict[str, Union[torch.optim.lr_scheduler.LRScheduler]]]:
        """Returns the optimizers that are used by thy pytorch-lightning trainer
        Has to be either a single optimizer or a dict of optimizer and lr scheduler"""
        raise NotImplementedError('[ERROR] No optimizer function')
    def get_training_callbacks(self,)->List[pl.Callback]:
        """Returns a list of pytorch-lightning callbacks that are used during training
        See navsim.planning.training.callbacks for examples"""
        return []