import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import numpy as np
import torch 
from gymnasium import spaces

from ppo.common.utils import get_device
from ppo.vec_env import VecNormalize
from ppo.common.preprocessing import get_action_dim, get_obs_shape
from ppo.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

try:
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """
    observation_space: spaces.Space
    obs_shape: tuple[int, ...]
    def __init__(self, 
        self,
        buffer_size: int,
        observation_spaces: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        n_envs: int = 1,
    )->None:
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_spaces
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_spaces)  

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        """
        shape = arr.shape
        if len(shape) < 3:
            return arr
        return arr.swapaxes(0, 1).reshape(-1, *shape[2:]) 

    def size(self)->int:
        """
        :return: the Number of element in the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos


    
