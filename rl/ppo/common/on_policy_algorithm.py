import sys
import time
import warnings
from typing import Any, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces

from ppo.common.base_class import BaseAlgorithm
from ppo.common.buffers import RolloutBuffer, DictRolloutBuffer
from ppo.common.callbacks import BaseCallback
from ppo.common.policies import ActorCriticPolicy
from ppo.common.type_aliases import GymEnv, MaybeCallback, Schedule
from ppo.common.utils import obs_as_tensor, safe_mean
from ppo.common.vec_env import VecEnv

