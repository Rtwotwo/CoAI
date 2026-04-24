import warnings
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch 
from gymnasium import spaces
from torch.nn import functional as F

from ppo.common.buffers import RolloutBuffer
from ppo.common.on_policy_algorithms import OnPolicyAlgorithm
from ppo.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from ppo.common.type_aliases import GymEnv, MaybeCallback, Schedule
from ppo.common.utils import FloatSchedule, explained_variance


class PPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    )->None:
        super().__init__(
            policy,
            env, 
            learning_rate = learning_rate,
            n_steps = n_steps,
            gamma = gamma,
            gae_lambda = gae_lambda,
            ent_coef = ent_coef,
            vf_coef = vf_coef,
            max_grad_norm = max_grad_norm,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            rollout_buffer_class = rollout_buffer_class,
            rollout_buffer_kwargs = rollout_buffer_kwargs,
            stats_window_size = stats_window_size,
            tensorboard_log = tensorboard_log,
            policy_kwargs = policy_kwargs,
            verbose = verbose,
            device = device,
            seed  = seed,
            _init_step_model = False,
            supported_action_spaces = (
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                )
        )
        
