from typing import List, Union, Optional

import torch
from torchtyping import TensorType

from utils.common import (set_device, set_float_precision, tfloat)


class Batch:
    """
    Class to handle GFlowNet batches.
    """

    def __init__(
        self,
        env,
        device: Union[str, torch.device],
        # TODO: update these dimension labels
        states: Optional[TensorType["n_states, state_dim"]] = None,
        policy_states: Optional[TensorType["n_states, policy_state_dim"]] = None,
        actions: Optional[TensorType["n_states, action_dim"]] = None,
        log_rewards: Optional[TensorType["n_states"]] = None,
        float_type: Union[int, torch.dtype] = 32,
    ):
        self.device = set_device(device)
        self.float = set_float_precision(float_type)
        self.env = env
        
        self.size = 0 if states is None else len(states)
        if log_rewards is None:
            self.rewards_available = False
        else:
            self.rewards_available = True
        
        self.states = states
        self.policy_states = policy_states
        self.actions = actions
        self.log_rewards = log_rewards
        self.is_valid()

    def __len__(self):
        return self.size

    def get_n_trajectories(self) -> int:
        return len(self.trajectories)

    def get_actions(self) -> TensorType["n_states, action_dim"]:
        # TODO: work out what these tfloat functions mean
        return tfloat(self.actions, float_type=self.float, device=self.device)

    def get_rewards(self, force_recompute: Optional[bool] = False) -> TensorType["n_states"]:
        if self.rewards_available is False or force_recompute is True:
            self._compute_rewards()

    def compute_rewards(self):
        # TODO: come back to this and see what the Done flag was about in the original code
        terminating_states = self.get_terminating_states()
        self.log_rewards = torch.zeros(len(self), dtype=self.float, device=self.device)
        self.log_rewards = self.env.proxy(*self.env.statebatch2conformerbatch(terminating_states))
        self.rewards_available = True

    def compute_forward_log_probs(self, gfn):
        self.forward_log_probs = torch.zeros((self.size,), device=gfn.device, dtype=self.float)
        for t in range(self.length):
            policy_dist = gfn.get_forward_policy_dist(self.policy_states[:, t, :])
            log_prob = policy_dist.log_prob(self.actions[:, t, :])
            self.logPF += log_prob
            self.log_fullPF[:, t] = log_prob

    def compute_backward_log_probs(self, gfn):
        self.backward_log_probs = torch.zeros((self.size,), device=gfn.device, dtype=self.float)
        for t in range(self.length, 1, -1):
            policy_dist = gfn.get_backward_policy_dist(self.policy_states[:, t, :])
            log_prob = policy_dist.log_prob(self.actions[:, t - 1, :])
            self.logPB += log_prob
            self.log_fullPB[:, t - 1] = log_prob

    def get_terminating_states(self):
        # Select the last state of each trajectory, ignoring the integer index of the state
        return self.states[:, -1, :-1]

    def get_terminating_rewards(self):
        if self.rewards_available:
            return self.log_rewards
        else:
            self.compute_rewards()
            return self.log_rewards

    def get_action_trajectories(self):
        return self.actions

    def merge(self, batches: List):
        if not isinstance(batches, list):
            batches = [batches]

        for batch in batches:
            if len(batch) == 0:
                continue
            
            if self.states is not None:
                self.check_is_compatible(batch)

                assert self.log_rewards is not None, "No rewards available in the current batch"
                assert batch.log_rewards is not None, "No rewards available in the batch to merge"

                self.size += batch.size
                self.states = torch.cat([self.states, batch.states], dim=0)
                self.policy_states = torch.cat([self.policy_states, batch.policy_states], dim=0)
                self.actions = torch.cat([self.actions, batch.actions], dim=0)
                self.log_rewards = torch.cat([self.log_rewards, batch.log_rewards], dim=0)
            else:
                self.size = batch.size
                self.states = batch.states
                self.policy_states = batch.policy_states
                self.actions = batch.actions
                self.log_rewards = batch.log_rewards

        assert self.is_valid()

        return self

    def check_is_compatible(self, batch):
        assert self.env == batch.env, "Environments are not compatible"
        assert self.device == batch.device, "Devices are not compatible"
        assert self.float == batch.float, "Float precision is not compatible"
        assert self.states.shape[1] == batch.states.shape[1], "Trajectory lengths are not compatible"
        assert self.states.shape[2] == batch.states.shape[2], "State dimensions are not compatible"

    def is_valid(self) -> bool:
        if self.states is not None and len(self.states) != self.size:
            raise ValueError("States and size are incompatible")
        if self.policy_states is not None and len(self.policy_states) != self.size:
            raise ValueError("Policy states and size are incompatible")
        if self.actions is not None and len(self.actions) != self.size:
            raise ValueError("Actions and size are incompatible")
        return True
