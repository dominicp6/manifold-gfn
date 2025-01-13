from typing import List, Union, Optional

import torch
from torchtyping import TensorType

from utils.common import (set_device, set_float_precision)


class Batch:
    """
    Class to handle GFlowNet batches.
    """

    def __init__(
        self,
        env,
        device: Union[str, torch.device],
        float_type,
        trajectories: Optional[TensorType["batch_size, traj_length + 1, n_dim"]] = None,
        policy_trajectories: Optional[TensorType["batch_size, traj_length + 1, policy_state_dim"]] = None,
        actions: Optional[TensorType["batch_size, traj_length, action_dim"]] = None,
        log_rewards: Optional[TensorType["batch_size"]] = None,
    ):
        self.device = set_device(device)
        self.float = set_float_precision(float_type)
        self.env = env
        
        self.size = 0 if trajectories is None else len(trajectories)
        if log_rewards is None:
            self.rewards_available = False
        else:
            self.rewards_available = True
        
        self.trajectories = trajectories
        self.policy_trajectories = policy_trajectories
        self.actions = actions
        self.log_rewards = log_rewards
        self.is_valid()

    def __len__(self):
        return self.size

    def get_n_trajectories(self) -> int:
        return len(self.trajectories)

    def get_actions(self) -> TensorType["batch_size, action_dim"]:
        return self.actions

    def get_rewards(self, force_recompute: Optional[bool] = False) -> TensorType["batch_size"]:
        if self.rewards_available is False or force_recompute is True:
            self._compute_rewards()

    def compute_rewards(self):
        terminating_states = self.get_terminating_states()
        self.log_rewards = torch.zeros(len(self), dtype=self.float, device=self.device)
        self.log_rewards = self.env.proxy(*self.env.statebatch2conformerbatch(terminating_states))
        self.rewards_available = True

    def compute_forward_log_probs(self, gfn):
        self.forward_log_probs = torch.zeros((self.size,), device=gfn.device, dtype=self.float)
        for t in range(self.length):
            policy_dist = gfn.get_forward_policy_dist(self.policy_trajectories[:, t, :])
            log_prob = policy_dist.log_prob(self.actions[:, t, :])
            self.logPF += log_prob
            self.log_fullPF[:, t] = log_prob

    def compute_backward_log_probs(self, gfn):
        self.backward_log_probs = torch.zeros((self.size,), device=gfn.device, dtype=self.float)
        for t in range(self.length, 1, -1):
            policy_dist = gfn.get_backward_policy_dist(self.policy_trajectories[:, t, :])
            log_prob = policy_dist.log_prob(self.actions[:, t - 1, :])
            self.logPB += log_prob
            self.log_fullPB[:, t - 1] = log_prob

    def get_terminating_states(self):
        # Select the last state of each trajectory, ignoring the integer index of the state
        return self.trajectories[:, -1, :-1]

    def get_terminating_rewards(self):
        if self.rewards_available:
            return self.log_rewards
        else:
            self.compute_rewards()
            return self.log_rewards

    def get_actions(self):
        return self.actions

    def merge(self, batches: List):
        if not isinstance(batches, list):
            batches = [batches]

        for batch in batches:
            if len(batch) == 0:
                continue
            
            if self.trajectories is not None:
                self.check_is_compatible(batch)

                assert self.log_rewards is not None, "No rewards available in the current batch"
                assert batch.log_rewards is not None, "No rewards available in the batch to merge"

                self.size += batch.size
                self.trajectories = torch.cat([self.trajectories, batch.trajectories], dim=0)
                self.policy_trajectories = torch.cat([self.policy_trajectories, batch.policy_trajectories], dim=0)
                self.actions = torch.cat([self.actions, batch.actions], dim=0)
                self.log_rewards = torch.cat([self.log_rewards, batch.log_rewards], dim=0)
            else:
                self.size = batch.size
                self.trajectories = batch.trajectories
                self.policy_trajectories = batch.policy_trajectories
                self.actions = batch.actions
                self.log_rewards = batch.log_rewards

        assert self.is_valid()

        return self

    def check_is_compatible(self, batch):
        assert self.env == batch.env, "Environments are not compatible"
        assert self.device == batch.device, "Devices are not compatible"
        assert self.float == batch.float, "Float precision is not compatible"
        assert self.trajectories.shape[1] == batch.trajectories.shape[1], "Trajectory lengths are not compatible"
        assert self.trajectories.shape[2] == batch.trajectories.shape[2], "State dimensions are not compatible"

    def is_valid(self) -> bool:
        if self.trajectories is not None and len(self.trajectories) != self.size:
            raise ValueError("Trajectories and size are incompatible")
        if self.policy_trajectories is not None and len(self.policy_trajectories) != self.size:
            raise ValueError("Policy trajectories and size are incompatible")
        if self.actions is not None and len(self.actions) != self.size:
            raise ValueError("Actions and size are incompatible")
        return True
