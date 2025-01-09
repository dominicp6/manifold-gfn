import pickle

import numpy as np
import torch

from batch import Batch

class GeneralisedRewardBuffer():

    def __init__(self, env, device, capacity: int, priority_capacity: int, priority_ratio: float):
        self.env = env
        self.device = device
        self.cyclic_buffer = RewardBuffer(env, device, capacity)
        self.priority_buffer = RewardBuffer(env, device, priority_capacity)
        self.priority_ratio = priority_ratio
        self.size = 0

    def add(self, batch: Batch):
        self.cyclic_buffer.add(batch.get_terminating_states(), batch.log_rewards)
        self.priority_buffer.priority_add(batch.get_terminating_states(), batch.log_rewards)
        self.size = self.cyclic_buffer.size + self.priority_buffer.size

    def sample(self, batch_size):
        terminal_states = torch.zeros((batch_size, self.env.n_dim), device=self.device)
        log_rewards = torch.zeros((batch_size), device=self.device)

        n_priority = int(self.priority_ratio * batch_size)
        n_cyclic = batch_size - n_priority

        cyclic_terminal_states, cyclic_log_rewards = self.cyclic_buffer.sample(n_cyclic)
        assert not torch.any(cyclic_log_rewards == -torch.inf), "Some cyclic log rewards are -inf"
        priority_terminal_states, priority_log_rewards = self.priority_buffer.sample(n_priority)
        assert not torch.any(priority_log_rewards == -torch.inf), "Some priority log rewards are -inf"

        terminal_states[:n_priority, :] = priority_terminal_states
        terminal_states[n_priority:, :] = cyclic_terminal_states

        log_rewards[:n_priority] = priority_log_rewards
        log_rewards[n_priority:] = cyclic_log_rewards

        return terminal_states, log_rewards
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path, use_old_priority_ratio=False):
        with open(path, "rb") as f:
            buffer = pickle.load(f)
        
        self.cyclic_buffer = buffer.cyclic_buffer
        self.priority_buffer = buffer.priority_buffer
        self.size = buffer.size
        assert self.env == buffer.env, "Loaded buffer does not match the current environment"
        if use_old_priority_ratio:
            self.priority_ratio = buffer.priority_ratio

class RewardBuffer():

    def __init__(self, env, device, capacity: int):
        self.env = env
        self.device = device
        self.capacity = capacity
        self.current_index = 0
        self.size = 0
        self.full = False

        self.terminating_states = torch.zeros((capacity, env.n_dim), device=device)
        self.log_rewards = torch.full((capacity,), -torch.inf, dtype=torch.float32, device=device)

    def add(self, terminating_states, log_rewards):
        batch_size = terminating_states.shape[0]
        assert batch_size > 0, "Batch size must be greater than 0 to add to a replay buffer"
        if batch_size > self.capacity:
            raise ValueError(f"Batch size {batch_size} exceeds buffer capacity {self.capacity}")
        
        if self.current_index + batch_size < self.capacity:
            self._insert(terminating_states, log_rewards, self.current_index, self.current_index + batch_size)
            self.current_index += batch_size
            if not self.full:
                self.size += batch_size
        else:
            remaining_capacity = self.capacity - self.current_index
            self._insert(terminating_states[:remaining_capacity], log_rewards[:remaining_capacity], self.current_index, self.capacity)
            self._insert(terminating_states[remaining_capacity:], log_rewards[remaining_capacity:], 0, batch_size - remaining_capacity)
            self.current_index = batch_size - remaining_capacity
            self.full = True
            self.size = self.capacity

        self.empty = False

    def _get_priority_indices(self, log_rewards):
        high_mask = log_rewards > self.log_rewards.min()
        if not high_mask.any():
            # No high rewards to add
            return
        
        high_rewards = log_rewards[high_mask]

        # Combine current and new rewards, then sort
        total_rewards = torch.cat((self.log_rewards, high_rewards))
        sorted_indices = torch.argsort(total_rewards, descending=False, stable=True)

        # Determine which indices to keep and replace
        num_candidate_additions = len(high_rewards)
        high_indices = sorted_indices[num_candidate_additions:]
        low_indices = sorted_indices[:num_candidate_additions]
        
        # Replace the states, policy_states, actions, and log_rewards
        indices_to_select = high_indices[high_indices >= self.capacity] - self.capacity
        indices_to_replace = low_indices[low_indices < self.capacity]
        assert len(indices_to_select) == len(indices_to_replace), f"Number of indices to select does not match number of indices to replace: {len(indices_to_select)} != {len(indices_to_replace)}"
        num_new_additions = len(indices_to_select)

        return indices_to_replace, indices_to_select, num_new_additions

    def priority_add(self, terminating_states, log_rewards):
        batch_size = terminating_states.shape[0]
        assert batch_size > 0, "Batch size must be greater than 0 to add to a replay buffer"

        indices_to_replace, indices_to_select, num_new_additions = self._get_priority_indices(log_rewards)
        
        if indices_to_replace is not None:
            self.terminating_states[indices_to_replace] = terminating_states[indices_to_select]
            self.log_rewards[indices_to_replace] = log_rewards[indices_to_select]

        if not self.full:
            self.size += num_new_additions
            if self.size >= self.capacity:
                self.full = True
                self.size = self.capacity

    def _insert(self, states, log_rewards, start_idx, end_idx):
        self.terminating_states[start_idx:end_idx] = states
        self.log_rewards[start_idx:end_idx] = log_rewards

    def sample(self, batch_size):
        assert self.size > 0, "Buffer is empty"
        # TODO: consider whether this should be done on the CPU with numpy or on the GPU with torch
        indices = np.random.choice(self.size, batch_size, replace=False)
        terminating_states = self.terminating_states[indices]
        log_rewards = self.log_rewards[indices]

        return terminating_states, log_rewards

    def _get_biased_sample_indices(self, batch_size):
        assert self.size > 0, "Buffer is empty"
  
        # Get a fraction alpha of the trajectories from the beta fraction of trajectories with 
        # highest rewards and a fraction 1 - alpha of trajectories from the remaining.

        # Compute the number of samples to draw
        num_high_reward = int(self.alpha * batch_size)
        num_other = batch_size - num_high_reward

        valid_log_rewards = self.log_rewards[:self.size]

        # Compute thresholds for the top and bottom beta fractions
        top_threshold = self.beta * self.size
        bottom_threshold = self.size - top_threshold

        # Find indices of the top and bottom beta fractions
        top_indices = np.argpartition(valid_log_rewards, -top_threshold)[-top_threshold:]
        bottom_indices = np.argpartition(valid_log_rewards, bottom_threshold)[:bottom_threshold]

        # Sample indices from the top and bottom sets
        selected_top_indices = np.random.choice(top_indices, num_high_reward, replace=False)
        selected_bottom_indices = np.random.choice(bottom_indices, num_other, replace=False)

        indices = np.concatenate((selected_top_indices, selected_bottom_indices))

        return indices
    
    def biased_sample(self, batch_size):
        indices = self._get_biased_sample_indices(batch_size)

        terminating_states = self.terminating_states[indices]
        log_rewards = self.log_rewards[indices]

        return terminating_states, log_rewards
    
    def compute_stats(self):
        mean_log_reward = np.mean(self.log_rewards)
        std_log_reward = np.std(self.log_rewards)

        return mean_log_reward, std_log_reward

class TrajectoryBuffer(RewardBuffer):

    def __init__(self, env, device, capacity: int):
        super().__init__(env, device, capacity)
        self.trajectories = torch.zeros((capacity, env.n_dim), device=device)
        self.policy_trajectories = torch.zeros((capacity, env.policy_input_dim), device=device)
        self.actions = torch.zeros((capacity, env.n_dim), device=device)

    def add(self, trajectories, policy_trajectories, actions, log_rewards):
        batch_size = trajectories.shape[0]
        assert batch_size > 0, "Batch size must be greater than 0 to add to a replay buffer"
        if batch_size > self.capacity:
            raise ValueError(f"Batch size {batch_size} exceeds buffer capacity {self.capacity}")
        
        if self.current_index + batch_size < self.capacity:
            self._insert(trajectories, policy_trajectories, actions, log_rewards, self.current_index, self.current_index + batch_size)
            self.current_index += batch_size
        else:
            remaining_capacity = self.capacity - self.current_index
            self._insert(trajectories[:remaining_capacity], policy_trajectories[:remaining_capacity], actions[:remaining_capacity], log_rewards[:remaining_capacity], self.current_index, self.capacity)
            self._insert(trajectories[remaining_capacity:], policy_trajectories[remaining_capacity:], actions[remaining_capacity:], log_rewards[remaining_capacity:], 0, batch_size - remaining_capacity)
            self.current_index = batch_size - remaining_capacity

        self.empty = False

    def priority_add(self, trajectories, policy_trajectories, actions, log_rewards):
        batch_size = trajectories.shape[0]
        assert batch_size > 0, "Batch size must be greater than 0 to add to a replay buffer"
        
        indices_to_replace, indices_to_select, num_new_additions = self._get_priority_indices(log_rewards)

        self.trajectories[indices_to_replace] = trajectories[indices_to_select]
        self.policy_trajectories[indices_to_replace] = policy_trajectories[indices_to_select]
        self.actions[indices_to_replace] = actions[indices_to_select]
        self.log_rewards[indices_to_replace] = log_rewards[indices_to_select]

        if not self.full:
            self.size += num_new_additions
            if self.size >= self.capacity:
                self.full = True
                self.size = self.capacity

    def _insert(self, trajectories, policy_trajectories, actions, log_rewards, start_idx, end_idx):
        self.trajectories[start_idx:end_idx] = trajectories
        self.policy_trajectories[start_idx:end_idx] = policy_trajectories
        self.actions[start_idx:end_idx] = actions
        self.log_rewards[start_idx:end_idx] = log_rewards

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch = self._get_batch(indices)

        return batch
    
    def sample_terminating_states(self, batch_size):
        assert self.size > 0, "Buffer is empty"
        indices = np.random.choice(self.size, batch_size, replace=False)
        terminating_states, log_rewards = self._get_terminating_states(indices)

        return terminating_states, log_rewards
    
    def biased_sample(self, batch_size):
        # Get a fraction alpha of the trajectories from the beta fraction of trajectories with 
        # highest rewards and a fraction 1 - alpha of trajectories from the remaining.

        indices = self._get_biased_sample_indices(batch_size)
        return self._get_batch(indices)
    
    def biased_sample_terminating_states(self, batch_size):
        indices = self._get_biased_sample_indices(batch_size)
        terminating_states, log_rewards = self._get_terminating_states(indices)

        return terminating_states, log_rewards
    
    def _get_batch(self, indices):
        trajectories = self.trajectories[indices]
        policy_trajectories = self.policy_trajectories[indices]
        actions = self.actions[indices]
        log_rewards = self.log_rewards[indices]

        return Batch(self.env, trajectories=trajectories, policy_trajectories=policy_trajectories, actions=actions, log_rewards=log_rewards)
    
    def _get_terminating_states(self, indices):
        terminating_states = self.trajectories[indices][:, -1, :-1]
        log_rewards = self.log_rewards[indices]

        return terminating_states, log_rewards
