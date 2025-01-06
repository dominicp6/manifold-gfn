import numpy as np
import torch

from batch import Batch

# TODO: add ability to save replay buffers

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

class RewardBuffer():

    def __init__(self, env, device, capacity: int):
        self.env = env
        self.device = device
        self.capacity = capacity
        self.current_index = 0
        self.size = 0
        self.full = False

        self.terminal_states = torch.zeros((capacity, env.n_dim), device=device)
        self.log_rewards = torch.full((capacity,), -torch.inf, dtype=torch.float32, device=device)

    def add(self, terminal_states, log_rewards):
        batchsize = terminal_states.shape[0]
        assert batchsize > 0, "Batch size must be greater than 0 to add to a replay buffer"
        if batchsize > self.capacity:
            raise ValueError(f"Batch size {batchsize} exceeds buffer capacity {self.capacity}")
        
        if self.current_index + batchsize < self.capacity:
            self._insert(terminal_states, log_rewards, self.current_index, self.current_index + batchsize)
            self.current_index += batchsize
            if not self.full:
                self.size += batchsize
        else:
            remaining_capacity = self.capacity - self.current_index
            self._insert(terminal_states[:remaining_capacity], log_rewards[:remaining_capacity], self.current_index, self.capacity)
            self._insert(terminal_states[remaining_capacity:], log_rewards[remaining_capacity:], 0, batchsize - remaining_capacity)
            self.current_index = batchsize - remaining_capacity
            self.full = True
            self.size = self.capacity

        self.empty = False

    def priority_add(self, terminal_states, log_rewards):
        batchsize = terminal_states.shape[0]
        assert batchsize > 0, "Batch size must be greater than 0 to add to a replay buffer"
        high_mask = log_rewards > self.log_rewards.min()
        if not high_mask.any():
            # No high rewards to add
            return
        
        high_rewards = log_rewards[high_mask]

        # Combine current and new rewards, then sort
        total_rewards = torch.cat((self.log_rewards, high_rewards))
        sorted_indices = torch.argsort(total_rewards, descending=True)

        # Determine which indices to keep and replace
        high_indices = sorted_indices[:-len(high_rewards)]
        
        # Find the len(high_rewards) indices of lowest reward (and if tied, of lowest index) to replace
        indices_to_replace = torch.argsort(self.log_rewards, descending=False, stable=True)[:len(high_rewards)]

        # Replace the states, policy_states, actions, and log_rewards
        indices_to_select = high_indices[high_indices >= self.capacity] - self.capacity
        self.terminal_states[indices_to_replace] = terminal_states[indices_to_select]
        self.log_rewards[indices_to_replace] = log_rewards[indices_to_select]

        if not self.full:
            self.size += len(high_rewards)
            if self.size >= self.capacity:
                self.full = True
                self.size = self.capacity

    def _insert(self, states, log_rewards, start_idx, end_idx):
        self.terminal_states[start_idx:end_idx] = states
        self.log_rewards[start_idx:end_idx] = log_rewards

    def sample(self, batch_size):
        assert self.size > 0, "Buffer is empty"
        # TODO: consider whether this should be done on the CPU with numpy or on the GPU with torch
        indices = np.random.choice(self.size, batch_size, replace=False)
        terminal_states = self.terminal_states[indices]
        log_rewards = self.log_rewards[indices]

        return terminal_states, log_rewards
    
    def biased_sample(self, batch_size):
        # TODO: needs to be fixed so that it does not select "empty" entries in the buffer
        assert self.size > 0, "Buffer is empty"
        raise NotImplementedError
        # # Get a fraction alpha of the trajectories from the beta fraction of trajectories with 
        # # highest rewards and a fraction 1 - alpha of trajectories from the remaining.

        # num_top = int(self.alpha * batch_size)
        # num_bottom = batch_size - num_top

        # top_threshold = self.beta * len(self)
        # bottom_threshold = len(self) - top_threshold

        # top_indices = np.argpartition(self.log_rewards, -top_threshold)[-top_threshold:]
        # bottom_indices = np.argpartition(self.log_rewards, bottom_threshold)[:bottom_threshold]

        # selected_top_indices = np.random.choice(top_indices, num_top, replace=False)
        # selected_bottom_indices = np.random.choice(bottom_indices, num_bottom, replace=False)

        # indices = np.concatenate((selected_top_indices, selected_bottom_indices))

        # terminal_states = self.terminal_states[indices]
        # log_rewards = self.log_rewards[indices]

        # return terminal_states, log_rewards
    
    def compute_stats(self):
        mean_log_reward = np.mean(self.log_rewards)
        std_log_reward = np.std(self.log_rewards)

        return mean_log_reward, std_log_reward

# TODO: finish implementation of the TrajectoryBuffer
# class TrajectoryBuffer(Buffer):

#     def add(self, states, policy_states, actions, log_rewards):
#         batchsize = states.shape[0]
#         assert batchsize > 0, "Batch size must be greater than 0 to add to a replay buffer"
#         if batchsize > self.capacity:
#             raise ValueError(f"Batch size {batchsize} exceeds buffer capacity {self.capacity}")
        
#         if self.current_index + batchsize < self.capacity:
#             self._insert(states, policy_states, actions, log_rewards, self.current_index, self.current_index + batchsize)
#             self.current_index += batchsize
#         else:
#             remaining_capacity = self.capacity - self.current_index
#             self._insert(states[:remaining_capacity], policy_states[:remaining_capacity], actions[:remaining_capacity], log_rewards[:remaining_capacity], self.current_index, self.capacity)
#             self._insert(states[remaining_capacity:], policy_states[remaining_capacity:], actions[remaining_capacity:], log_rewards[remaining_capacity:], 0, batchsize - remaining_capacity)
#             self.current_index = batchsize - remaining_capacity

#         self.empty = False

#     def priority_add(self, states, policy_states, actions, log_rewards):
#         batchsize = states.shape[0]
#         assert batchsize > 0, "Batch size must be greater than 0 to add to a replay buffer"
#         high_mask = log_rewards > self.log_rewards.min()
#         if not high_mask.any():
#             # No high rewards to add
#             return
        
#         high_rewards = log_rewards[high_mask]

#         # Combine current and new rewards, then sort
#         total_rewards = torch.cat((self.log_rewards, high_rewards))
#         sorted_indices = torch.argsort(total_rewards, descending=True)

#         # Determine which indices to keep and replace
#         high_indices = sorted_indices[:len(high_rewards)]
#         indices_to_keep = np.array(high_indices[high_indices < self.capacity])
#         indices_to_replace = np.setdiff1d(np.arange(self.capacity), indices_to_keep)

#         # Replace the states, policy_states, actions, and log_rewards
#         indices_to_select = high_indices[high_indices > self.capacity] - self.capacity
#         self.states[indices_to_replace] = states[indices_to_select]
#         self.policy_states[indices_to_replace] = policy_states[indices_to_select]
#         self.actions[indices_to_replace] = actions[indices_to_select]
#         self.log_rewards[indices_to_replace] = log_rewards[indices_to_select]

#         self.empty = False

#     def _insert(self, states, policy_states, actions, log_rewards, start_idx, end_idx):
#         self.states[start_idx:end_idx] = states
#         self.policy_states[start_idx:end_idx] = policy_states
#         self.actions[start_idx:end_idx] = actions
#         self.log_rewards[start_idx:end_idx] = log_rewards

#     def sample_terminal_states(self, batch_size):
#         assert self.size > 0, "Buffer is empty"
#         indices = np.random.choice(self.size, batch_size, replace=False)
#         terminal_states, log_rewards = self._get_back_sampled_batch(indices)

#         return terminal_states, log_rewards

#     def sample(self, batch_size):
#         indices = np.random.choice(len(self), batch_size, replace=False)
#         batch = self._get_batch(indices)

#         return batch
    
#     def biased_sample(self, batch_size):
#         # Get a fraction alpha of the trajectories from the beta fraction of trajectories with 
#         # highest rewards and a fraction 1 - alpha of trajectories from the remaining.

#         num_top = int(self.alpha * batch_size)
#         num_bottom = batch_size - num_top

#         top_threshold = self.beta * len(self)
#         bottom_threshold = len(self) - top_threshold

#         top_indices = np.argpartition(self.log_rewards, -top_threshold)[-top_threshold:]
#         bottom_indices = np.argpartition(self.log_rewards, bottom_threshold)[:bottom_threshold]

#         selected_top_indices = np.random.choice(top_indices, num_top, replace=False)
#         selected_bottom_indices = np.random.choice(bottom_indices, num_bottom, replace=False)

#         indices = np.concatenate((selected_top_indices, selected_bottom_indices))

#         return self._get_batch(indices)
    
#     def _get_batch(self, indices):
#         states = self.states[indices]
#         actions = self.actions[indices]
#         log_rewards = self.log_rewards[indices]

#         return Batch(self.env, states=states, actions=actions, log_rewards=log_rewards)
    
#     def _get_back_sampled_batch(self, indices):
#         terminal_states = self.states[indices][:, -1, :-1]
#         log_rewards = self.log_rewards[indices]

#         return terminal_states, log_rewards
    
#     def compute_stats(self):
#         mean_log_reward = np.mean(self.log_rewards)
#         std_log_reward = np.std(self.log_rewards)

#         return mean_log_reward, std_log_reward
