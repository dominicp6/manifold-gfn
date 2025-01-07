"""
GFlowNet
"""
from typing import Optional
from copy import copy
from itertools import chain

import torch
import torch.nn as nn
from tqdm import tqdm

from batch import Batch
from buffer import GeneralisedRewardBuffer
from utils.common import set_device

class GFlowNet:
    def __init__(
        self,
        env,
        forward_policy,
        backward_policy,
        optimizer_config,
        device,
        batch_size,
        regular_capacity = 1000,
        priority_capacity = 500,
        priority_ratio = 0.5,
        replay_sampling = True
    ):
        self.device = set_device(device)
        self.env = env
        self.traj_length = self.env.traj_length
        self.batch_size = batch_size

        # Define logZ as the SUM of the values in this tensor. Dimension > 1 only to accelerate learning.
        self.logZ = nn.Parameter(optimizer_config.initial_z_scaling * torch.ones(optimizer_config.z_dim) / optimizer_config.z_dim) 

        # Buffers
        self.replay_sampling = replay_sampling
        self.buffer = GeneralisedRewardBuffer(self.env, self.device, regular_capacity, priority_capacity, priority_ratio)

        # Policy models
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
       
        # Optimizer
        self.opt_config = optimizer_config
        self._init_optimizer()

    def parameters(self):
        return list(self.forward_policy.model.parameters()) + list(self.backward_policy.model.parameters())

    def _init_optimizer(self):
        """
        Set up the optimizer
        """
        params = self.parameters()
        if not len(params):
            return ValueError("No parameters found.")
        if self.opt_config.method == "adam":
            opt = torch.optim.Adam(params, self.opt_config.lr, betas=(self.opt_config.adam_beta1, self.opt_config.adam_beta2))
        elif self.opt_config.method == "msgd":
            opt = torch.optim.SGD(params, self.opt_config.lr, momentum=self.opt_config.momentum)
        opt.add_param_group({"params": self.logZ, "lr": self.opt_config.lr * self.opt_config.lr_z_mult})
        self.opt = opt

        # Learning rate scheduling
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.opt_config.lr_decay_period, gamma=self.opt_config.lr_decay_gamma)
        self.lr_scheduler = lr_scheduler
        
    def parameters(self):
        return chain(self.forward_policy.model.parameters(), self.backward_policy.model.parameters())
    
    def optimization_step(self, loss):
        loss.backward()
        if self.opt_config.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.opt_config.clip_value)
        self.opt.step()
        self.opt.zero_grad()
        self.lr_scheduler.step()
    
    def sample_action(self, state, backward: Optional[bool] = False):
        if backward:
            model = self.backward_policy
        else:
            model = self.forward_policy

        policy_state = self.env.encode_state(state)
        policy_output = model(policy_state)

        # Sample actions from policy outputs
        action = self.env.sample_actions_batch(policy_output, state, backward)

        return action, policy_state
    
    def backward_sample_trajectories(self, terminal_states):
        batch_size = terminal_states.shape[0]
        actions = torch.zeros((batch_size, self.traj_length, self.env.n_dim), device=self.device) 
        states = torch.zeros((batch_size, self.traj_length + 1, self.env.n_dim + 1), device=self.device) 
        policy_states = torch.zeros((batch_size, self.traj_length + 1, self.env.policy_input_dim), device=self.device)
        states[:, -1, :] = torch.cat([terminal_states, self.traj_length * torch.ones(batch_size, 1, device=self.device)], dim=1) 
        
        for t in range(self.traj_length - 1, -1, -1):
            # t ranges from (self.trajectory_length - 1) to (0)
            #  TODO: check the index when sampling actions
            actions[:, t, :], policy_states[:, t, :] = self.sample_action(states[:, t + 1, :], backward=True)
            self.env.step_backwards(actions[:, t, :], states[:, t, :])
            states[:, t, :] = copy(self.env.state)

        return states, policy_states, actions

    @torch.no_grad()
    def sample_batch(self, n_onpolicy: int = 0, n_replay: int = 0):
        """
        Builds a batch of data by sampling online and/or offline trajectories.
        """

        # Initialise an on-policy batch 
        actions = torch.zeros((n_onpolicy, self.traj_length, self.env.n_dim), device=self.device) 
        states = torch.zeros((n_onpolicy, self.traj_length + 1, self.env.n_dim + 1), device=self.device) 
        policy_states = torch.zeros((n_onpolicy, self.traj_length + 1, self.env.policy_input_dim), device=self.device)
        log_rewards = torch.zeros(n_onpolicy, device=self.device)

        states[:, 0, :] = self.env.source
        # Generate on-policy trajectories
        for t in range(self.traj_length):
            actions[:, t, :], policy_states[:, t, :] = self.sample_action(states[:, t, :])
            states[:, t + 1, :] = self.env.step(actions[:, t, :], states[:, t, :])

        on_policy_batch = Batch(env=self.env, device=self.device, states=states, policy_states=policy_states, actions=actions)
        on_policy_batch.compute_rewards()

        # Sample a replay buffer batch if the buffer is not empty
        if self.buffer.size > 0:
            terminal_states, log_rewards = self.buffer.sample(n_replay)
            states, policy_states, actions = self.backward_sample_trajectories(terminal_states)
            replay_buffer_batch = Batch(env=self.env, device=self.device, states=states, policy_states=policy_states, actions=actions, log_rewards=log_rewards)

            # Merge the batches
            final_batch = on_policy_batch.merge(replay_buffer_batch)

            return final_batch
        else:
            return on_policy_batch

    def compute_logprobs_trajectories(self, batch: Batch, backward: bool = False):
        if backward:
            # Backward trajectories
            policy_output_b = self.backward_policy(batch.policy_states)
            logprobs = self.env.get_traj_batch_logprobs(policy_output_b, batch.actions, backwards=True)
        else:
            # Forward trajectories
            policy_output_f = self.forward_policy(batch.policy_states)
            logprobs = self.env.get_traj_batch_logprobs(policy_output_f, batch.actions)
        
        return logprobs

    def trajectorybalance_loss(self, batch):
        # Get logprobs of forward and backward transitions
        logprobs_f = self.compute_logprobs_trajectories(batch, backward=False)
        logprobs_b = self.compute_logprobs_trajectories(batch, backward=True)
        # Get rewards from batch
        log_rewards = batch.get_terminating_rewards()

        # Trajectory balance loss
        loss = ((self.logZ.sum() + logprobs_f - logprobs_b - log_rewards).pow(2).mean())

        return loss

    def train(self, n_train_steps):
        pbar = tqdm(range(1, n_train_steps + 1))
        for it in pbar:
            batch = self.sample_batch(n_onpolicy=self.batch_size["forward"], n_replay=self.batch_size["replay"])
            for opt_step in range(self.opt_config.steps_per_batch):
                loss = self.trajectorybalance_loss(batch) 
                self.optimization_step(loss)

            self.buffer.add(batch)
    
