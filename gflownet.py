"""
GFlowNet
"""
import os
from datetime import datetime
from typing import Optional
from copy import copy
from itertools import chain

import wandb
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from batch import Batch
from buffer import GeneralisedRewardBuffer
from utils.common import set_device, set_float_precision

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

class GFlowNet:
    def __init__(
        self,
        device,
        float_precision,
        env,
        forward_policy,
        backward_policy,
        config,
        logging_dir: Optional[str] = None
    ):
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        self.config = config
        self.env = env
        self.traj_length = self.env.traj_length
        self.batch_size = config.gflownet.batch_size
        self.opt_config = config.optimizer_config
        self.it = 0
        boltzmann_constant = 3.1668114e-6 # hartree/K
        self.beta = 1.0 / (boltzmann_constant * self.config.general.T) # hartree^-1 

        # Define logZ as the SUM of the values in this tensor. Dimension > 1 only to accelerate learning.
        self.logZ = nn.Parameter(self.opt_config.initial_z_scaling * torch.ones(self.opt_config.z_dim) / self.opt_config.z_dim) 

        # Buffers
        self.buffer = GeneralisedRewardBuffer(self.env.n_dim, self.device, self.float, config.gflownet.regular_capacity, config.gflownet.priority_capacity, config.gflownet.priority_ratio)

        # Policy models
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
       
        # Optimizer
        self._init_optimizer()

        # Logging
        self.logger_config = config.logging
        if logging_dir is not None:
            assert self.config.logging.checkpoints, "Checkpoints must be enabled to use a custom logging directory."
            self.logging_dir = logging_dir
            # Check that the directory exists
            if not os.path.exists(self.logging_dir):
                raise ValueError(f"Logging directory {self.logging_dir} does not exist.")
        else:
            if self.config.logging.checkpoints:
                date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.logging_dir = f"{config.logging.log_directory}/{date_time}"
                os.makedirs(self.logging_dir)

    def parameters(self):
        return list(self.forward_policy.model.parameters()) + list(self.backward_policy.model.parameters())

    def _init_optimizer(self):
        """
        Set up the optimizer
        """
        params = self.parameters()
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
        if not self.config.backward_policy.uniform:
            return chain(self.forward_policy.model.parameters(), self.backward_policy.model.parameters())
        else:
            return self.forward_policy.model.parameters()
    
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
        actions = torch.zeros((batch_size, self.traj_length, self.env.n_dim), device=self.device, dtype=self.float) 
        trajectories = torch.zeros((batch_size, self.traj_length + 1, self.env.n_dim + 1), device=self.device, dtype=self.float) 
        policy_trajectories = torch.zeros((batch_size, self.traj_length + 1, self.env.policy_input_dim), device=self.device, dtype=self.float)
        trajectories[:, -1, :] = torch.cat([terminal_states, self.traj_length * torch.ones(batch_size, 1, device=self.device, dtype=self.float)], dim=1) 
        
        for t in range(self.traj_length - 1, -1, -1):
            # t ranges from (self.trajectory_length - 1) to (0)
            actions[:, t, :], policy_trajectories[:, t, :] = self.sample_action(trajectories[:, t + 1, :], backward=True)
            self.env.step_backwards(actions[:, t, :], trajectories[:, t, :])
            trajectories[:, t, :] = copy(self.env.state)

        return trajectories, policy_trajectories, actions

    @torch.no_grad()
    def sample_batch(self, n_onpolicy: int = 0, n_replay: int = 0):
        """
        Builds a batch of data by sampling online and/or offline trajectories.
        """

        # Initialise an on-policy batch 
        actions = torch.zeros((n_onpolicy, self.traj_length, self.env.n_dim), device=self.device, dtype=self.float) 
        trajectories = torch.zeros((n_onpolicy, self.traj_length + 1, self.env.n_dim + 1), device=self.device, dtype=self.float) 
        policy_trajectories = torch.zeros((n_onpolicy, self.traj_length + 1, self.env.policy_input_dim), device=self.device, dtype=self.float)
        log_rewards = torch.zeros(n_onpolicy, device=self.device, dtype=self.float)

        trajectories[:, 0, :] = self.env.source
        # Generate on-policy trajectories
        for t in range(self.traj_length):
            actions[:, t, :], policy_trajectories[:, t, :] = self.sample_action(trajectories[:, t, :])
            trajectories[:, t + 1, :] = self.env.step(actions[:, t, :], trajectories[:, t, :])

        on_policy_batch = Batch(env=self.env, device=self.device, float_type=self.float, trajectories=trajectories, policy_trajectories=policy_trajectories, actions=actions)
        on_policy_batch.compute_rewards()

        # Sample a replay buffer batch if the buffer is not empty
        if self.buffer.size > 0 and n_replay > 0:
            terminal_states, log_rewards = self.buffer.sample(n_replay)
            trajectories, policy_trajectories, actions = self.backward_sample_trajectories(terminal_states)
            replay_buffer_batch = Batch(env=self.env, device=self.device, float_type=self.float, trajectories=trajectories, policy_trajectories=policy_trajectories, actions=actions, log_rewards=log_rewards)

            # Merge the batches
            final_batch = on_policy_batch.merge(replay_buffer_batch)

            return final_batch
        else:
            return on_policy_batch

    def compute_logprobs_trajectories(self, batch: Batch, backward: bool = False):
        if backward:
            # Backward trajectories
            policy_output_b = self.backward_policy(batch.policy_trajectories)
            logprobs = self.env.get_traj_batch_logprobs(policy_output_b, batch.actions, backwards=True)
        else:
            # Forward trajectories
            policy_output_f = self.forward_policy(batch.policy_trajectories)
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
            self.it += 1
            batch = self.sample_batch(n_onpolicy=self.batch_size.forward, n_replay=self.batch_size.replay)
            for opt_step in range(self.opt_config.steps_per_batch):
                loss = self.trajectorybalance_loss(batch) 
                self.optimization_step(loss)
            self.buffer.add(batch)

            # Logging
            wandb.log({"loss": loss.item()})
            pbar.set_description(f"Loss: {loss.item():.3f}")

            if it % self.logger_config.log_interval == 0:
                pbar.set_description(f"Loss: {loss.item():.3f} [Evaluating metrics]")
                l1_divergence, kl_divergence, jsd_divergence = self.compute_metrics()
                wandb.log({"l1_divergence": l1_divergence, "kl_divergence": kl_divergence, "jsd_divergence": jsd_divergence})

            if it % self.logger_config.checkpoint_interval == 0 and self.logger_config.checkpoints:
                self.save_checkpoint(pbar, loss)

    def save_checkpoint(self, pbar, loss):
        pbar.set_description(f"Loss: {loss.item():.3f} [Saving checkpoint]")
        fwd_pol_state_dict = self.forward_policy.model.state_dict()
        # Append "model." to the keys of the forward policy state dict
        fwd_pol_state_dict = {f"model.{k}": v for k, v in fwd_pol_state_dict.items()}
        bkwd_pol_state_dict = self.backward_policy.model.state_dict() if not self.config.backward_policy.uniform else None
        # Append "model." to the keys of the backward policy state dict
        if bkwd_pol_state_dict is not None:
            bkwd_pol_state_dict = {f"model.{k}": v for k, v in bkwd_pol_state_dict.items()}
        checkpoint = {
            "forward_policy": fwd_pol_state_dict,
            "backward_policy": bkwd_pol_state_dict,
            "optimizer": self.opt.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "logZ": self.logZ,
            "buffer": self.buffer,
            "it": self.it,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, f"{self.logging_dir}/checkpoint.pt")
        self.update_status_file()   

    def update_status_file(self):
        with open(f"{self.logging_dir}/status.txt", "w") as f:
            f.write(f"Checkpoint saved at step {self.it}\n")

    def _compute_boltzman_weights(self, states):
        conformers = self.env.statebatch2conformerbatch(states)
        energies = self.env.proxy(*conformers).cpu().numpy()
        
        if self.config.proxy.normalise:
            energies = (energies + 1) * (self.env.proxy.max_energy - self.env.proxy.min_energy) 
        
        boltzmann_weights = np.exp( - energies * self.beta)

        return boltzmann_weights
    
    def _compute_free_energy_histogram(self, states, boltzmann_weights):
        n_bins_per_dimension = self.logger_config.num_bins
        histograms = []
        for dim in range(self.env.n_dim):
            # Extract the angles for the current dimension
            dim_marginal_states = states[:, dim].cpu().numpy()

            # Compute the histogram and weights for the Boltzmann density
            hist, _ = np.histogram(dim_marginal_states, bins=n_bins_per_dimension, range=(0, 2 * np.pi), weights=boltzmann_weights)
            norm_counts, _ = np.histogram(dim_marginal_states, bins=n_bins_per_dimension, range=(0, 2 * np.pi))
            average_density = np.divide(hist, norm_counts, out=np.zeros_like(hist), where=norm_counts > 0)
            normalised_density = np.divide(average_density, np.sum(average_density))

            # Store results
            histograms.append(normalised_density)
        
        return histograms

    def compute_metrics(self):
        """
        Computes metrics by sampling trajectories from the forward policy.
        """

        n_samples = self.logger_config.n_uniform_samples

        # Compute marginal histograms based on random samples 
        angles_uniform = 2 * np.pi * torch.rand((n_samples, self.env.n_dim), device=self.device, dtype=self.float) 
        boltzmann_weights = self._compute_boltzman_weights(angles_uniform)
        histograms = self._compute_free_energy_histogram(angles_uniform, boltzmann_weights)

        # Compute marginal histograms based on samples from the forward policy
        batch = self.sample_batch(n_onpolicy=self.logger_config.n_onpolicy_samples, n_replay=0)
        terminating_states = batch.get_terminating_states()
        boltzmann_weights = self._compute_boltzman_weights(terminating_states)
        histograms_policy = self._compute_free_energy_histogram(terminating_states, boltzmann_weights)

        # Compute the average L1, KL and JSD divergences between the uniform-estimated Boltzmann densities and the policy-estimated Boltzmann densities
        l1_divergences = [np.mean(np.abs(histograms[dim] - histograms_policy[dim])) for dim in range(self.env.n_dim)]
        kl_divergences = [kl_divergence(histograms[dim], histograms_policy[dim]) for dim in range(self.env.n_dim)]
        
        if any(np.isinf(kl_divergences)):
            raise ValueError("KL divergence is infinite when computing the metric. This is likely due to a lack of samples during evaluation. Increase the number of samples and try again.")

        jsd_divergences = [0.5 * kl_divergence(histograms[dim], 0.5 * (histograms[dim] + histograms_policy[dim])) + 0.5 * kl_divergence(histograms_policy[dim], 0.5 * (histograms[dim] + histograms_policy[dim])) for dim in range(self.env.n_dim)]

        return np.mean(l1_divergences), np.mean(kl_divergences), np.mean(jsd_divergences) 
    
