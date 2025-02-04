"""
GFlowNet
"""
import os
from datetime import datetime
from typing import Optional
from copy import copy
from itertools import chain, product

import wandb
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from batch import Batch
from buffer import GeneralisedRewardBuffer
from utils.common import set_device, set_float_precision

def kl_divergence(p, q, epsilon=1e-10):
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
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
    
    def sample_action(self, state, backward: Optional[bool] = False, action_to_source: Optional[bool] = False):
        if backward:
            model = self.backward_policy
        else:
            model = self.forward_policy

        policy_state = self.env.encode_state(state)
        policy_output = model(policy_state)

        if torch.any(torch.isnan(policy_output)):
            if backward:
                raise ValueError("NaNs encountered in backward policy output.")
            else:
                raise ValueError("NaNs encountered in forward policy output.") 

        # Sample actions from policy outputs
        action = self.env.sample_actions_batch(policy_output, state, backward, action_to_source)

        return action, policy_state
    
    def backward_sample_trajectories(self, terminal_states):
        batch_size = terminal_states.shape[0]
        actions = torch.zeros((batch_size, self.traj_length, self.env.n_dim), device=self.device, dtype=self.float) 
        trajectories = torch.zeros((batch_size, self.traj_length + 1, self.env.n_dim + 1), device=self.device, dtype=self.float) 
        policy_trajectories = torch.zeros((batch_size, self.traj_length + 1, self.env.policy_input_dim), device=self.device, dtype=self.float)
        trajectories[:, -1, :] = torch.cat([terminal_states, self.traj_length * torch.ones(batch_size, 1, device=self.device, dtype=self.float)], dim=1) 
        
        for t in range(self.traj_length - 1, -1, -1):
            # t ranges from (self.trajectory_length - 1) to (0)
            actions[:, t, :], policy_trajectories[:, t + 1, :] = self.sample_action(trajectories[:, t + 1, :], backward=True, action_to_source=(t == 0))
            trajectories[:, t, :] = self.env.step_backwards(actions[:, t, :], trajectories[:, t + 1, :])
        # Compute the policy state for the first state in the trajectory
        policy_trajectories[:, 0, :] = self.env.encode_state(trajectories[:, 0, :])

        return trajectories, policy_trajectories, actions

    @torch.no_grad()
    def sample_batch(self, n_onpolicy: int = 0, n_replay: int = 0):
        """
        Builds a batch of data by sampling online and/or offline trajectories.
        """

        # Initialise batch tensors
        actions = torch.zeros((n_onpolicy, self.traj_length, self.env.n_dim), device=self.device, dtype=self.float) 
        trajectories = torch.zeros((n_onpolicy, self.traj_length + 1, self.env.n_dim + 1), device=self.device, dtype=self.float) 
        policy_trajectories = torch.zeros((n_onpolicy, self.traj_length + 1, self.env.policy_input_dim), device=self.device, dtype=self.float)
        log_rewards = torch.zeros(n_onpolicy, device=self.device, dtype=self.float) 

        # Sample an on-policy batch if n_onpolicy > 0
        if n_onpolicy > 0:
            trajectories[:, 0, :] = self.env.source
            for t in range(self.traj_length):
                actions[:, t, :], policy_trajectories[:, t, :] = self.sample_action(trajectories[:, t, :])
                trajectories[:, t + 1, :] = self.env.step(actions[:, t, :], trajectories[:, t, :])
            # Compute the policy state for the last state in the trajectory
            policy_trajectories[:, -1, :] = self.env.encode_state(trajectories[:, -1, :])

            on_policy_batch = Batch(env=self.env, device=self.device, float_type=self.float, trajectories=trajectories, policy_trajectories=policy_trajectories, actions=actions)
            on_policy_batch.compute_rewards()

        # Sample a replay buffer batch if the buffer is not empty and n_replay > 0
        if self.buffer.size > 0 and n_replay > 0:
            terminal_states, log_rewards = self.buffer.sample(n_replay)
            trajectories, policy_trajectories, actions = self.backward_sample_trajectories(terminal_states)
            replay_buffer_batch = Batch(env=self.env, device=self.device, float_type=self.float, trajectories=trajectories, policy_trajectories=policy_trajectories, actions=actions, log_rewards=log_rewards)

        # Output the final batch
        if n_onpolicy > 0 and n_replay > 0 and self.buffer.size > 0:
            final_batch = on_policy_batch.merge(replay_buffer_batch)
            return final_batch, on_policy_batch
        elif n_onpolicy > 0:
            return on_policy_batch, on_policy_batch
        elif n_replay > 0:
            return replay_buffer_batch, None
        else:
            raise ValueError("At least one of n_onpolicy or n_replay must be greater than 0.")

    def compute_logprobs_trajectories(self, batch: Batch, backward: bool = False):
        if backward:
            # Backward trajectories
            policy_output_b = self.backward_policy(batch.policy_trajectories)
            if torch.any(torch.isnan(batch.policy_trajectories)):
                raise ValueError("NaNs encountered in policy trajectories.")
            if torch.any(torch.isnan(policy_output_b)):
                policy_output_b = self.backward_policy(batch.policy_trajectories)
                raise ValueError("NaNs encountered in backward policy output.")
            logprobs = self.env.get_traj_batch_logprobs(policy_output_b, batch.actions, backwards=True)
        else:
            # Forward trajectories
            policy_output_f = self.forward_policy(batch.policy_trajectories)
            if torch.any(torch.isnan(batch.policy_trajectories)):
                raise ValueError("NaNs encountered in policy trajectories.")
            if torch.any(torch.isnan(policy_output_f)):
                policy_output_f = self.forward_policy(batch.policy_trajectories)
                raise ValueError("NaNs encountered in forward policy output.")
            logprobs = self.env.get_traj_batch_logprobs(policy_output_f, batch.actions)
        
        return logprobs

    def trajectorybalance_loss(self, batch):
        # Get logprobs of forward and backward transitions
        logprobs_f = self.compute_logprobs_trajectories(batch, backward=False)
        logprobs_b = self.compute_logprobs_trajectories(batch, backward=True)
        # Get rewards from batch
        log_rewards = batch.get_terminating_rewards()

        # Trajectory balance loss
        loss = ((self.logZ.sum() + logprobs_f - logprobs_b - log_rewards.clip(min=self.config.gflownet.log_reward_min)).pow(2).mean())

        return loss

    def train(self, n_train_steps):
        expected_density, ground_truth_density = self.compute_ground_truth_density()
        pbar = tqdm(range(1, n_train_steps + 1))
        for it in pbar:
            self.it += 1
            batch, on_policy_batch = self.sample_batch(n_onpolicy=self.batch_size.forward, n_replay=self.batch_size.replay)
            for opt_step in range(self.opt_config.steps_per_batch):
                loss = self.trajectorybalance_loss(batch) 
                self.optimization_step(loss)
            self.buffer.add(on_policy_batch)
            # Logging
            wandb.log({"loss": loss.item(), 'logZ': self.logZ.sum().item(), "lr": self.opt.param_groups[0]['lr']})
            pbar.set_description(f"Loss: {loss.item():.3f}, logZ: {self.logZ.sum().item():.3f}, lr: {self.opt.param_groups[0]['lr']:.3e}")

            if it % self.logger_config.log_interval == 0:
                pbar.set_description(f"Loss: {loss.item():.3f} [Evaluating metrics]")
                if self.env.n_dim == 2:
                    l1_divergence, kl_divergence, jsd_divergence = self.compute_divergence_metrics(expected_density, ground_truth_density)
                else:
                    l1_divergence, kl_divergence, jsd_divergence = self.estimate_divergence_metrics()
                wandb.log({"l1_divergence_e": l1_divergence[0], "kl_divergence_e": kl_divergence[0], "jsd_divergence_e": jsd_divergence[0]})
                wandb.log({"l1_divergence": l1_divergence[1], "kl_divergence": kl_divergence[1], "jsd_divergence": jsd_divergence[1]})

            if it % self.logger_config.checkpoint_interval == 0 and self.logger_config.checkpoints:
                self.save_checkpoint(pbar, loss)

            if it % self.logger_config.visualisation_interval == 0 and self.env.n_dim == 2:
                self.visualise(pbar, loss)

    def compute_ground_truth_density(self):
        if self.env.n_dim != 2:
            raise ValueError("Ground truth density can only be computed for 2D environments.")
        
        grid_size = self.config.logging.num_bins
        angles_per_dim = torch.linspace(-torch.pi, torch.pi, grid_size, device=self.device, dtype=self.float)

        # Generate a regular grid by taking the Cartesian product
        grid = torch.tensor(list(product(*[angles_per_dim] * self.env.n_dim)), device=self.device, dtype=self.float)
        conformers = self.env.statebatch2conformerbatch(grid)

        energies = self.env.proxy(*conformers)
        energies_grid = energies.reshape(grid_size, grid_size).cpu()
        
        # Plot heatmaps of the log reward and ground truth density
        log_rewards = (-self.env.beta * energies_grid).clamp(min=self.config.gflownet.log_reward_min)
        expected_density = torch.exp(log_rewards)
        expected_density /= expected_density.sum()
        fig, ax = plt.subplots()
        cax = ax.imshow(log_rewards.T, cmap="RdBu", extent=(-np.pi, np.pi, -np.pi, np.pi), origin="lower")
        ax.set_title("Log Reward")
        fig.colorbar(cax)
        wandb.log({"log_reward": wandb.Image(fig)})
        plt.close(fig)

        # Compute ground truth density
        ground_truth_density = torch.exp(-self.env.beta * energies_grid)
        
        # Check for NaNs in the ground truth density
        if torch.any(torch.isnan(ground_truth_density)):
            raise ValueError("NaNs encountered in ground truth density.")
        
        # Normalize the density
        ground_truth_density /= (ground_truth_density).sum()

        # Plot the ground truth density
        fig, ax = plt.subplots()
        cax = ax.imshow(ground_truth_density.view(grid_size, grid_size).T.cpu(), cmap="Reds", extent=(-np.pi, np.pi, -np.pi, np.pi), origin="lower")
        ax.set_title("Ground Truth Density")
        fig.colorbar(cax)
        wandb.log({"ground_truth_density": wandb.Image(fig)})
        plt.close(fig)

        return expected_density.numpy(), ground_truth_density.numpy()

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
        # TODO: check that this is still correct
        conformers = self.env.statebatch2conformerbatch(states)
        energies = self.env.proxy(*conformers).cpu().numpy()
        
        if self.config.proxy.normalise:
            energies = (energies + 1) * (self.env.proxy.max_energy - self.env.proxy.min_energy) 
        
        boltzmann_weights = np.exp( - energies * self.env.beta)

        return boltzmann_weights
    
    def _compute_free_energy_histogram(self, states, boltzmann_weights, num_bins_per_dimension):
        histograms = []
        for dim in range(self.env.n_dim):
            # Extract the angles for the current dimension
            dim_marginal_states = states[:, dim].cpu().numpy()

            # Compute the histogram and weights for the Boltzmann density
            hist, _ = np.histogram(dim_marginal_states, bins=num_bins_per_dimension, range=(0, 2 * np.pi), weights=boltzmann_weights)
            norm_counts, _ = np.histogram(dim_marginal_states, bins=num_bins_per_dimension, range=(0, 2 * np.pi))
            average_density = np.divide(hist, norm_counts, out=np.zeros_like(hist), where=norm_counts > 0)
            normalised_density = np.divide(average_density, np.sum(average_density))

            # Store results
            histograms.append(normalised_density)
        
        return histograms
    
    def compute_divergence_metrics(self, expected_density, ground_truth_density):
        assert self.env.n_dim == 2, "Ground truth density can only be computed for 2D environments."

        n_samples = self.logger_config.n_uniform_samples
        batch, _ = self.sample_batch(n_onpolicy=n_samples, n_replay=0)
        terminating_states = batch.get_terminating_states()
        hist, _, _ = np.histogram2d(terminating_states[:,0].cpu().numpy(), terminating_states[:,1].cpu().numpy(), bins=(self.config.logging.num_bins,self.config.logging.num_bins), density=True)
        current_sum = np.sum(hist)# should be a number very close to 1
        hist /= current_sum       # normalise the histogram, so that the integral is 1
        assert np.isclose(np.sum(hist), 1), "The empirical histogram must be normalised."
        assert np.isclose(np.sum(ground_truth_density), 1), "The ground truth density must be normalised."
        
        # Compute divergence with expected density
        l1_div_e = np.mean(np.abs(expected_density - hist))
        kl_div_e = kl_divergence(expected_density, hist)
        jsd_div_e = 0.5 * kl_divergence(expected_density, 0.5 * (expected_density + hist)) + 0.5 * kl_divergence(hist, 0.5 * (expected_density + hist))

        # Compute divergence with ground truth density
        l1_div = np.mean(np.abs(ground_truth_density - hist))
        kl_div = kl_divergence(ground_truth_density, hist)
        jsd_div = 0.5 * kl_divergence(ground_truth_density, 0.5 * (ground_truth_density + hist)) + 0.5 * kl_divergence(hist, 0.5 * (ground_truth_density + hist))

        return (l1_div_e, l1_div), (kl_div_e, kl_div), (jsd_div_e, jsd_div)

    def estimate_divergence_metrics(self):
        """
        Computes metrics by sampling trajectories from the forward policy.
        """

        n_samples = self.logger_config.n_uniform_samples

        # Compute marginal histograms based on random samples 
        angles_uniform = 2 * np.pi * torch.rand((n_samples, self.env.n_dim), device=self.device, dtype=self.float) - np.pi
        boltzmann_weights = self._compute_boltzman_weights(angles_uniform)
        histograms = self._compute_free_energy_histogram(angles_uniform, boltzmann_weights, num_bins_per_dimension=self.logger_config.num_bins)

        # Compute marginal histograms based on samples from the forward policy
        batch, _ = self.sample_batch(n_onpolicy=self.logger_config.n_onpolicy_samples, n_replay=0)
        terminating_states = batch.get_terminating_states()
        histograms_policy = [np.histogram(terminating_states[:, dim].cpu().numpy(), bins=self.logger_config.num_bins, density=True)[0] for dim in range(self.env.n_dim)]

        # Compute the average L1, KL and JSD divergences between the uniform-estimated Boltzmann densities and the policy-estimated Boltzmann densities
        l1_divs = [np.mean(np.abs(histograms[dim] - histograms_policy[dim])) for dim in range(self.env.n_dim)]
        kl_divs = [kl_divergence(histograms[dim], histograms_policy[dim]) for dim in range(self.env.n_dim)]
        
        if any(np.isinf(kl_divs)):
            raise ValueError("KL divergence is infinite when computing the metric. This is likely due to a lack of samples during evaluation. Increase the number of samples and try again.")

        jsd_divs = [0.5 * kl_divergence(histograms[dim], 0.5 * (histograms[dim] + histograms_policy[dim])) + 0.5 * kl_divergence(histograms_policy[dim], 0.5 * (histograms[dim] + histograms_policy[dim])) for dim in range(self.env.n_dim)]

        return np.mean(l1_divs), np.mean(kl_divs), np.mean(jsd_divs) 
    
    def visualise(self, pbar, loss):
        """Visualise samples from on-policy and the replay buffer.."""

        pbar.set_description(f"Loss: {loss.item():.3f} [Visualising samples]")

        mosaic = {"mosaic" : [["op_hist", "op_1", "op_2"],
                              ["rb_cyclic_hist", "rb_cyclic_1", "rb_cyclic_2"],
                              ["rb_priority_hist", "rb_priority_1", "rb_priority_2"],],
                 "figsize" : (12, 8),}

        fig, axs = plt.subplot_mosaic(**mosaic)

        onpolicy_batch, _ = self.sample_batch(n_onpolicy=5000, n_replay=0)
        terminating_states = onpolicy_batch.get_terminating_states()
        
        axs[f"op_hist"].hist2d(terminating_states[:,0].cpu().numpy(), terminating_states[:,1].cpu().numpy(), bins=(20,20), density=True, cmap="Reds")
        axs[f"op_hist"].set_xlim(-np.pi, np.pi)
        axs[f"op_hist"].set_ylim(-np.pi, np.pi)
        axs[f"op_1"].hist(terminating_states[:,0].cpu().numpy(), bins=20, density=True, color="red", alpha=0.5, linewidth=0.05)
        axs[f"op_2"].hist(terminating_states[:,1].cpu().numpy(), bins=20, density=True, color="red", alpha=0.5, linewidth=0.05)

        terminating_states, _ = self.buffer.cyclic_buffer.sample(batch_size=5000)

        axs[f"rb_cyclic_hist"].hist2d(terminating_states[:,0].cpu().numpy(), terminating_states[:,1].cpu().numpy(), bins=(20,20), density=True, cmap="Greens")
        axs[f"rb_cyclic_hist"].set_xlim(-np.pi, np.pi)
        axs[f"rb_cyclic_hist"].set_ylim(-np.pi, np.pi)
        axs[f"rb_cyclic_1"].hist(terminating_states[:,0].cpu().numpy(), bins=20, density=True, color="green", alpha=0.5, linewidth=0.05)
        axs[f"rb_cyclic_2"].hist(terminating_states[:,1].cpu().numpy(), bins=20, density=True, color="green", alpha=0.5, linewidth=0.05)

        terminating_states, _ = self.buffer.priority_buffer.sample(batch_size=5000)

        axs[f"rb_priority_hist"].hist2d(terminating_states[:,0].cpu().numpy(), terminating_states[:,1].cpu().numpy(), bins=(20,20), density=True, cmap="Greens")
        axs[f"rb_priority_hist"].set_xlim(-np.pi, np.pi)
        axs[f"rb_priority_hist"].set_ylim(-np.pi, np.pi)
        axs[f"rb_priority_1"].hist(terminating_states[:,0].cpu().numpy(), bins=20, density=True, color="green", alpha=0.5, linewidth=0.05)
        axs[f"rb_priority_2"].hist(terminating_states[:,1].cpu().numpy(), bins=20, density=True, color="green", alpha=0.5, linewidth=0.05)

        wandb.log({"visualisation": wandb.Image(fig)})