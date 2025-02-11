import os
from itertools import product
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from conformer import ConformerEnvironment

class AlanineDipeptide(ConformerEnvironment):

    def __init__(self, device, float_precision, proxy, config):
        molecule_name = "alanine_dipeptide"
        smiles = "CC(=O)N[C@@H](C)C(=O)NC"
        super().__init__(device, float_precision, proxy, config, molecule_name=molecule_name, smiles=smiles)

    def plot_density(self, log_rewards, ground_truth_density):
        # Plot log reward
        fig, ax = plt.subplots()
        cax = ax.imshow(log_rewards.T.cpu(), cmap="RdBu", extent=(-np.pi, np.pi, -np.pi, np.pi), origin="lower")
        ax.set_title("Log Reward")
        fig.colorbar(cax)
        wandb.log({"log_reward": wandb.Image(fig)})
        plt.close(fig)

        # Plot ground truth density
        fig, ax = plt.subplots()
        cax = ax.imshow(ground_truth_density.T.cpu(), cmap="Reds", extent=(-np.pi, np.pi, -np.pi, np.pi), origin="lower")
        ax.set_title("Ground Truth Density")
        fig.colorbar(cax)
        wandb.log({"ground_truth_density": wandb.Image(fig)})
        plt.close(fig)

    def compute_ground_truth_density(self):
        # TODO: make this function more robust and refactor
        if self.n_dim not in [2, 4]:
            raise ValueError("Ground truth density can only be computed for 2D or 4D environments (alanine dipeptide).")
        
        # Check if the ground truth density has already been computed for this molecule and these parameters
        if os.path.exists(f"./groundtruth/{self.smiles}_{self.config.logging.num_bins}_{self.config.logging.num_samples_per_cell}_{self.n_dim}_expected_density.npy") and os.path.exists(f"./groundtruth/{self.smiles}_{self.config.logging.num_bins}_{self.config.logging.num_samples_per_cell}_{self.n_dim}_ground_truth_density.npy"):
            expected_density = np.load(f"./groundtruth/{self.smiles}_{self.config.logging.num_bins}_{self.config.logging.num_samples_per_cell}_{self.n_dim}_expected_density.npy")
            ground_truth_density = np.load(f"./groundtruth/{self.smiles}_{self.config.logging.num_bins}_{self.config.logging.num_samples_per_cell}_{self.n_dim}_ground_truth_density.npy")
            print("Ground truth density already computed. Loading from file.")
            log_rewards = torch.log(torch.tensor(expected_density, device=self.device, dtype=self.float))
            self.plot_density(log_rewards, torch.tensor(ground_truth_density, device=self.device, dtype=self.float))
            return expected_density, ground_truth_density

        print("Ground truth density not found for these parameters. Computing now.")
        grid_size = self.config.logging.num_bins
        N = self.config.logging.num_samples_per_cell  # Number of random samples per coarse grid cell
        cell_size = 2 * torch.pi / grid_size

        # Define centers of the coarse grid cells.
        centers = torch.linspace(-torch.pi + cell_size/2, torch.pi - cell_size/2, grid_size, device=self.device, dtype=self.float)

        if self.n_dim == 2:
            grid_centers = torch.tensor(list(product(centers, repeat=2)), device=self.device, dtype=self.float)
            energies_grid = torch.zeros(grid_size, grid_size, device=self.device, dtype=self.float)

            pbar = tqdm(range(grid_centers.shape[0]))
            pbar.set_description("Computing ground truth density")
            for idx, center in enumerate(grid_centers):
                # Sample uniformly within [-cell_size/2, cell_size/2] for each dimension.
                random_offsets = (torch.rand(N, 2, device=self.device, dtype=self.float) - 0.5) * cell_size
                sampled_points = center.unsqueeze(0) + random_offsets  # (N, 2)
                energy_samples = self.proxy(*self.statebatch2conformerbatch(sampled_points))
                cell_energy = energy_samples.mean()
                energies_grid[idx // grid_size, idx % grid_size] = cell_energy
                pbar.update(1)

        elif self.n_dim == 4:
            grid_centers = torch.tensor(list(product(centers, repeat=4)), device=self.device, dtype=self.float)
            energies_grid = torch.zeros(grid_size, grid_size, grid_size, grid_size, device=self.device, dtype=self.float)

            pbar = tqdm(range(grid_centers.shape[0]))
            pbar.set_description("Computing ground truth density")
            for idx, center in enumerate(grid_centers):
                random_offsets = (torch.rand(N, 4, device=self.device, dtype=self.float) - 0.5) * cell_size
                sampled_points = center.unsqueeze(0) + random_offsets  # (N, 4)
                energy_samples = self.proxy(*self.statebatch2conformerbatch(sampled_points))
                cell_energy = energy_samples.mean()
                # Unravel the flat index into 4D indices (assuming lexicographic order)
                i = idx // (grid_size**3) % grid_size
                j = idx // (grid_size**2) % grid_size
                k = idx // grid_size % grid_size
                l = idx % grid_size
                energies_grid[i, j, k, l] = cell_energy
                pbar.update(1)

            # Marginalize over omega1 and omega2 (assumed to be the last two dimensions)
            boltzmann_weights = torch.exp(-self.beta * energies_grid)
            weighted_energies_grid = energies_grid * boltzmann_weights
            weighted_energies_grid = weighted_energies_grid.sum(dim=(2, 3))
            normalizing_constant = boltzmann_weights.sum(dim=(2, 3))
            energies_grid = weighted_energies_grid / normalizing_constant

        log_rewards = (-self.beta * energies_grid)
        expected_density = torch.exp(log_rewards)
        expected_density /= expected_density.sum()
        
        ground_truth_density = torch.exp(-self.beta * energies_grid)
        if torch.any(torch.isnan(ground_truth_density)):
            raise ValueError("NaNs encountered in ground truth density.")
        ground_truth_density /= ground_truth_density.sum()

        # Compute log reward and expected density
        self.plot_density(log_rewards, ground_truth_density)

        # Save the expected and ground truth densities as numpy arrays to the folder ./groundtruth
        # Give them the name of the molecule's smile string, followed by _ separated num_bins, num_samples_per_cell and n_dim
        np.save(f"./groundtruth/{self.smiles}_{grid_size}_{N}_{self.n_dim}_expected_density.npy", expected_density.cpu().numpy())
        np.save(f"./groundtruth/{self.smiles}_{grid_size}_{N}_{self.n_dim}_ground_truth_density.npy", ground_truth_density.cpu().numpy())

        return expected_density.cpu().numpy(), ground_truth_density.cpu().numpy()