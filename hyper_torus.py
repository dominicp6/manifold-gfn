from typing import List, Optional, Tuple

import numpy as np
import torch

from torch.distributions import Categorical, MixtureSameFamily, VonMises
from torchtyping import TensorType
from torchtyping import TensorType

from utils.common import set_float_precision


class HyperTorus:
    def __init__(self, proxy, device, n_dim: int, float_precision, config):
        self.device = device
        self.float = set_float_precision(float_precision)
        
        self.proxy = proxy

        self.n_dim = n_dim
        self.n_comp = config.env.n_comp
        self.traj_length = config.env.traj_length
        self.encoding_multiplier = config.env.encoding_multiplier

        self.policy_input_dim = self.n_dim * self.encoding_multiplier * 2 + 1
        self.policy_output_dim = self.n_dim * self.n_comp * 3

        self.source_angles = torch.zeros(self.n_dim, device=self.device)
        self.source = torch.zeros(self.n_dim + 1, device=self.device)
        self.state = self.source

        self.vonmises_min_concentration = config.env.vonmises_min_concentration
        self.config = config

        print("HyperTorus initialised")
        print(f"n_dim: {self.n_dim}")
        print(f"n_comp: {self.n_comp}")
        print(f"traj_length: {self.traj_length}")
        print(f"encoding_multiplier: {self.encoding_multiplier}")
    
    def get_mixture_distribution(self, policy_outputs: TensorType["batch_size", "policy_output_dim"]):
        batch_size, output_size = policy_outputs.shape
        
        # Initialise VonMises mixture distribution:
        # --> Compute mixture weights
        indices = torch.arange(0, output_size // 3, device=self.device)
        mix_logits = torch.index_select(policy_outputs, dim=1, index=indices).reshape(-1, self.n_dim, self.n_comp)
        mixture_probs = Categorical(logits=mix_logits)

        # --> Compute VonMises means
        indices = torch.arange(output_size // 3, 2 * output_size // 3, device=self.device)
        means = torch.index_select(policy_outputs, dim=1, index=indices).reshape(-1, self.n_dim, self.n_comp)

        # --> Compute VonMises concentrations
        indices = torch.arange(2 * output_size // 3, output_size, device=self.device)
        concentrations = torch.index_select(policy_outputs, dim=1, index=indices).reshape(-1, self.n_dim, self.n_comp)

        # --> Initialise distribution
        vonmises = VonMises(means, torch.exp(concentrations) + self.vonmises_min_concentration)
        mixture_distribution = MixtureSameFamily(mixture_probs, vonmises)

        return mixture_distribution, batch_size
    
    def get_traj_mixture_distribution(self, traj_policy_outputs: TensorType["batch_size", "policy_output_dim"]):
        batch_size, traj_length, output_size = traj_policy_outputs.shape
        
        # Initialise VonMises mixture distribution:
        # --> Compute mixture weights
        indices = torch.arange(0, output_size // 3, device=self.device)
        mix_logits = torch.index_select(traj_policy_outputs, dim=2, index=indices).reshape(-1, self.traj_length, self.n_dim, self.n_comp)
        mixture_probs = Categorical(logits=mix_logits)

        # --> Compute VonMises means
        indices = torch.arange(output_size // 3, 2 * output_size // 3, device=self.device)
        means = torch.index_select(traj_policy_outputs, dim=2, index=indices).reshape(-1, self.traj_length, self.n_dim, self.n_comp)

        # --> Compute VonMises concentrations
        indices = torch.arange(2 * output_size // 3, output_size, device=self.device)
        concentrations = torch.index_select(traj_policy_outputs, dim=2, index=indices).reshape(-1, self.traj_length, self.n_dim, self.n_comp)

        # --> Initialise distribution
        vonmises = VonMises(means, torch.exp(concentrations) + self.vonmises_min_concentration)
        mixture_distribution = MixtureSameFamily(mixture_probs, vonmises)

        return mixture_distribution, batch_size
    
    def encode_state(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch", "policy_input_dim"]:
        # Last column of the state dim represents the trajectory step (integer)
        step_index = states[:, -1:]

        # All other columns represent the state angles
        angles = states[:, :-1]
        
        # Create trigonmetry coefficients with broadcasting
        int_coeff = torch.arange(1, self.encoding_multiplier + 1, device=angles.device).view(1, 1, -1)

        # Broadcast angles and compute encodings
        encoding = angles.unsqueeze(-1) * int_coeff  # Shape: (batch, num_angles, self.encoding_multiplier)
        encoding = encoding.flatten(start_dim=1)     # Flatten to (batch, num_angles * self.encoding_multiplier)

        # Compute trigonometric components
        cos_sin_encoding = torch.cat([torch.cos(encoding), torch.sin(encoding)], dim=1)

        # Concatenate with step
        states = torch.cat([cos_sin_encoding, step_index], dim=1)

        return states 
    
    def postprocess_params(self, params):
        # Process VonMises means
        params[:, :self.n_comp] = np.pi * torch.atan(params[:, :self.n_comp]) * 2

        # Process VonMises concentrations
        params[:, self.n_comp: 2*self.n_comp] = torch.exp(self.min_log_conc + (self.max_log_conc - self.min_log_conc) * torch.sigmoid(params[:, self.n_comp: 2*self.n_comp]))

        # Process Weights
        params[:, 2*self.n_comp: 3*self.n_comp] = torch.softmax(params[:, 2*self.n_comp: 3*self.n_comp])

        return params

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["batch_size", "policy_output_dim"],
        states_from: Optional[List] = None,
        backward: Optional[bool] = False,
    ) -> Tuple[TensorType["batch_size, traj_length, n_dim"], TensorType["batch_size"]]:

        mixture_distribution, batch_size = self.get_mixture_distribution(policy_outputs)

        # Initialise a logprobs and actions tensor
        logprobs = torch.zeros((batch_size, self.n_dim), dtype=self.float, device=self.device)
        actions_tensor = torch.zeros((batch_size, self.n_dim), dtype=self.float, device=self.device)    

        # Sample angles and evaluate logprobs
        actions_tensor = mixture_distribution.sample()

        # Catch special case for backwards back-to-source (BTS) actions
        if backward:
            source_angles = self.source[: self.n_dim]
            states_from_angles = states_from[:, : self.n_dim]
            actions_bts = states_from_angles - source_angles
            actions_tensor = actions_bts

        return actions_tensor

    def get_traj_batch_logprobs(
        self,
        policy_outputs: TensorType["batch_size", "traj_length + 1", "policy_output_dim"],
        actions: TensorType["batch_size", "traj_length", "n_dim"],
        backwards: bool = False,
    ) -> TensorType["batch_size"]:
        # Ignore the last policy output because it is not used
        if backwards:
            if self.config.backward_policy.uniform:
                logprobs = torch.ones(actions.shape[0], device=self.device) * -np.log(2 * np.pi)
                return logprobs
            else:
                mixture_distribution, _ = self.get_traj_mixture_distribution(policy_outputs[:, 1:, :])
        else:
            mixture_distribution, _ = self.get_traj_mixture_distribution(policy_outputs[:, :-1, :])
        logprobs = mixture_distribution.log_prob(actions)
        logprobs = logprobs.sum(dim=(1, 2))

        return logprobs

    def step(self, action, state):
        updated_angles = (state[..., :-1] + action) % (2 * np.pi)
        updated_index = state[..., -1] + 1
        
        return torch.cat([updated_angles, updated_index.unsqueeze(-1)], dim=-1)

    def step_backwards(self, action, state):
        updated_angles = (state[..., :-1] - action) % (2 * np.pi)
        updated_index = state[..., -1] - 1

        return torch.cat([updated_angles, updated_index.unsqueeze(-1)], dim=-1)

    def get_max_traj_length(self):
        return int(self.length_traj) + 1