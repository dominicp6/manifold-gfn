import numpy as np

import torch
from torch import nn

from utils.common import set_device, set_float_precision


class MLP_Policy(nn.Module):
    def __init__(self, 
                 env, 
                 device, 
                 float_precision,
                 n_hid,
                 n_layers,
                 shared_weights=False,
                 base=None):
        super(MLP_Policy, self).__init__()
        # Device and float precision
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        # Input and output dimensions
        self.state_dim = env.policy_input_dim
        self.output_dim = env.policy_output_dim
        # Optional base model
        self.base = base
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.shared_weights = shared_weights
        self.instantiate()

    def instantiate(self):
        self.model = self.make_mlp(nn.LeakyReLU()).to(self.device, dtype=self.float)

    def __call__(self, states):
        return self.model(states)

    def make_mlp(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim] + [self.n_hid] * self.n_layers + [(self.output_dim)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                )
            )
        else:
            raise ValueError("Base Model must be provided when shared_weights is set to True")
        
        return mlp.to(dtype=self.float)
        

class Uniform_Policy:
    def __init__(self, env, device, float_precision):
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        self.output_dim = env.policy_output_dim

    def __call__(self, states):
        return torch.rand(states.shape[0], self.output_dim, device=self.device, dtype=self.float) * 2 * np.pi

    def sample(self, states):
        return self(states)