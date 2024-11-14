""" VonMisesNet definition. """
from typing import Tuple

import torch

from GNN_utils import get_normalization_module, GCNStack, ShallowFullyConnected


class GNN2VonMisesTransitions(torch.nn.Module):
    """
     A simple GNN architecture that predicts parameters of the Markov Transition Kernels's (mtk) Von Mises mixture distribution.

     Input: 
        - molecular graph (graph object)
        - trajectory step (int)
     Output: 
        mtk parameters for every rotateable bond on the molecular graph
        - means    (means of Von Mises mixture)
        - concs    (concentrations of Von Mises Mixture)
        - weights  (corresponding weights of Von Mises Mixture)
     """

    def __init__(self, num_vertices, num_vertex_features, hidden_dim: int = 256, num_layers: int = 32,
                 MLP_hidden_dim: int = 1024, final_output_dim: int = 1, reduce_method: str = 'mean', min_conc: float = 1.0,
                 max_conc: float = 20.0, init_norm: str = "", linear_first: bool = True, extra_norm: str = "",
                 extra_layers: int = 0, end_norm: str = "", conc_norm: str = ""):
        """
        :param num_vertices: Number of vertices (padded) per graph in the expected inputs.
        :param num_vertex_features: The number of features per node (vertex) in the expected inputs.
        :param hidden_dim: Dimension of hidden layers.
        :param num_layers: Number of times to repeat the message passing steps.
        :param MLP_hidden_dim: Dimension of final linear layers that make mtk predictions from embeddings.
        :param final_output_dim: Output dimension (for each of the three outputs).
        :param reduce_method: How to combine messages at each vertex during message passing.
        :param min_conc: Minimum allowed concentration per von Mises distribution.
        :param max_conc: Maximum allowed concentration per von Mises distribution.
        :param init_norm: What type of normalization method to use before message passing, if any.
        :param linear_first: Whether to have linear layers before (True) or after (False) each message passing step.
        :param extra_norm: What type of normalization method to use between each extra linear layer, if any.
        :param extra_layers: How many extra linear layers to run after each message passing step.
        :param end_norm: What type of normalization method to use after all message passing steps, if any.
        :param conc_norm: What type of normalization method to use on the concentration predictions, if any.
        """
        super(GNN2VonMisesTransitions, self).__init__()
        self.reduce_method = reduce_method
        self.min_conc, self.max_conc = min_conc, max_conc
        self.num_vertices, self.num_vertex_features = num_vertices, num_vertex_features
        self.hidden_size, self.num_layers = hidden_dim, num_layers
        self.MLP_hidden_dim = MLP_hidden_dim
        self.final_output_dim = final_output_dim
        self.vertex_featurize = torch.nn.Linear(self.num_vertex_features, self.hidden_size)
        self.stack = GCNStack(self.hidden_size, linear_first, self.num_layers,
                              num_vertices=self.num_vertices, extra_norm=extra_norm, extra_layers=extra_layers,
                              end_norm=end_norm, reduce=self.reduce_method)

        self.init_norm = get_normalization_module(init_norm, self.hidden_size, num_vertices=self.num_vertices)
        self.conc_norm = get_normalization_module(conc_norm, self.hidden_size, num_vertices=self.num_vertices)

        # mtk = Markov Transition Kernel (In this case, VonMisesDistributions)
        self.mtk_mean = ShallowFullyConnected(self.hidden_size, self.MLP_hidden_dim, self.final_output_dim)
        self.mtk_conc_layer = ShallowFullyConnected(self.hidden_size, self.MLP_hidden_dim, self.final_output_dim, self.conc_norm)
        self.mtk_weight = ShallowFullyConnected(self.hidden_size, self.MLP_hidden_dim, self.final_output_dim)
        self.weight_softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, batch) -> Tuple:
        """
        Forward pass.

        :param batch: Data batch.
        :return: Predictions.
        """
        # TODO: need to update the way that the batch is indexed
        v_i_in = self.vertex_featurize(batch['x'])
        G = batch['edge_index']

        if self.init_norm is not None:
            v_i_in = self.init_norm(v_i_in)

        v_i_in = self.stack(v_i_in, G=G)

        means = self.mtk_mean(v_i_in)

        concs = self.mtk_conc_layer(v_i_in)
        # Concentration must be strictly positive, between a certain minimum and maximum value
        concs = self.max_conc * torch.sigmoid(concs) + self.min_conc

        weights = self.mtk_weight(v_i_in)
        weights = self.weight_softmax(weights)

        return means, concs, weights



