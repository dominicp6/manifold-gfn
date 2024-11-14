import torch
import torch.nn.functional as F

NORMALIZATIONS = {
    "layer": torch.nn.LayerNorm,
    "batch": torch.nn.BatchNorm1d,
    "instance": torch.nn.InstanceNorm1d
}

def get_normalization_module(norm_type: str, hidden_size: int, num_vertices: int = None):
    """
    Get normalization module.

    :param norm_type: Normalization type.
    :param hidden_size: Hidden size.
    :param num_vertices: Number of vertices.
    :return: Normalization module.
    """
    norm = NORMALIZATIONS.get(norm_type, None)
    if norm is not None:
        if norm_type == "batch":
            norm = norm(num_vertices)
        else:
            norm = norm(hidden_size)
    return norm

class GCNStack(torch.nn.Module):
    """
    A stack of GCN layers with options for how to loop over each layer.
    """

    def __init__(self,
                 hidden_size: int,
                 order: bool,
                 num_layers: int,
                 reduce: str = 'mean',
                 num_vertices: int = None,
                 extra_norm: str = "",
                 extra_layers: int = 0,
                 end_norm: str = "",
                 use_residual: bool = True):
        """
        :param hidden_size: Feature size of expected input tensors.
        :param num_vertices: Number of vertices per graph of expected inputs.
        :param extra_norm: Normalizations before each extra layer (if non-empty, length should equal extra_layers).
        :param extra_layers: Number of extra linear layers to add.
        :param end_norm: Normalization to apply after all extra layers.
        :param order: Whether to do the linear layer before the message passing (True) or after (False).
        :param reduce: How to merge messages passed to each vertex (see reduce in pytorch_scatter).
        :param use_residual: Whether to add the residual at each layer.
        """
        super(GCNStack, self).__init__()
        self.hidden_size = hidden_size
        self.num_vertices = num_vertices
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.reduce = reduce
        self.layers = torch.nn.ModuleList([GCNLayer(self.hidden_size, reduce=self.reduce,
                                                    num_vertices=self.num_vertices, extra_norm=extra_norm,
                                                    extra_layers=extra_layers, end_norm=end_norm, order=order)
                                           for _ in range(self.num_layers)])

    def forward(self, v_i_in, G=None):
        """
        Forward pass.

        :param v_i_in: Features tensors for nodes in graph.
        :param G: Adjacency matrix of graph.
        """
        for l in self.layers:
            v_i_c = l(v_i_in, G=G)
            if self.use_residual:
                v_i_in = v_i_in + v_i_c
            else:
                v_i_in = v_i_c
        return v_i_in


class GCNLayer(torch.nn.Module):
    """
    A single GCN layer with message passing.
    """

    def __init__(self, hidden_size,
                 reduce: str = "mean",
                 num_vertices: int = None,
                 extra_norm: str = "",
                 extra_layers: int = 0,
                 end_norm: str = "",
                 order: bool = True):
        """
        :param hidden_size: Feature size of expected input tensors.
        :param num_vertices: Number of vertices per graph of expected inputs.
        :param extra_norm: Normalizations before each extra layer (if non-empty, length should equal extra_layers).
        :param extra_layers: Number of extra linear layers to add.
        :param end_norm: Normalization to apply after all extra layers.
        :param order: Whether to do the linear layer before the message passing (True) or after (False).
        """
        super(GCNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.reduce = reduce
        self.num_vertices = num_vertices
        self.L_v = torch.nn.Linear(self.hidden_size, self.hidden_size)  # Linear layers for vertices
        self.N_extra = torch.nn.ModuleList([get_normalization_module(extra_norm, self.hidden_size,
                                                                     self.num_vertices) for _ in range(extra_layers)])
        self.L_extra = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size)
                                            for _ in range(extra_layers)])
        self.N_end = get_normalization_module(end_norm, self.hidden_size, self.num_vertices)
        self.order = order

    def forward(self, v_i_in, G=None):
        """
        Forward pass.

        :param v_i_in: Features tensors for nodes in graph.
        :param G: Adjacency matrix of graph.
        """
        if self.order:
            v_i_prime = self.L_v(v_i_in)
            v_i_e = self.message_passing(v_i_prime, G=G)
        else:
            v_i_prime = self.message_passing(v_i_in, G=G)
            v_i_e = self.L_v(v_i_prime)
        for i, l in enumerate(self.L_extra):
            if not self.N_extra[i] is None:
                v_i_e = self.N_extra[i](v_i_e)
            v_i_e = F.relu(l(v_i_e))

        v_i_e = F.relu(v_i_e)            
        if self.N_end is not None:
            v_i_e = self.N_end(v_i_e)
        return v_i_e

    def message_passing(self, v_i_in, G=None):
        """
        Completes message passing.

        :param v_i_in: Features tensors for nodes in graph.
        :param G: Adjacency matrix of graph.
        """
        if not self.reduce == "nodes_only":
            n = torch.sum(G, 2)
            e = torch.eye(self.num_vertices)
            if G.is_cuda:
                n = n.cuda()
                e = e.cuda()
            G = G + torch.einsum('ij,jk->ijk', [n, e])
        out = torch.einsum('ijk,ikl->ijl', [G, v_i_in])
        if self.reduce == 'mean':
            diag = torch.diagonal(G, dim1=-2, dim2=-1)
            add_ones = torch.zeros_like(diag)
            add_ones[diag == 0] = 1
            diag = diag + add_ones  # Avoid divide by zeros
            out = out / diag.unsqueeze(2).repeat(1, 1, self.hidden_size)
        return out


class ShallowFullyConnected(torch.nn.Module):
    """
    A 2-layer MLP for creating prediction outputs.
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 end_norm=None):
        """
        :param input_dim: Feature size of expected input tensors.
        :param hidden_dim: Hidden dimension of intermediate layer.
        :param output_dim: Output size of MLP.
        :param end_norm: Normalization function to apply to final output, if any.
        """
        super(ShallowFullyConnected, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.end_norm = end_norm
        self.layer_norm = torch.nn.LayerNorm(self.input_dim)
        self.layer1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.layer2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, v_i_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param v_i_in: input feature tensor.
        :return: Predictions.
        """
        preds = self.layer_norm(v_i_in)
        preds = self.layer1(preds)
        preds = F.relu(preds)
        preds = self.layer2(preds)
        if self.end_norm is not None:
            preds = self.end_norm(preds)
        return preds