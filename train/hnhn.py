from typing import Literal

import torch
from torch.nn.parameter import Parameter

from topomodelx.base.conv import Conv
from dataclasses import dataclass
import torch
import numpy as np
 
from train.train_utils import (
    DEVICE,
    WEIGHT_DTYPE,
    ONE_HOT_0_ENCODING_SIZE,
    ONE_HOT_1_ENCODING_SIZE,
    ONE_HOT_2_ENCODING_SIZE,
    generate_x_0,
    generate_x_1_combinatorial,
    generate_x_2_combinatorial,
    EnhancedGraph
)
class ModifiedHNHNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        incidence_1=None,
        use_bias=True,
        use_normalized_incidence=True,
        alpha=-1.5,
        beta=-0.5,
        bias_gain=1.414,
        **kwargs,
    ):
        super().__init__()
        self.use_bias = use_bias
        self.use_normalized_incidence = use_normalized_incidence
        self.alpha = alpha
        self.beta = beta
        
        self.conv_0_to_1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            aggr_norm=False,
            update_func=None,
        )

        self.conv_1_to_0 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            aggr_norm=False,
            update_func=None,
        )

        if self.use_bias:
            self.bias_1_to_0 = Parameter(torch.zeros(1, hidden_channels))
            self.bias_0_to_1 = Parameter(torch.zeros(1, hidden_channels))

    def compute_normalization_factors(self, incidence_1):
        """Compute normalization factors without storing matrices"""
        B1 = incidence_1.to_dense()
        edge_sum = B1.sum(0)
        node_sum = B1.sum(1)
        
        edge_norm = edge_sum ** self.alpha
        node_norm = node_sum ** self.beta
        
        return edge_norm, node_norm

    def normalize_incidence(self, incidence_1):
        """Normalize incidence matrix directly"""
        edge_norm, node_norm = self.compute_normalization_factors(incidence_1)
        
        B1 = incidence_1.to_dense()
        normalized = B1.clone()
        
        # Apply normalization
        for i in range(B1.shape[0]):
            for j in range(B1.shape[1]):
                if B1[i,j] != 0:
                    edge_factor = 1.0 / (edge_norm[B1[i,:].bool()].sum())
                    node_factor = 1.0 / (node_norm[B1[:,j].bool()].sum())
                    normalized[i,j] *= edge_factor * edge_norm[j] * node_factor * node_norm[i]
        
        # Convert back to sparse
        indices = torch.nonzero(normalized, as_tuple=True)
        values = normalized[indices]
        return torch.sparse_coo_tensor(
            indices=torch.stack(indices),
            values=values,
            size=normalized.size(),
            device=normalized.device
        )

    def forward(self, x_0, incidence_1):
        device = x_0.device
        incidence_1 = incidence_1.to(device)
        
        if self.use_normalized_incidence:
            incidence_1 = self.normalize_incidence(incidence_1)
        
        incidence_1_transpose = incidence_1.transpose(1, 0)
        
        # Compute output features
        x_1 = self.conv_0_to_1(x_0, incidence_1_transpose)
        if self.use_bias:
            x_1 += self.bias_0_to_1.to(device)
            
        x_0 = self.conv_1_to_0(x_1, incidence_1)
        if self.use_bias:
            x_0 += self.bias_1_to_0.to(device)
            
        return torch.relu(x_0), torch.relu(x_1)

class HNHNModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dimensions,
        n_layers=2,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            ModifiedHNHNLayer(
                in_channels=ONE_HOT_0_ENCODING_SIZE if i == 0 else hidden_dimensions,
                hidden_channels=hidden_dimensions,
                use_normalized_incidence=True
            )
            for i in range(n_layers)
        ])
        
        self.linear = torch.nn.Linear(hidden_dimensions, 1)
        self.out_pool = True
        self.to(DEVICE)

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"]
        cc = graph.data.combinatorial_complex
        
        incidence_dense = torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to(DEVICE).to(WEIGHT_DTYPE)
        indices = torch.nonzero(incidence_dense, as_tuple=True)
        values = incidence_dense[indices]
        incidence_1 = torch.sparse_coo_tensor(
            indices=torch.stack(indices),
            values=values,
            size=incidence_dense.size(),
            device=DEVICE
        )

        x_0_processed = x_0
        for layer in self.layers:
            x_0_processed, _ = layer(x_0_processed, incidence_1)

        x = torch.max(x_0_processed, dim=0)[0] if self.out_pool else x_0_processed
        return self.linear(x)

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        cc = enhanced_graph.data.combinatorial_complex
        x_0 = generate_x_0(cc).to(DEVICE)
        x_1 = generate_x_1_combinatorial(cc).to(DEVICE)
        x_2 = generate_x_2_combinatorial(cc).to(DEVICE)
        enhanced_graph.graph_matrices = {
            "x_0": x_0,
            "x_1": x_1,
            "x_2": x_2,
        }
