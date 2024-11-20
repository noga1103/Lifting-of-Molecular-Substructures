from dataclasses import dataclass
import torch
import numpy as np
from topomodelx.nn.hypergraph.hnhn import HNHN
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
class HNHNModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dimensions,
        n_layers=2,
    ):
        super().__init__()

        # Create dummy incidence matrix on CPU first
        dummy_incidence = torch.sparse_coo_tensor(
            indices=torch.zeros((2, 1), dtype=torch.long),
            values=torch.zeros(1),
            size=(ONE_HOT_0_ENCODING_SIZE, 1)
        )

        self.base_model = HNHN(
            in_channels=ONE_HOT_0_ENCODING_SIZE,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers,
            incidence_1=dummy_incidence
        )

        self.linear = torch.nn.Linear(hidden_dimensions, 1)
        self.out_pool = True
        # Move entire model to device after initialization
        self.to(DEVICE)

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"]
        cc = graph.data.combinatorial_complex
        
        # Convert numpy matrix to torch tensor on CPU first
        incidence_dense = torch.from_numpy(cc.incidence_matrix(0, 1).todense())
        # Move to device and convert dtype
        incidence_dense = incidence_dense.to(DEVICE).to(WEIGHT_DTYPE)
        
        indices = torch.nonzero(incidence_dense, as_tuple=True)
        values = incidence_dense[indices]
        incidence_1 = torch.sparse_coo_tensor(
            indices=torch.stack(indices),
            values=values,
            size=incidence_dense.size()
        ).to(DEVICE)

        # Ensure all normalization matrices in HNHNLayer are on the correct device
        for layer in self.base_model.layers:
            # Ensure incidence matrices are on the correct device
            layer.incidence_1 = incidence_1
            layer.incidence_1_transpose = incidence_1.transpose(1, 0)
            
            if layer.use_normalized_incidence:
                layer.n_nodes, layer.n_edges = incidence_1.shape
                # Ensure normalization matrices are computed and moved to the correct device
                layer.compute_normalization_matrices()

                # Move normalization matrices to DEVICE after computation
                layer.D0_left_alpha_inverse = layer.D0_left_alpha_inverse.to(DEVICE)
                layer.D1_left_beta_inverse = layer.D1_left_beta_inverse.to(DEVICE)
                layer.D1_right_alpha = layer.D1_right_alpha.to(DEVICE)
                layer.D0_right_beta = layer.D0_right_beta.to(DEVICE)
                
                layer.normalize_incidence_matrices()

        # Process the graph with the updated incidence matrix
        x_0_processed, _ = self.base_model(x_0, incidence_1=incidence_1)

        # Pooling the output if needed
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
