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
        
        # Create a dummy incidence matrix for initialization
        # This will be replaced during forward pass
        dummy_incidence = torch.sparse_coo_tensor(
            indices=torch.zeros((2, 1), dtype=torch.long),
            values=torch.zeros(1),
            size=(ONE_HOT_0_ENCODING_SIZE, 1)
        )
        
        # Define the model with dummy incidence matrix
        self.base_model = HNHN(
            in_channels=ONE_HOT_0_ENCODING_SIZE,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers,
            incidence_1=dummy_incidence
        )
        
        # Readout
        self.linear = torch.nn.Linear(hidden_dimensions, 1)
        self.out_pool = True  # For graph-level tasks

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"]
        cc = graph.data.combinatorial_complex
        
        # Build hypergraph edges from CC
        edges = []
        for edge_idx in range(cc.boundary_1.shape[1]):
            nodes = cc.boundary_1[:, edge_idx].nonzero()[0].tolist()
            if nodes:  # Only add non-empty edges
                edges.append(nodes)
                
        # Create incidence matrix
        num_nodes = cc.boundary_1.shape[0]
        incidence_1 = np.zeros((num_nodes, len(edges)))
        for edge_idx, nodes in enumerate(edges):
            incidence_1[nodes, edge_idx] = 1
            
        incidence_1 = torch.from_numpy(incidence_1).to(DEVICE).to(WEIGHT_DTYPE)
        
        # Convert to sparse format
        if not incidence_1.is_sparse:
            indices = torch.nonzero(incidence_1).t()
            values = incidence_1[indices[0], indices[1]]
            incidence_1 = torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=incidence_1.size()
            )
        
        self.base_model.incidence_1 = incidence_1
        x_0_processed, _ = self.base_model(x_0, incidence_1=incidence_1)
        
        x = torch.max(x_0_processed, dim=0)[0] if self.out_pool else x_0_processed
        return self.linear(x)

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        """Add required matrices to the graph object."""
        cc = enhanced_graph.data.combinatorial_complex
        
        # Generate features
        x_0 = generate_x_0(cc).to(DEVICE)
        x_1 = generate_x_1_combinatorial(cc).to(DEVICE)
        x_2 = generate_x_2_combinatorial(cc).to(DEVICE)
        
        # Store matrices
        enhanced_graph.graph_matrices = {
            "x_0": x_0,
            "x_1": x_1,
            "x_2": x_2,
        }
