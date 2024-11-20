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
        
        # Create dummy incidence matrix directly on correct device and dtype
        dummy_incidence = torch.sparse_coo_tensor(
            indices=torch.zeros((2, 1), dtype=torch.long, device=DEVICE),
            values=torch.zeros(1, device=DEVICE, dtype=WEIGHT_DTYPE),
            size=(ONE_HOT_0_ENCODING_SIZE, 1)
        )
        
        self.base_model = HNHN(
            in_channels=ONE_HOT_0_ENCODING_SIZE,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers,
            incidence_1=dummy_incidence
        ).to(DEVICE)
        
        self.linear = torch.nn.Linear(hidden_dimensions, 1).to(DEVICE)
        self.out_pool = True

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"]
        cc = graph.data.combinatorial_complex
        
        # Convert incidence matrix to sparse tensor directly on correct device
        incidence_dense = torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to(DEVICE).to(WEIGHT_DTYPE)
        indices = torch.nonzero(incidence_dense, as_tuple=True)
        values = incidence_dense[indices]
        incidence_1 = torch.sparse_coo_tensor(
            indices=torch.stack(indices),
            values=values,
            size=incidence_dense.size(),
            device=DEVICE
        )
        
        self.base_model.incidence_1 = incidence_1
        x_0_processed, _ = self.base_model(x_0, incidence_1=incidence_1)
        
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
