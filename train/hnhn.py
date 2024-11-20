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
    def __init__(self, hidden_dimensions, n_layers=2):
        super().__init__()
        
        # Initialize on CPU first
        indices = torch.zeros((2, 1), dtype=torch.long)
        values = torch.zeros(1)
        dummy_incidence = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(ONE_HOT_0_ENCODING_SIZE, 1)
        )
        
        # Create base model on CPU
        self.base_model = HNHN(
            in_channels=ONE_HOT_0_ENCODING_SIZE,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers,
            incidence_1=dummy_incidence
        )
        
        self.linear = torch.nn.Linear(hidden_dimensions, 1)
        self.out_pool = True

        # Move to device after full initialization
        self.to(DEVICE)

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"].to(DEVICE)
        cc = graph.data.combinatorial_complex
        
        incidence_1 = cc.incidence_matrix(0, 1)
        incidence_1 = torch.from_numpy(incidence_1.todense()).to(DEVICE, dtype=WEIGHT_DTYPE)
        
        indices = torch.nonzero(incidence_1).t()
        values = incidence_1[indices[0], indices[1]]
        incidence_1 = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=incidence_1.size()
        ).to(DEVICE)
        
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

