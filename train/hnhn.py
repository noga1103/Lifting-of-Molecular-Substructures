from dataclasses import dataclass
import torch
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

        # Define the model
        self.base_model = HNHN(
            in_channels=ONE_HOT_0_ENCODING_SIZE,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers,
        )

        # Readout
        self.linear = torch.nn.Linear(hidden_dimensions, 1)
        self.out_pool = True  # For graph-level tasks

    def forward(self, graph):
        # Get features and incidence matrix
        x_0 = graph.graph_matrices["x_0"]
        cc = graph.data.combinatorial_complex
        hg = cc.to_hypergraph()
        incidence_1 = hg.incidence_matrix()
        incidence_1 = torch.from_numpy(incidence_1.todense()).to(DEVICE).to(WEIGHT_DTYPE)
        
        # Base model
        x_0_processed, _ = self.base_model(x_0, incidence_1=incidence_1)

        # Pool over all nodes in the hypergraph
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
