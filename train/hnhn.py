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
        
        # Input linear layers for different dimensional features
        self.lin_0_input = torch.nn.Linear(ONE_HOT_0_ENCODING_SIZE, hidden_dimensions)
        self.lin_1_input = torch.nn.Linear(ONE_HOT_1_ENCODING_SIZE, hidden_dimensions)
        self.lin_2_input = torch.nn.Linear(ONE_HOT_2_ENCODING_SIZE, hidden_dimensions)
        
        # HNHN base model
        self.base_model = HNHN(
            in_channels=hidden_dimensions,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers
        )
        
        # Output linear layers
        self.lin_0 = torch.nn.Linear(hidden_dimensions, 1)
        self.lin_1 = torch.nn.Linear(hidden_dimensions, 1)
        self.lin_2 = torch.nn.Linear(hidden_dimensions, 1)

    def forward(self, graph: EnhancedGraph) -> torch.Tensor:
        # Extract features 
        x_0 = graph.graph_matrices["x_0"]
        x_1 = graph.graph_matrices["x_1"]
        x_2 = graph.graph_matrices["x_2"]
        
        # Initial feature transformations
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        x_2 = self.lin_2_input(x_2)
        
        # Get hypergraph structure
        cc = graph.data.combinatorial_complex
        hg = cc.to_hypergraph()
        incidence_1 = hg.incidence_matrix()
        incidence_1 = torch.from_numpy(incidence_1.todense()).to(DEVICE).to(WEIGHT_DTYPE)
        
        # Process through HNHN
        x_0_processed, _ = self.base_model(x_0, incidence_1=incidence_1)
        
        # Final transformations
        x_0 = self.lin_0(x_0_processed)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)
        
        # Calculate means and handle NaN values
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0

        return zero_dimensional_cells_mean + one_dimensional_cells_mean + two_dimensional_cells_mean

    @staticmethod
    def add_graph_matrices(enhanced_graph: EnhancedGraph) -> None:
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
