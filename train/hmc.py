from topomodelx.utils.sparse import from_sparse
import torch
import torch.nn.functional as F
from train.train_utils import DEVICE, ONE_OUT_0_ENCODING_SIZE, ONE_OUT_1_ENCODING_SIZE, WEIGHT_DTYPE

class SimplifiedHMCLayer(torch.nn.Module):
    """Simplified version of the HMC layer that maintains hierarchical message passing
    between different dimensional cells (0D, 1D, 2D) without complex attention mechanisms."""
    
    def __init__(self, in_channels_0, in_channels_1, in_channels_2):
        super().__init__()
        # Level 1 transforms
        self.level1_0to0 = torch.nn.Linear(in_channels_0, in_channels_0)  # vertex to vertex
        self.level1_1to0 = torch.nn.Linear(in_channels_1, in_channels_0)  # edge to vertex
        self.level1_1to1 = torch.nn.Linear(in_channels_1, in_channels_1)  # edge to edge
        self.level1_2to1 = torch.nn.Linear(in_channels_2, in_channels_1)  # face to edge
        
        # Level 2 transforms
        self.level2_0to0 = torch.nn.Linear(in_channels_0, in_channels_0)  # vertex to vertex
        self.level2_0to1 = torch.nn.Linear(in_channels_0, in_channels_1)  # vertex to edge
        self.level2_1to1 = torch.nn.Linear(in_channels_1, in_channels_1)  # edge to edge
        self.level2_1to2 = torch.nn.Linear(in_channels_1, in_channels_2)  # edge to face
        self.level2_2to2 = torch.nn.Linear(in_channels_2, in_channels_2)  # face to face

    def forward(self, x_0, x_1, x_2, adjacency_0, incidence_2_t):
        # Level 1: First message passing step
        # Update 0-cells (vertices)
        x_0_level1 = F.relu(
            adjacency_0 @ self.level1_0to0(x_0) +  # Changed order here
            incidence_2_t.T @ self.level1_1to0(x_1)  # Changed order here
        )
        
        # Update 1-cells (edges)
        x_1_level1 = F.relu(
            self.level1_1to1(x_1) + 
            incidence_2_t @ self.level1_2to1(x_2)  # Changed order here
        )
        
        # Update 2-cells (faces)
        x_2_level1 = x_2
        
        # Level 2: Second message passing step
        # Update 0-cells (vertices)
        x_0_out = F.relu(adjacency_0 @ self.level2_0to0(x_0_level1))  # Changed order
        
        # Update 1-cells (edges)
        x_1_out = F.relu(
            incidence_2_t.T @ self.level2_0to1(x_0_level1) +  # Changed order
            self.level2_1to1(x_1_level1)
        )
        
        # Update 2-cells (faces)
        x_2_out = F.relu(
            incidence_2_t @ self.level2_1to2(x_1_level1) +  # Changed order
            self.level2_2to2(x_2_level1)
        )
        
        return x_0_out, x_1_out, x_2_out

class HMCModel(torch.nn.Module):
    """Simplified Hierarchical Message-passing Classifier Model that follows CCXN structure."""
    
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        n_layers=2,
    ):
        super().__init__()
        # Input projections
        self.lin_0_input = torch.nn.Linear(ONE_OUT_0_ENCODING_SIZE, in_channels_0)
        self.lin_1_input = torch.nn.Linear(ONE_OUT_1_ENCODING_SIZE, in_channels_1)
        
        # HMC layers
        self.layers = torch.nn.ModuleList([
            SimplifiedHMCLayer(in_channels_0, in_channels_1, in_channels_2)
            for _ in range(n_layers)
        ])
        
        # Output projections
        self.lin_0 = torch.nn.Linear(in_channels_0, 1)
        self.lin_1 = torch.nn.Linear(in_channels_1, 1)
        self.lin_2 = torch.nn.Linear(in_channels_2, 1)
        
    def forward(self, graph):
      
        # Get initial features and matrices
        x_0, x_1 = graph.x_0, graph.x_1
        adjacency_0 = graph.graph_matrices["adjacency_0"]
        incidence_2_t = graph.graph_matrices["incidence_2_t"]
        
        print("\nInitial shapes:")
        print(f"x_0: {x_0.shape}")
        print(f"x_1: {x_1.shape}")
        print(f"adjacency_0: {adjacency_0.shape}")
        print(f"incidence_2_t: {incidence_2_t.shape}")
        
        # Initial projections
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        x_2 = torch.zeros((incidence_2_t.size(0), x_0.size(-1)), device=x_0.device)
        
        print("\nAfter initial projections:")
        print(f"x_0: {x_0.shape}")
        print(f"x_1: {x_1.shape}")
        print(f"x_2: {x_2.shape}")
        # Apply HMC layers
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, adjacency_0, incidence_2_t)
        
        # Final projections
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)
        
        # Global pooling with nanmean (following CCXN structure)
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        
        return two_dimensional_cells_mean + one_dimensional_cells_mean + zero_dimensional_cells_mean
    
    @staticmethod
    def add_graph_matrices(enhanced_graph):
        """Store required matrices for the model, following CCXN structure."""
        incidence_2_t = enhanced_graph.cell_complex.incidence_matrix(rank=2).T
        adjacency_0 = enhanced_graph.cell_complex.adjacency_matrix(rank=0)
        
        incidence_2_t = from_sparse(incidence_2_t).to(WEIGHT_DTYPE).to(DEVICE)
        adjacency_0 = from_sparse(adjacency_0).to(WEIGHT_DTYPE).to(DEVICE)
        
        enhanced_graph.graph_matrices["incidence_2_t"] = incidence_2_t
        enhanced_graph.graph_matrices["adjacency_0"] = adjacency_0
