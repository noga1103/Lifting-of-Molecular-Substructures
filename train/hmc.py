from topomodelx.utils.sparse import from_sparse
import torch
import torch.nn.functional as F
from train.train_utils import DEVICE, ONE_OUT_0_ENCODING_SIZE, ONE_OUT_1_ENCODING_SIZE, WEIGHT_DTYPE

class HMCLayer(torch.nn.Module):
    """HMC layer implementation."""
    
    def __init__(self, in_channels_0, in_channels_1, in_channels_2):
        super().__init__()
        # Level 1 transforms
        self.level1_0to0 = torch.nn.Linear(in_channels_0, in_channels_0)
        self.level1_1to0 = torch.nn.Linear(in_channels_1, in_channels_0)
        self.level1_1to1 = torch.nn.Linear(in_channels_1, in_channels_1)
        self.level1_2to1 = torch.nn.Linear(in_channels_2, in_channels_1)
        
        # Level 2 transforms
        self.level2_0to0 = torch.nn.Linear(in_channels_0, in_channels_0)
        self.level2_0to1 = torch.nn.Linear(in_channels_0, in_channels_1)
        self.level2_1to1 = torch.nn.Linear(in_channels_1, in_channels_1)
        self.level2_1to2 = torch.nn.Linear(in_channels_1, in_channels_2)
        self.level2_2to2 = torch.nn.Linear(in_channels_2, in_channels_2)

    def forward(self, x_0, x_1, x_2, adjacency_0, incidence_2_t):
        # Convert sparse matrices to dense if they're not already
        if adjacency_0.is_sparse_csr:
            adjacency_0 = adjacency_0.to_dense()
        if incidence_2_t.is_sparse_csr:
            incidence_2_t = incidence_2_t.to_dense()
            
        # Level 1: First message passing step
        # Messages to 0-cells (vertices)
        x_0_from_0 = self.level1_0to0(x_0)
        x_0_from_1 = self.level1_1to0(x_1)
        
        # Update 0-cells
        x_0_level1 = F.relu(
            x_0_from_0 @ adjacency_0 + 
            x_0_from_1 @ incidence_2_t.T
        )
        
        # Update 1-cells (edges)
        x_1_from_1 = self.level1_1to1(x_1)
        x_1_from_2 = self.level1_2to1(x_2)
        x_1_level1 = F.relu(
            x_1_from_1 + 
            x_1_from_2 @ incidence_2_t
        )
        
        # Update 2-cells (faces)
        x_2_level1 = x_2
        
        # Level 2: Second message passing step
        # Update 0-cells
        x_0_out = F.relu(self.level2_0to0(x_0_level1) @ adjacency_0)
        
        # Update 1-cells
        x_1_out = F.relu(
            self.level2_0to1(x_0_level1) @ incidence_2_t.T +
            self.level2_1to1(x_1_level1)
        )
        
        # Update 2-cells
        x_2_out = F.relu(
            self.level2_1to2(x_1_level1) @ incidence_2_t +
            self.level2_2to2(x_2_level1)
        )
        
        return x_0_out, x_1_out, x_2_out

class HMCModel(torch.nn.Module):
    """Hierarchical Message-passing Classifier Model."""
    
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        n_layers=2,
    ):
        super().__init__()
        self.lin_0_input = torch.nn.Linear(ONE_OUT_0_ENCODING_SIZE, in_channels_0)
        self.lin_1_input = torch.nn.Linear(ONE_OUT_1_ENCODING_SIZE, in_channels_1)
        
        self.layers = torch.nn.ModuleList([
            HMCLayer(in_channels_0, in_channels_1, in_channels_2)
            for _ in range(n_layers)
        ])
        
        self.lin_0 = torch.nn.Linear(in_channels_0, 1)
        self.lin_1 = torch.nn.Linear(in_channels_1, 1)
        self.lin_2 = torch.nn.Linear(in_channels_2, 1)
        
    def forward(self, graph):
        x_0, x_1 = graph.x_0, graph.x_1
        adjacency_0 = graph.graph_matrices["adjacency_0"]
        incidence_2_t = graph.graph_matrices["incidence_2_t"]
        
        # Initial projections
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        x_2 = torch.zeros((incidence_2_t.size(0), x_0.size(-1)), device=x_0.device)
        
        # Apply HMC layers
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, adjacency_0, incidence_2_t)
        
        # Final projections
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)
        
        # Global pooling with nanmean
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        
        return two_dimensional_cells_mean + one_dimensional_cells_mean + zero_dimensional_cells_mean
    
    @staticmethod
    def add_graph_matrices(enhanced_graph):
        incidence_2_t = enhanced_graph.cell_complex.incidence_matrix(rank=2).T
        adjacency_0 = enhanced_graph.cell_complex.adjacency_matrix(rank=0)
        
        incidence_2_t = from_sparse(incidence_2_t).to(WEIGHT_DTYPE).to(DEVICE)
        adjacency_0 = from_sparse(adjacency_0).to(WEIGHT_DTYPE).to(DEVICE)
        
        enhanced_graph.graph_matrices["incidence_2_t"] = incidence_2_t
        enhanced_graph.graph_matrices["adjacency_0"] = adjacency_0
