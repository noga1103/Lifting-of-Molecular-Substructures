from dataclasses import dataclass
import torch
import numpy as np
from torch_geometric.utils import to_undirected
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
        # Ensure it's on the correct device
        dummy_size = 10  # Small size for initialization
        dummy_incidence = torch.zeros((dummy_size, dummy_size), device=DEVICE, dtype=WEIGHT_DTYPE)
        # Add some values to ensure it's not empty
        dummy_incidence[0, 0] = 1
        dummy_incidence[1, 1] = 1
        dummy_incidence = dummy_incidence.to_sparse_coo()
        
        # Define the model
        self.base_model = HNHN(
            in_channels=ONE_HOT_0_ENCODING_SIZE,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers,
            incidence_1=dummy_incidence
        ).to(DEVICE)

        # Readout
        self.linear = torch.nn.Linear(hidden_dimensions, 1).to(DEVICE)
        self.out_pool = True

    def create_hypergraph_incidence(self, graph):
        """Create hypergraph incidence matrix from graph."""
        # Get edge index from graph
        edge_index = graph.data.combinatorial_complex.incidence_matrix(0, 1).todense()
        edge_index = torch.from_numpy(edge_index).nonzero().t().to(DEVICE)
        edge_index = to_undirected(edge_index)
        
        # Create one-hop neighborhoods
        num_nodes = graph.graph_matrices["x_0"].shape[0]
        one_hop_neighborhoods = []
        for node in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == node]
            one_hop_neighborhoods.append(neighbors.cpu().numpy())
        
        # Create unique hyperedges
        unique_hyperedges = set()
        hyperedges = []
        for neighborhood in one_hop_neighborhoods:
            neighborhood = tuple(sorted(neighborhood))
            if neighborhood not in unique_hyperedges and len(neighborhood) > 0:
                hyperedges.append(list(neighborhood))
                unique_hyperedges.add(neighborhood)
        
        if not hyperedges:
            # If no valid hyperedges, create a minimal valid hypergraph
            hyperedges = [[0, 1]]
        
        # Create incidence matrix
        incidence = np.zeros((num_nodes, len(hyperedges)))
        for col, neighborhood in enumerate(hyperedges):
            for row in neighborhood:
                incidence[row, col] = 1
        
        # Convert to tensor and verify
        incidence = torch.from_numpy(incidence).to(DEVICE).to(WEIGHT_DTYPE)
        assert torch.all(incidence.sum(0) > 0), "Some hyperedges are empty"
        assert torch.all(incidence.sum(1) > 0), "Some nodes are not in any hyperedges"
        
        return incidence.to_sparse_coo()

    def forward(self, graph):
        # Get node features
        x_0 = graph.graph_matrices["x_0"].to(DEVICE)
        
        # Create hypergraph incidence matrix
        incidence_1 = self.create_hypergraph_incidence(graph)
        
        # Update base model's incidence matrix
        self.base_model.incidence_1 = incidence_1
        
        # Process through HNHN
        x_0_processed, _ = self.base_model(x_0)
        
        # Pool over all nodes in the hypergraph
        x = torch.max(x_0_processed, dim=0)[0] if self.out_pool else x_0_processed
        
        return self.linear(x)

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        """Add required matrices to the graph object."""
        cc = enhanced_graph.data.combinatorial_complex
        
        # Generate features and ensure they're on the correct device
        x_0 = generate_x_0(cc).to(DEVICE)
        x_1 = generate_x_1_combinatorial(cc).to(DEVICE)
        x_2 = generate_x_2_combinatorial(cc).to(DEVICE)
        
        # Store matrices
        enhanced_graph.graph_matrices = {
            "x_0": x_0,
            "x_1": x_1,
            "x_2": x_2,
        }
