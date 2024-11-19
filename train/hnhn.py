import torch
import numpy as np
from torch_geometric.utils import to_undirected
from topomodelx.nn.hypergraph.hnhn import HNHN

class HNHNModel(torch.nn.Module):
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        hidden_channels,
        n_layers=2,
    ):
        super().__init__()
        
        # Input linear layers for different dimensional features
        self.lin_0_input = torch.nn.Linear(in_channels_0, hidden_channels)
        self.lin_1_input = torch.nn.Linear(in_channels_1, hidden_channels)
        self.lin_2_input = torch.nn.Linear(in_channels_2, hidden_channels)
        
        # HNHN base model
        self.base_model = HNHN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers
        )
        
        # Output linear layers
        self.lin_0_output = torch.nn.Linear(hidden_channels, 1)
        self.lin_1_output = torch.nn.Linear(hidden_channels, 1)
        self.lin_2_output = torch.nn.Linear(hidden_channels, 1)
        
    def create_hyperedge_structure(self, adjacency_0, adjacency_1, incidence_1, incidence_2):
        """Create hyperedge structure from combinatorial complex matrices."""
        num_nodes = adjacency_0.size(0)
        
        # Convert sparse matrices to dense for processing
        adj_0_dense = adjacency_0.to_dense()
        inc_1_dense = incidence_1.to_dense()
        
        # Create hyperedges using both 1-hop neighborhoods and higher-order relationships
        hyperedges = []
        unique_hyperedges = set()
        
        # Add 1-hop neighborhoods
        for node in range(num_nodes):
            neighbors = (adj_0_dense[node] > 0).nonzero().view(-1)
            if len(neighbors) > 0:
                he = tuple(sorted(neighbors.cpu().numpy()))
                if he not in unique_hyperedges:
                    hyperedges.append(list(he))
                    unique_hyperedges.add(he)
        
        # Add relationships from incidence matrices
        for col in range(inc_1_dense.size(1)):
            nodes = (inc_1_dense[:, col] > 0).nonzero().view(-1)
            if len(nodes) > 0:
                he = tuple(sorted(nodes.cpu().numpy()))
                if he not in unique_hyperedges:
                    hyperedges.append(list(he))
                    unique_hyperedges.add(he)
        
        # Create new incidence matrix
        incidence = torch.zeros((num_nodes, len(hyperedges)), device=adjacency_0.device)
        for idx, he in enumerate(hyperedges):
            incidence[he, idx] = 1
            
        return incidence.to_sparse()
    
    def forward(self, graph):
        # Extract features and matrices
        x_0 = graph.graph_matrices["x_0"]
        x_1 = graph.graph_matrices["x_1"]
        x_2 = graph.graph_matrices["x_2"]
        adjacency_0 = graph.graph_matrices["adjacency_0"]
        adjacency_1 = graph.graph_matrices["adjacency_1"]
        incidence_1 = graph.graph_matrices["incidence_1"]
        incidence_2 = graph.graph_matrices["incidence_2"]
        
        # Initial feature transformations
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        x_2 = self.lin_2_input(x_2)
        
        # Create hyperedge structure
        hyperedge_incidence = self.create_hyperedge_structure(
            adjacency_0, adjacency_1, incidence_1, incidence_2
        )
        
        # Process through HNHN
        x_0_processed, _ = self.base_model(x_0, incidence_1=hyperedge_incidence)
        
        # Final transformations
        out_0 = self.lin_0_output(x_0_processed)
        out_1 = self.lin_1_output(x_1)
        out_2 = self.lin_2_output(x_2)
        
        # Calculate means and handle NaN values
        mean_0 = torch.nanmean(out_0, dim=0)
        mean_0[torch.isnan(mean_0)] = 0
        mean_1 = torch.nanmean(out_1, dim=0)
        mean_1[torch.isnan(mean_1)] = 0
        mean_2 = torch.nanmean(out_2, dim=0)
        mean_2[torch.isnan(mean_2)] = 0
        
        return mean_0 + mean_1 + mean_2
    
    @staticmethod
    def add_graph_matrices(enhanced_graph):
        """Add required matrices to the graph object."""
        cc = enhanced_graph.data.combinatorial_complex
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate features
        x_0 = generate_x_0(cc).to(device)
        x_1 = generate_x_1_combinatorial(cc).to(device)
        x_2 = generate_x_2_combinatorial(cc).to(device)
        
        # Generate matrices
        adjacency_0 = torch.from_numpy(cc.adjacency_matrix(0, 1).todense()).to_sparse().to(device)
        adjacency_1 = torch.from_numpy(cc.adjacency_matrix(1, 2).todense()).to_sparse().to(device)
        
        B = cc.incidence_matrix(rank=1, to_rank=2)
        coadjacency_2 = B.T @ B
        coadjacency_2.setdiag(0)
        coadjacency_2 = torch.from_numpy(coadjacency_2.todense()).to_sparse().to(device)
        
        incidence_1 = torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to_sparse().to(device)
        incidence_2 = torch.from_numpy(cc.incidence_matrix(1, 2).todense()).to_sparse().to(device)
        
        # Store matrices
        enhanced_graph.graph_matrices = {
            "x_0": x_0,
            "x_1": x_1,
            "x_2": x_2,
            "adjacency_0": adjacency_0,
            "adjacency_1": adjacency_1,
            "coadjacency_2": coadjacency_2,
            "incidence_1": incidence_1,
            "incidence_2": incidence_2
        }
