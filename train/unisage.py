from topomodelx.nn.hypergraph.unisage_layer import UniSAGELayer
from train.train_utils import (
    DEVICE,
    WEIGHT_DTYPE,
    ONE_HOT_0_ENCODING_SIZE,
    ONE_HOT_1_ENCODING_SIZE,
    generate_x_0,
    generate_x_1_combinatorial,
)
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix


class UNISAGEModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dimensions,
        n_layers=2,
        input_drop=0.2,
        layer_drop=0.2,
        e_aggr="sum",
        v_aggr="mean",
        use_norm=False,
    ):
        super().__init__()

        self.lin_0_input = torch.nn.Linear(ONE_HOT_0_ENCODING_SIZE, hidden_dimensions)
        self.lin_1_input = torch.nn.Linear(ONE_HOT_1_ENCODING_SIZE, hidden_dimensions)

        self.base_model = UniSAGELayer(
            in_channels=hidden_dimensions,
            hidden_channels=hidden_dimensions,
            input_drop=input_drop,
            layer_drop=layer_drop,
            n_layers=n_layers,
            e_aggr=e_aggr,
            v_aggr=v_aggr,
            use_norm=use_norm,
        )

        # Output linear layers
        self.lin_0 = torch.nn.Linear(hidden_dimensions, 1)
        self.lin_1 = torch.nn.Linear(hidden_dimensions, 1)

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"]
        x_1 = graph.graph_matrices["x_1"]
        incidence_1 = graph.graph_matrices["incidence_1"]

        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)

        x_0, x_1 = self.base_model(x_0, incidence_1)

        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)

        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0

        return zero_dimensional_cells_mean + one_dimensional_cells_mean

    @staticmethod
    def convert_to_hypergraph(cc):
        vertices = list(cc.cells[0].keys())
        
        edges = list(cc.cells[1].keys()) if 1 in cc.cells else []
        faces = list(cc.cells[2].keys()) if 2 in cc.cells else []
        
        # Create hyperedges from both edges and faces
        hyperedges = []
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        
        # Get incidence matrices
        incidence_1 = cc.incidence_matrix(0, 1)
        incidence_2 = cc.incidence_matrix(0, 2) if 2 in cc.cells else None
        
        # Add regular edges as hyperedges
        for i, edge in enumerate(edges):
            # Get vertices connected by this edge from incidence matrix
            vertex_indices = [v for v, row in enumerate(incidence_1[:,i].toarray()) if row[0] != 0]
            if len(vertex_indices) > 1:  # Only add if hyperedge contains at least 2 vertices
                hyperedges.append(vertex_indices)
        
        # Add faces as hyperedges
        if incidence_2 is not None:
            for i, face in enumerate(faces):
                # Get vertices that form the boundary of the face from incidence matrix
                vertex_indices = [v for v, row in enumerate(incidence_2[:,i].toarray()) if row[0] != 0]
                if len(vertex_indices) > 2:  # Only add if hyperedge contains at least 3 vertices
                    hyperedges.append(vertex_indices)
        
        n_vertices = len(vertices)
        n_hyperedges = len(hyperedges)
        rows = []
        cols = []
        data = []
        
        for i, he in enumerate(hyperedges):
            for v in he:
                rows.append(v)
                cols.append(i)
                data.append(1)
        
        incidence_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_vertices, n_hyperedges)
        )
        
        return vertices, hyperedges, incidence_matrix

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        cc = enhanced_graph.data.combinatorial_complex
        
        # Convert combinatorial complex to hypergraph
        vertices, hyperedges, incidence_matrix = UNISAGEModel.convert_to_hypergraph(cc)
        
        x_0 = generate_x_0(cc).to(DEVICE)
        x_1 = generate_x_1_combinatorial(cc).to(DEVICE)
        
        incidence_1 = torch.from_numpy(incidence_matrix.todense()).to_sparse().to(DEVICE, dtype=WEIGHT_DTYPE)
        
        enhanced_graph.graph_matrices["x_0"] = x_0
        enhanced_graph.graph_matrices["x_1"] = x_1
        enhanced_graph.graph_matrices["incidence_1"] = incidence_1
        
        enhanced_graph.hypergraph_info = {
            "vertices": vertices,
            "hyperedges": hyperedges
        }
