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
    @staticmethod
    def convert_to_hypergraph(cc):
        vertices = list(cc.cells[0].keys())
        
        incidence_1 = cc.incidence_matrix(0, 1)
        
        incidence_dense = incidence_1.todense()
        
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        
        rows, cols = np.nonzero(incidence_dense)
        
        hyperedges = []
        unique_cols = np.unique(cols)
        for col in unique_cols:
            # Get all vertices connected in this hyperedge
            connected_vertices = rows[cols == col].tolist()
            if len(connected_vertices) > 1:  # Only add if at least 2 vertices
                hyperedges.append(connected_vertices)
        
        # Create new incidence matrix with correct dimensions
        n_vertices = len(vertices)
        n_hyperedges = len(hyperedges)
        new_rows = []
        new_cols = []
        data = []
        
        for i, he in enumerate(hyperedges):
            for v in he:
                new_rows.append(v)
                new_cols.append(i)
                data.append(1)
        
        # Create the sparse matrix with explicit dimensions
        incidence_matrix = csr_matrix(
            (data, (new_rows, new_cols)),
            shape=(n_vertices, n_hyperedges)
        )
        
        # Add debug print statements
        print(f"Number of vertices: {n_vertices}")
        print(f"Number of hyperedges: {n_hyperedges}")
        print(f"Max row index: {max(new_rows) if new_rows else -1}")
        print(f"Max col index: {max(new_cols) if new_cols else -1}")
        
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
