from topomodelx.nn.cell.cwn_layer import CWNLayer
from topomodelx.nn.cell.cwn import CWN
from topomodelx.utils.sparse import from_sparse
from train.train_utils import DEVICE, ONE_HOT_0_ENCODING_SIZE, ONE_HOT_1_ENCODING_SIZE, ONE_HOT_2_ENCODING_SIZE, WEIGHT_DTYPE, generate_x_0, generate_x_1, generate_x_2
import torch
import torch.nn.functional as F


class CWNModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dimension,
        n_layers=2,
    ):
        super().__init__()

        # Input linear layers
        self.lin_0_input = torch.nn.Linear(ONE_HOT_0_ENCODING_SIZE, hidden_dimension)
        self.lin_1_input = torch.nn.Linear(ONE_HOT_1_ENCODING_SIZE, hidden_dimension)
        self.lin_2_input = torch.nn.Linear(ONE_HOT_2_ENCODING_SIZE, hidden_dimension)

        self.base_model = CWN(hidden_dimension, hidden_dimension, hidden_dimension, hidden_dimension, n_layers=n_layers)

        # Output linear layers
        self.lin_0 = torch.nn.Linear(hidden_dimension, 1)
        self.lin_1 = torch.nn.Linear(hidden_dimension, 1)
        self.lin_2 = torch.nn.Linear(hidden_dimension, 1)

    def forward(self, graph):
        x_0, x_1, x_2 = graph.graph_matrices["x_0"], graph.graph_matrices["x_1"], graph.graph_matrices["x_2"]
        adjacency_1, incidence_2, incidence_1_t = (graph.graph_matrices["adjacency_1"], graph.graph_matrices["incidence_2"], graph.graph_matrices["incidence_1_t"])

        # Initial linear transformations
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        x_2 = self.lin_2_input(x_2)

        # Process through CWN layers
        x_0, x_1, x_2 = self.base_model(x_0, x_1, x_2, adjacency_1, incidence_2, incidence_1_t)

        # Final linear transformations
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)

        # Calculate means and handle NaN values
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0

        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0

        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0

        return two_dimensional_cells_mean + one_dimensional_cells_mean + zero_dimensional_cells_mean

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        x_0 = generate_x_0(enhanced_graph.data.cell_complex).to(DEVICE)
        x_1 = generate_x_1(enhanced_graph.data.cell_complex).to(DEVICE)
        x_2 = generate_x_2(enhanced_graph.data.cell_complex).to(DEVICE)

        incidence_2 = enhanced_graph.data.cell_complex.incidence_matrix(rank=2)
        adjacency_1 = enhanced_graph.data.cell_complex.adjacency_matrix(rank=1)
        incidence_1_t = enhanced_graph.data.cell_complex.incidence_matrix(rank=1).T

        incidence_2 = from_sparse(incidence_2).to(WEIGHT_DTYPE).to(DEVICE)
        adjacency_1 = from_sparse(adjacency_1).to(WEIGHT_DTYPE).to(DEVICE)
        incidence_1_t = from_sparse(incidence_1_t).to(WEIGHT_DTYPE).to(DEVICE)

        enhanced_graph.graph_matrices["x_0"] = x_0
        enhanced_graph.graph_matrices["x_1"] = x_1
        enhanced_graph.graph_matrices["x_2"] = x_2
        enhanced_graph.graph_matrices["incidence_2"] = incidence_2
        enhanced_graph.graph_matrices["adjacency_1"] = adjacency_1
        enhanced_graph.graph_matrices["incidence_1_t"] = incidence_1_t
