from topomodelx.nn.combinatorial.hmc import HMC
from train.train_utils import (
    DEVICE,
    WEIGHT_DTYPE,
    ONE_HOT_0_ENCODING_SIZE,
    ONE_HOT_1_ENCODING_SIZE,
    ONE_HOT_2_ENCODING_SIZE,
    generate_x_0,
    generate_x_1_combinatorial,
    generate_x_2_combinatorial,
)
import torch
import torch.nn.functional as F

HIDDEN_DIMENSIONS = 30


class HMCModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dimensions,
        n_layers=2,
    ):
        super().__init__()

        # Input linear layers
        self.lin_0_input = torch.nn.Linear(ONE_HOT_0_ENCODING_SIZE, HIDDEN_DIMENSIONS)
        self.lin_1_input = torch.nn.Linear(ONE_HOT_1_ENCODING_SIZE, HIDDEN_DIMENSIONS)
        self.lin_2_input = torch.nn.Linear(ONE_HOT_2_ENCODING_SIZE, HIDDEN_DIMENSIONS)

        channels_per_layer = [[(HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS)] * 3]
        self.base_model = HMC(channels_per_layer)

        # Output linear layers
        self.lin_0 = torch.nn.Linear(HIDDEN_DIMENSIONS, 1)
        self.lin_1 = torch.nn.Linear(HIDDEN_DIMENSIONS, 1)
        self.lin_2 = torch.nn.Linear(HIDDEN_DIMENSIONS, 1)

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"]
        x_1 = graph.graph_matrices["x_1"]
        x_2 = graph.graph_matrices["x_2"]
        adjacency_0 = graph.graph_matrices["adjacency_0"]
        adjacency_1 = graph.graph_matrices["adjacency_1"]
        coadjacency_2 = graph.graph_matrices["coadjacency_2"]
        incidence_1 = graph.graph_matrices["incidence_1"]
        incidence_2 = graph.graph_matrices["incidence_2"]

        # Initial linear transformations
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        x_2 = self.lin_2_input(x_2)

        x_0, x_1, x_2 = self.base_model(x_0, x_1, x_2, adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2)

        # Final linear transformations
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)

        # Calculate means and handle NaN values
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)

        return zero_dimensional_cells_mean + one_dimensional_cells_mean + two_dimensional_cells_mean

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        cc = enhanced_graph.data.combinatorial_complex
        x_0 = generate_x_0(cc).to(DEVICE)
        x_1 = generate_x_1_combinatorial(cc).to(DEVICE)
        x_2 = generate_x_2_combinatorial(cc).to(DEVICE)

        adjacency_0 = torch.from_numpy(cc.adjacency_matrix(0, 1).todense()).to_sparse().to(DEVICE, dtype=WEIGHT_DTYPE)
        adjacency_1 = torch.from_numpy(cc.adjacency_matrix(1, 2).todense()).to_sparse().to(DEVICE, dtype=WEIGHT_DTYPE)

        B = cc.incidence_matrix(rank=1, to_rank=2)
        coadjacency_2 = B.T @ B
        coadjacency_2.setdiag(0)
        coadjacency_2 = torch.from_numpy(coadjacency_2.todense()).to_sparse().to(DEVICE, dtype=WEIGHT_DTYPE)

        incidence_1 = torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to_sparse().to(DEVICE, dtype=WEIGHT_DTYPE)
        incidence_2 = torch.from_numpy(cc.incidence_matrix(1, 2).todense()).to_sparse().to(DEVICE, dtype=WEIGHT_DTYPE)

        enhanced_graph.graph_matrices["x_0"] = x_0
        enhanced_graph.graph_matrices["x_1"] = x_1
        enhanced_graph.graph_matrices["x_2"] = x_2
        enhanced_graph.graph_matrices["adjacency_0"] = adjacency_0
        enhanced_graph.graph_matrices["adjacency_1"] = adjacency_1
        enhanced_graph.graph_matrices["coadjacency_2"] = coadjacency_2
        enhanced_graph.graph_matrices["incidence_1"] = incidence_1
        enhanced_graph.graph_matrices["incidence_2"] = incidence_2
