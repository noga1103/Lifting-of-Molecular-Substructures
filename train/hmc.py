from topomodelx.nn.combinatorial.hmc_layer import HMCLayer
from topomodelx.utils.sparse import from_sparse
from train.train_utils import DEVICE, ONE_OUT_0_ENCODING_SIZE, ONE_OUT_1_ENCODING_SIZE, WEIGHT_DTYPE
import torch
import torch.nn.functional as F


class HMCModel(torch.nn.Module):
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        n_layers=2,
    ):
        super().__init__()

        # Store dimensions
        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2

        # Input linear layers
        self.lin_0_input = torch.nn.Linear(ONE_OUT_0_ENCODING_SIZE, self.in_channels_0)
        self.lin_1_input = torch.nn.Linear(ONE_OUT_1_ENCODING_SIZE, self.in_channels_1)

        # Define intermediate and output channels for each layer
        intermediate_channels = [60, 60, 60]
        out_channels = [60, 60, 60]

        # HMC layers
        self.layers = torch.nn.ModuleList(
            [
                HMCLayer(
                    in_channels=[self.in_channels_0, self.in_channels_1, self.in_channels_2],
                    intermediate_channels=intermediate_channels,
                    out_channels=out_channels,
                    negative_slope=0.2,
                    softmax_attention=True,
                    update_func_attention="relu",
                    update_func_aggregation="relu",
                )
                for _ in range(n_layers)
            ]
        )

        # Output linear layers
        self.lin_0 = torch.nn.Linear(out_channels[0], 1)
        self.lin_1 = torch.nn.Linear(out_channels[1], 1)
        self.lin_2 = torch.nn.Linear(out_channels[2], 1)

    def forward(self, graph):
        x_0, x_1 = graph.x_0, graph.x_1
        adjacency_0 = graph.graph_matrices["adjacency_0"]
        adjacency_1 = graph.graph_matrices["adjacency_1"]
        coadjacency_2 = graph.graph_matrices["coadjacency_2"]
        incidence_1 = graph.graph_matrices["incidence_1"]
        incidence_2 = graph.graph_matrices["incidence_2"]

        try:
            # Initial linear transformations
            x_0 = self.lin_0_input(x_0)
            x_1 = self.lin_1_input(x_1)
            x_2 = torch.zeros(incidence_2.shape[1], self.in_channels_2, dtype=WEIGHT_DTYPE, device=DEVICE)

            # Convert to sparse tensors if needed
            matrices = [adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2]
            for i, matrix in enumerate(matrices):
                if not isinstance(matrix, torch.sparse.Tensor):
                    matrices[i] = matrix.to_sparse()
            adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2 = matrices

            # Process through HMC layers
            x_0_current, x_1_current, x_2_current = x_0, x_1, x_2
            for layer in self.layers:
                x_0_current, x_1_current, x_2_current = layer(x_0_current, x_1_current, x_2_current, adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2)
                x_0_current = F.dropout(x_0_current, p=0.5, training=self.training)
                x_1_current = F.dropout(x_1_current, p=0.5, training=self.training)
                x_2_current = F.dropout(x_2_current, p=0.5, training=self.training)

            # Final linear transformations
            x_0_out = self.lin_0(x_0_current)
            x_1_out = self.lin_1(x_1_current)
            x_2_out = self.lin_2(x_2_current)

            # Calculate means and handle NaN values
            two_dimensional_cells_mean = torch.nanmean(x_2_out, dim=0)
            two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0

            one_dimensional_cells_mean = torch.nanmean(x_1_out, dim=0)
            one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0

            zero_dimensional_cells_mean = torch.nanmean(x_0_out, dim=0)
            zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0

            return two_dimensional_cells_mean + one_dimensional_cells_mean + zero_dimensional_cells_mean

        except RuntimeError as e:
            print(f"Shape debug info:")
            print(f"x_0 shape: {x_0.shape}")
            print(f"x_1 shape: {x_1.shape}")
            print(f"x_2 shape: {x_2.shape}")
            print(f"adjacency_0 shape: {adjacency_0.shape}")
            print(f"adjacency_1 shape: {adjacency_1.shape}")
            print(f"coadjacency_2 shape: {coadjacency_2.shape}")
            print(f"incidence_1 shape: {incidence_1.shape}")
            print(f"incidence_2 shape: {incidence_2.shape}")
            raise e

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        incidence_1 = enhanced_graph.cell_complex.incidence_matrix(rank=1)
        incidence_2 = enhanced_graph.cell_complex.incidence_matrix(rank=2)
        adjacency_0 = enhanced_graph.cell_complex.adjacency_matrix(rank=0)
        adjacency_1 = enhanced_graph.cell_complex.adjacency_matrix(rank=1)
        coadjacency_2 = enhanced_graph.cell_complex.coadjacency_matrix(rank=2)

        matrices = [incidence_1, incidence_2, adjacency_0, adjacency_1, coadjacency_2]
        for i, matrix in enumerate(matrices):
            matrices[i] = from_sparse(matrix).to(WEIGHT_DTYPE).to(DEVICE)

        enhanced_graph.graph_matrices["incidence_1"] = matrices[0]
        enhanced_graph.graph_matrices["incidence_2"] = matrices[1]
        enhanced_graph.graph_matrices["adjacency_0"] = matrices[2]
        enhanced_graph.graph_matrices["adjacency_1"] = matrices[3]
        enhanced_graph.graph_matrices["coadjacency_2"] = matrices[4]
