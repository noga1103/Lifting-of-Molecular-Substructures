from topomodelx.nn.cell.cwn_layer import CWNLayer
from topomodelx.utils.sparse import from_sparse
from train.train_utils import DEVICE, ONE_HOT_0_ENCODING_SIZE, ONE_HOT_1_ENCODING_SIZE, WEIGHT_DTYPE, generate_x_0, generate_x_1, generate_x_2
import torch
import torch.nn.functional as F


class CWNModel(torch.nn.Module):
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
        self.lin_0_input = torch.nn.Linear(ONE_HOT_0_ENCODING_SIZE, self.in_channels_0)
        self.lin_1_input = torch.nn.Linear(ONE_HOT_1_ENCODING_SIZE, self.in_channels_1)

        # CWN layers
        self.layers = torch.nn.ModuleList(
            [
                CWNLayer(in_channels_0=self.in_channels_0, in_channels_1=self.in_channels_1, in_channels_2=self.in_channels_2, out_channels=self.in_channels_1)
                for _ in range(n_layers)
            ]
        )

        # Output linear layers
        self.lin_0 = torch.nn.Linear(self.in_channels_0, 1)
        self.lin_1 = torch.nn.Linear(self.in_channels_1, 1)
        self.lin_2 = torch.nn.Linear(self.in_channels_2, 1)

    def forward(self, graph):
        x_0, x_1 = graph.graph_matrices["x_0"], graph.graph_matrices["x_1"]
        adjacency_1, incidence_2, incidence_1_t = (graph.graph_matrices["adjacency_1"], graph.graph_matrices["incidence_2"], graph.graph_matrices["incidence_1_t"])

        try:
            # Initial linear transformations
            x_0 = self.lin_0_input(x_0)
            x_1 = self.lin_1_input(x_1)

            # Convert to sparse tensors if needed
            if not isinstance(adjacency_1, torch.sparse.Tensor):
                adjacency_1 = adjacency_1.to_sparse()
            if not isinstance(incidence_2, torch.sparse.Tensor):
                incidence_2 = incidence_2.to_sparse()
            if not isinstance(incidence_1_t, torch.sparse.Tensor):
                incidence_1_t = incidence_1_t.to_sparse()

            # Process through CWN layers
            x_1_current = x_1
            for layer in self.layers:
                x_1_current = layer(x_0=x_0, x_1=x_1_current, x_2=x_2, adjacency_0=adjacency_1, incidence_2=incidence_2, incidence_1_t=incidence_1_t)
                x_1_current = F.dropout(x_1_current, p=0.5, training=self.training)

            # Final linear transformations
            x_0_out = self.lin_0(x_0)
            x_1_out = self.lin_1(x_1_current)
            x_2_out = self.lin_2(x_2)

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
            print(f"adjacency_1 shape: {adjacency_1.shape}")
            print(f"incidence_2 shape: {incidence_2.shape}")
            print(f"incidence_1_t shape: {incidence_1_t.shape}")
            if isinstance(x_1_current, torch.Tensor):
                print(f"x_1_current shape: {x_1_current.shape}")
            raise e

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        x_0 = generate_x_0(enhanced_graph.data.cell_complex).to(DEVICE)
        x_1 = generate_x_1(enhanced_graph.data.cell_complex).to(DEVICE)
        x_2 = generate_x_2(enhanced_graph.data.cell_complex).to(DEVICE)

        # TODO: this was copied from above. is it right or will generate_x_2 do the same thing?
        # x_2 = torch.ones(incidence_2.shape[1], self.in_channels_2, dtype=WEIGHT_DTYPE, device=DEVICE)

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
