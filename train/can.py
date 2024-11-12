from topomodelx.nn.cell.can_layer import CANLayer
from topomodelx.utils.sparse import from_sparse
from train.train_utils import DEVICE, ONE_OUT_0_ENCODING_SIZE, ONE_OUT_1_ENCODING_SIZE, WEIGHT_DTYPE
import torch
import torch.nn.functional as F

class CANModel(torch.nn.Module):
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        n_layers=2,
    ):
        super().__init__()
        
        # Input linear layers
        self.lin_0_input = torch.nn.Linear(ONE_OUT_0_ENCODING_SIZE, in_channels_0)
        self.lin_1_input = torch.nn.Linear(ONE_OUT_1_ENCODING_SIZE, in_channels_1)
        
        # CAN layers with default parameters that match the original implementation
        self.layers = torch.nn.ModuleList([
            CANLayer(
                in_channels=in_channels_1,
                out_channels=in_channels_1,
                heads=4,
                dropout=0.1,
                concat=True,
                skip_connection=True,
                att_activation=torch.nn.LeakyReLU(),
                add_self_loops=True,
                aggr_func="sum",
                update_func="relu",
                version="v1"  # Using original CAN version
            )
            for _ in range(n_layers)
        ])
        
        # Output linear layers
        self.lin_0 = torch.nn.Linear(in_channels_0, 1)
        self.lin_1 = torch.nn.Linear(in_channels_1, 1)
        self.lin_2 = torch.nn.Linear(in_channels_2, 1)

    def forward(self, graph):
        x_0, x_1 = graph.x_0, graph.x_1
        adjacency_0, incidence_2_t = graph.graph_matrices["adjacency_0"], graph.graph_matrices["incidence_2_t"]
        
        try:
            # Initial linear transformations
            x_0 = self.lin_0_input(x_0)
            x_1 = self.lin_1_input(x_1)
            
            # Convert to sparse tensors if they aren't already
            if not isinstance(adjacency_0, torch.sparse.Tensor):
                adjacency_0 = adjacency_0.to_sparse()
            if not isinstance(incidence_2_t, torch.sparse.Tensor):
                incidence_2_t = incidence_2_t.to_sparse()
            
            # Create padded versions of the matrices to ensure consistent dimensions
            max_dim = max(adjacency_0.shape[0], incidence_2_t.shape[0], incidence_2_t.shape[1])
            
            # Function to pad sparse matrix
            def pad_sparse(sparse_tensor, new_size):
                if sparse_tensor.shape[0] == new_size and sparse_tensor.shape[1] == new_size:
                    return sparse_tensor
                
                indices = sparse_tensor.indices()
                values = sparse_tensor.values()
                return torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=(new_size, new_size),
                    device=DEVICE
                ).coalesce()
            
            # Pad the matrices
            down_laplacian_1 = pad_sparse(adjacency_0, max_dim)
            up_laplacian_1 = pad_sparse(incidence_2_t, max_dim)
            
            # If x_1 needs padding
            if x_1.shape[0] < max_dim:
                padding = torch.zeros(
                    (max_dim - x_1.shape[0], x_1.shape[1]),
                    dtype=x_1.dtype,
                    device=x_1.device
                )
                x_1 = torch.cat([x_1, padding], dim=0)
            
            # Process through CAN layers
            x_1_current = x_1
            for layer in self.layers:
                x_1_current = layer(
                    x=x_1_current,
                    down_laplacian_1=down_laplacian_1,
                    up_laplacian_1=up_laplacian_1
                )
                x_1_current = F.dropout(x_1_current, p=0.5, training=self.training)
            
            # Trim back to original size if needed
            x_1_current = x_1_current[:incidence_2_t.shape[1]]
            
            # Final linear transformations
            x_0_out = self.lin_0(x_0)
            x_1_out = self.lin_1(x_1_current)
            x_2_out = self.lin_2(torch.zeros(incidence_2_t.shape[0], in_channels_2, dtype=WEIGHT_DTYPE, device=DEVICE))
            
            # Calculate means and handle NaN values
            two_dimensional_cells_mean = torch.nanmean(x_2_out, dim=0)
            two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
            
            one_dimensional_cells_mean = torch.nanmean(x_1_out, dim=0)
            one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
            
            zero_dimensional_cells_mean = torch.nanmean(x_0_out, dim=0)
            zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
            
            return two_dimensional_cells_mean + one_dimensional_cells_mean + zero_dimensional_cells_mean
            
        except RuntimeError as e:
            # Print shapes for debugging
            print(f"Shape debug info:")
            print(f"x_0 shape: {x_0.shape}")
            print(f"x_1 shape: {x_1.shape}")
            print(f"adjacency_0 shape: {adjacency_0.shape}")
            print(f"incidence_2_t shape: {incidence_2_t.shape}")
            if isinstance(x_1_current, torch.Tensor):
                print(f"x_1_current shape: {x_1_current.shape}")
            raise e

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        incidence_2_t = enhanced_graph.cell_complex.incidence_matrix(rank=2).T
        adjacency_0 = enhanced_graph.cell_complex.adjacency_matrix(rank=0)
        
        incidence_2_t = from_sparse(incidence_2_t).to(WEIGHT_DTYPE).to(DEVICE)
        adjacency_0 = from_sparse(adjacency_0).to(WEIGHT_DTYPE).to(DEVICE)
        
        enhanced_graph.graph_matrices["incidence_2_t"] = incidence_2_t
        enhanced_graph.graph_matrices["adjacency_0"] = adjacency_0
