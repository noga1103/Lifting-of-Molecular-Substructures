from topomodelx.nn.cell.can_layer import CANLayer, MultiHeadLiftLayer, PoolLayer
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
        
        # Adjust dimensions to match number of heads
        n_heads = 4
        head_dim = in_channels_1 // n_heads
        
        # Lift layer for node to edge features
        self.lift_0_to_1 = MultiHeadLiftLayer(
            in_channels_0=head_dim,  # Reduced to match head dimension
            heads=n_heads,
            signal_lift_dropout=0.5
        )
        
        # Pre-lift transformation to match dimensions
        self.pre_lift_transform = torch.nn.Linear(in_channels_0, head_dim * n_heads)
        
        # CAN layers
        self.layers = torch.nn.ModuleList()
        
        # First layer
        self.layers.append(
            CANLayer(
                in_channels=in_channels_1,  # Same dimension as input
                out_channels=in_channels_1,
                heads=n_heads,
                concat=True,
                skip_connection=True,
                att_activation=torch.nn.LeakyReLU(0.2),
                dropout=0.1,
                attention_dropout=0.1
            )
        )
        
        # Additional layers
        for _ in range(n_layers - 1):
            self.layers.append(
                CANLayer(
                    in_channels=in_channels_1,
                    out_channels=in_channels_1,
                    heads=n_heads,
                    concat=True,
                    skip_connection=True,
                    att_activation=torch.nn.LeakyReLU(0.2),
                    dropout=0.1,
                    attention_dropout=0.1
                )
            )
        
        # Output linear layers
        self.lin_0 = torch.nn.Linear(in_channels_0, 1)
        self.lin_1 = torch.nn.Linear(in_channels_1, 1)
        self.lin_2 = torch.nn.Linear(in_channels_2, 1)

    def forward(self, graph):
        x_0, x_1 = graph.x_0, graph.x_1
        adjacency_0, incidence_2_t = graph.graph_matrices["adjacency_0"], graph.graph_matrices["incidence_2_t"]
        
        # Initial linear transformations
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        
        # Transform node features to match lift layer dimensions
        x_0_transformed = self.pre_lift_transform(x_0)
        
        # Create upper and lower Laplacians
        down_laplacian_1 = adjacency_0.to_sparse()
        up_laplacian_1 = incidence_2_t.to_sparse()
        
        # Reshape x_1 to match the expected dimensions
        batch_size = x_1.size(0)
        n_heads = 4
        head_dim = x_1.size(1) // n_heads
        x_1_reshaped = x_1.view(batch_size, n_heads, head_dim)
        
        # Lift node features to edge level
        try:
            x_1_lifted = self.lift_0_to_1(x_0_transformed.view(batch_size, n_heads, -1), 
                                        down_laplacian_1, 
                                        x_1_reshaped)
            
            # Process through CAN layers
            x_1_current = x_1_lifted
            for layer in self.layers:
                x_1_current = layer(x_1_current, down_laplacian_1, up_laplacian_1)
                x_1_current = F.dropout(x_1_current, p=0.5, training=self.training)
        
        except RuntimeError as e:
            print(f"Error in forward pass. Shapes:")
            print(f"x_0_transformed shape: {x_0_transformed.shape}")
            print(f"x_1 shape: {x_1.shape}")
            print(f"adjacency_0 shape: {adjacency_0.shape}")
            print(f"incidence_2_t shape: {incidence_2_t.shape}")
            raise e
        
        # Final linear transformations
        x_0_out = self.lin_0(x_0)
        x_1_out = self.lin_1(x_1_current)
        x_2_out = self.lin_2(torch.zeros(incidence_2_t.shape[0], 1, dtype=WEIGHT_DTYPE, device=DEVICE))
        
        # Calculate means and handle NaN values
        two_dimensional_cells_mean = torch.nanmean(x_2_out, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        
        one_dimensional_cells_mean = torch.nanmean(x_1_out, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        
        zero_dimensional_cells_mean = torch.nanmean(x_0_out, dim=0)
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
