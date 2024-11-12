from topomodelx.nn.cell.can_layer import CANLayer, MultiHeadLiftLayer, PoolLayer
from topomodelx.utils.sparse import from_sparse
from train.train_utils import DEVICE, ONE_OUT_0_ENCODING_SIZE, ONE_OUT_1_ENCODING_SIZE, WEIGHT_DTYPE
import torch

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
        
        # CAN layers with correct parameters
        self.layers = torch.nn.ModuleList([
            CANLayer(
                in_channels=in_channels_1,
                out_channels=in_channels_1,
                heads=4,
                concat=True,
                dropout=0.1,
                attention_dropout=0.1
            )
            for _ in range(n_layers)
        ])
        
        # Multi-head lift layers for dimension interactions
        self.lift_0_to_1 = MultiHeadLiftLayer(
            channels_0=in_channels_0,
            channels_1=in_channels_1,
            heads=4
        )
        self.lift_1_to_2 = MultiHeadLiftLayer(
            channels_0=in_channels_1,
            channels_1=in_channels_2,
            heads=4
        )
        
        # Pool layers for aggregating information
        self.pool_2_to_1 = PoolLayer(
            channels_1=in_channels_2,
            channels_0=in_channels_1
        )
        self.pool_1_to_0 = PoolLayer(
            channels_1=in_channels_1,
            channels_0=in_channels_0
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
        
        # Create initial x_2 from lifting x_1
        x_2 = torch.zeros(
            (incidence_2_t.shape[0], x_1.shape[1]), 
            dtype=WEIGHT_DTYPE, 
            device=DEVICE
        )
        
        # Process through CAN layers
        for layer in self.layers:
            # Up dimension
            x_0_up = self.lift_0_to_1(x_0=x_0, x_1=None, boundary_1=adjacency_0)
            x_1_up = self.lift_1_to_2(x_0=x_1, x_1=None, boundary_1=incidence_2_t)
            
            # Down dimension
            x_2_down = self.pool_2_to_1(x_1=x_2, boundary_1=incidence_2_t.t())
            x_1_down = self.pool_1_to_0(x_1=x_1, boundary_1=adjacency_0.t())
            
            # CAN layer processing - note we primarily focus on x_1 since that's what CANLayer processes
            x_1 = layer(x=x_1 + x_0_up + x_2_down, edge_index=adjacency_0)
            
            # Update x_0 and x_2 based on the pooled and lifted values
            x_0 = x_0 + x_1_down
            x_2 = x_2 + x_1_up
        
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
        incidence_2_t = enhanced_graph.cell_complex.incidence_matrix(rank=2).T
        adjacency_0 = enhanced_graph.cell_complex.adjacency_matrix(rank=0)
        
        incidence_2_t = from_sparse(incidence_2_t).to(WEIGHT_DTYPE).to(DEVICE)
        adjacency_0 = from_sparse(adjacency_0).to(WEIGHT_DTYPE).to(DEVICE)
        
        enhanced_graph.graph_matrices["incidence_2_t"] = incidence_2_t
        enhanced_graph.graph_matrices["adjacency_0"] = adjacency_0
