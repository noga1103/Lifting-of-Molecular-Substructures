```python
class HNHNModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dimensions,
        n_layers=2,
    ):
        super().__init__()
        
        # Move everything to GPU first
        self.device = DEVICE
        
        # Create dummy incidence matrix on correct device
        indices = torch.zeros((2, 1), dtype=torch.long).to(self.device)
        values = torch.zeros(1).to(self.device)
        dummy_incidence = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(ONE_HOT_0_ENCODING_SIZE, 1),
            device=self.device
        )
        
        # Create and move model to device before initialization
        self.base_model = HNHN(
            in_channels=ONE_HOT_0_ENCODING_SIZE,
            hidden_channels=hidden_dimensions,
            n_layers=n_layers,
            incidence_1=dummy_incidence,
        )
        self.base_model = self.base_model.to(self.device)
        
        # Move linear layer to device
        self.linear = torch.nn.Linear(hidden_dimensions, 1).to(self.device)
        self.out_pool = True
        
        # Ensure all internal tensors are on correct device
        self = self.to(self.device)

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"].to(self.device)
        cc = graph.data.combinatorial_complex
        
        # Process incidence matrix
        incidence_1 = cc.incidence_matrix(0, 1)
        incidence_1 = torch.from_numpy(incidence_1.todense()).to(self.device, dtype=WEIGHT_DTYPE)
        
        # Create sparse tensor on correct device
        indices = torch.nonzero(incidence_1).t()
        values = incidence_1[indices[0], indices[1]]
        incidence_1 = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=incidence_1.size(),
            device=self.device
        )
        
        # Update model's incidence matrix
        self.base_model.incidence_1 = incidence_1
        
        # Forward pass
        x_0_processed, _ = self.base_model(x_0, incidence_1=incidence_1)
        x = torch.max(x_0_processed, dim=0)[0] if self.out_pool else x_0_processed
        return self.linear(x)

    @staticmethod
    def add_graph_matrices(enhanced_graph):
        cc = enhanced_graph.data.combinatorial_complex
        x_0 = generate_x_0(cc).to(DEVICE)
        x_1 = generate_x_1_combinatorial(cc).to(DEVICE)
        x_2 = generate_x_2_combinatorial(cc).to(DEVICE)
        
        enhanced_graph.graph_matrices = {
            "x_0": x_0,
            "x_1": x_1,
            "x_2": x_2,
        }
```
