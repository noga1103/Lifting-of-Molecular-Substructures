class HNHNModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dimensions,
        n_layers=2,
    ):
        super().__init__()
        # Move to device first
        self.to(DEVICE)
        
        # Create dummy incidence matrix on the correct device
        dummy_incidence = torch.sparse_coo_tensor(
            indices=torch.zeros((2, 1), dtype=torch.long, device=DEVICE),
            values=torch.zeros(1, device=DEVICE),
            size=(ONE_HOT_0_ENCODING_SIZE, 1)
        )
        
        # Initialize normalization matrices on device
        n_nodes, n_edges = ONE_HOT_0_ENCODING_SIZE, 1
        D0_left = torch.eye(n_nodes, device=DEVICE)
        D1_left = torch.eye(n_edges, device=DEVICE)
        D0_right = torch.eye(n_nodes, device=DEVICE)
        D1_right = torch.eye(n_edges, device=DEVICE)
        
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = HNHNLayer(
                in_channels=ONE_HOT_0_ENCODING_SIZE if i == 0 else hidden_dimensions,
                hidden_channels=hidden_dimensions,
                incidence_1=dummy_incidence,
                use_normalized_incidence=True
            )
            # Move layer matrices to device
            layer.D0_left_alpha_inverse = D0_left.clone()
            layer.D1_left_beta_inverse = D1_left.clone()
            layer.D1_right_alpha = D1_right.clone()
            layer.D0_right_beta = D0_right.clone()
            layer.to(DEVICE)
            self.layers.append(layer)
        
        self.linear = torch.nn.Linear(hidden_dimensions, 1)
        self.out_pool = True

    def forward(self, graph):
        x_0 = graph.graph_matrices["x_0"]
        cc = graph.data.combinatorial_complex
        
        incidence_dense = torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to(DEVICE).to(WEIGHT_DTYPE)
        indices = torch.nonzero(incidence_dense, as_tuple=True)
        values = incidence_dense[indices]
        incidence_1 = torch.sparse_coo_tensor(
            indices=torch.stack(indices),
            values=values,
            size=incidence_dense.size(),
            device=DEVICE
        )

        x_0_processed = x_0
        for layer in self.layers:
            layer.incidence_1 = incidence_1
            layer.incidence_1_transpose = incidence_1.transpose(1, 0)
            
            # Recompute normalization on device
            if layer.use_normalized_incidence:
                layer.n_nodes, layer.n_edges = incidence_1.shape
                layer.compute_normalization_matrices()
            
            x_0_processed, _ = layer(x_0_processed, incidence_1=incidence_1)

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
