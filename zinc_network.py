class Network(torch.nn.Module):
    def __init__(
        self,
        channels_per_layer,
        negative_slope=0.2,
    ):
        super().__init__()
        self.base_model = HMC(
            channels_per_layer,
            negative_slope,
        )
        # Change the output dimension to 1 for regression
        self.l0 = torch.nn.Linear(channels_per_layer[-1][2][0], 1)
        self.l1 = torch.nn.Linear(channels_per_layer[-1][2][1], 1)
        self.l2 = torch.nn.Linear(channels_per_layer[-1][2][2], 1)

    def forward(self, x_0, x_1, x_2, adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2):
      batch_size = x_0.size(0)
      
      outputs = []
      for i in range(batch_size):
          x_0_i = x_0[i][x_0[i].sum(dim=1) != 0]  # Remove padding
          x_1_i = x_1[i][x_1[i].sum(dim=1) != 0]
          x_2_i = x_2[i][x_2[i].sum(dim=1) != 0]
          
          # Process through the base model
          x_0_i, x_1_i, x_2_i = self.base_model(
              x_0_i, x_1_i, x_2_i, 
              adjacency_0[i], adjacency_1[i], coadjacency_2[i], 
              incidence_1[i], incidence_2[i]
          )
          
          # Apply linear layers
          x_0_i = self.l0(x_0_i)
          x_1_i = self.l1(x_1_i)
          x_2_i = self.l2(x_2_i)
          
          # Aggregate results
          x_i = torch.cat([x_0_i, x_1_i, x_2_i], dim=0)
          x_i = torch.mean(x_i, dim=0)
          
          outputs.append(x_i)
      
      return torch.stack(outputs)
