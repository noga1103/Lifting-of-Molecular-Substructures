class Trainer:
    def __init__(self, model, training_dataset, training_dataloader, learning_rate, device):
        print("Initializing Trainer...")
        self.device = device
        print(f"Moving model to device: {device}")
        try:
            self.model = model.to(device)
        except Exception as e:
            print(f"Error moving model to device: {str(e)}")
            raise e
        
        self.training_dataset = training_dataset
        self.training_dataloader = training_dataloader
        self.crit = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        print("Trainer initialized successfully")

    def _to_device(self, x) -> list[torch.Tensor]:
        """Converts tensors to the correct type and moves them to the device.

        Parameters
        ----------
        x : List[torch.Tensor]
            List of tensors to convert.
        Returns
        -------
        List[torch.Tensor]
            List of converted tensors to float type and moved to the device.
        """

        return [el[0].float().to(self.device) for el in x]

    def train(self, num_epochs=500, test_interval=25) -> None:
        """Trains the model for the specified number of epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train.
        test_interval : int
            Interval between testing epochs.
        """
        for epoch_i in range(num_epochs):
            training_accuracy, epoch_loss = self._train_epoch()
            print(
                f"Epoch: {epoch_i} loss: {epoch_loss:.4f} Train_acc: {training_accuracy:.4f}",
                flush=True,
            )
            if (epoch_i + 1) % test_interval == 0:
                test_accuracy = self.validate()
                print(f"Test_acc: {test_accuracy:.4f}", flush=True)


   
    def _train_epoch(self):
      training_samples = len(self.training_dataloader.dataset)
      total_loss = 0
      self.model.train()
      for batch in self.training_dataloader:
          x_0, x_1, x_2, adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2, y = batch
          
          # Move dense tensors to device
          x_0, x_1, x_2, y = [b.to(self.device) for b in [x_0, x_1, x_2, y]]
          
          # Move sparse tensors to device
          adjacency_0 = [adj.to(self.device) for adj in adjacency_0]
          adjacency_1 = [adj.to(self.device) for adj in adjacency_1]
          coadjacency_2 = [coadj.to(self.device) for coadj in coadjacency_2]
          incidence_1 = [inc.to(self.device) for inc in incidence_1]
          incidence_2 = [inc.to(self.device) for inc in incidence_2]

          self.opt.zero_grad()

          y_hat = self.model(x_0, x_1, x_2, adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2)
          
          loss = self.crit(y_hat, y)
          loss.backward()
          self.opt.step()

          total_loss += loss.item() * y.size(0)

      epoch_loss = total_loss / training_samples
      return epoch_loss
    def _compute_loss_and_update(self, y_hat, y) -> float:
        """Computes the loss, performs backpropagation, and updates the model's parameters.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output of the model.
        y : torch.Tensor
            The ground truth.

        Returns
        -------
        loss: float
            The loss value.
        """

        loss = self.crit(y_hat, y)
        loss.backward()
        self.opt.step()
        return loss.item()

    def validate(self) -> float:
        """Validates the model using the testing dataloader.

        Returns
        -------
        test_accuracy : float
            The mean testing accuracy.
        """
        correct = 0
        self.model.eval()
        test_samples = len(self.testing_dataloader.dataset)
        with torch.no_grad():
            for sample in self.testing_dataloader:
                (
                    x_0,
                    x_1,
                    x_2,
                    adjacency_0,
                    adjacency_1,
                    coadjacency_2,
                    incidence_1,
                    incidence_2,
                ) = self._to_device(sample[:-1])

                y_hat = self.model(
                    x_0,
                    x_1,
                    x_2,
                    adjacency_0,
                    adjacency_1,
                    coadjacency_2,
                    incidence_1,
                    incidence_2,
                )
                y = sample[-1][0].long().to(self.device)
                correct += (y_hat.argmax() == y).sum().item()
            return correct / test_samples
