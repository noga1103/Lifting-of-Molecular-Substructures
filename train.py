from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from train.ccxn import CCXNModel
from train.train_utils import DEVICE, WEIGHT_DTYPE, load_molhiv_data

torch.manual_seed(0)

HIDDEN_DIMENSIONS = 20

model = CCXNModel(HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, n_layers=8)
model = model.to(DEVICE)


full_data = load_molhiv_data()
[model.add_graph_matrices(graph) for graph in full_data]

# Loss function and optimizer
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Split dataset
train_data, test_data = train_test_split(full_data, test_size=0.2, shuffle=True)

# Training loop
test_interval = 10
num_epochs = 100
for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    for graph in train_data:
        y = torch.tensor([graph.regression_value], dtype=WEIGHT_DTYPE).to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(graph)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    if epoch_i % test_interval == 0:
        model.eval()
        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            train_mean_loss = np.mean(epoch_loss)
            test_losses = []
            for graph in test_data:
                y = torch.tensor([graph.regression_value], dtype=WEIGHT_DTYPE).to(DEVICE)
                y_hat = model(graph)
                test_loss = loss_fn(y_hat, y)
                y_true_list.append(y.item())
                y_pred_list.append(y_hat.item())
                test_losses.append(test_loss.item())
            test_mean_loss = np.mean(test_losses)
            r2 = r2_score(y_true_list, y_pred_list)
            mae = mean_absolute_error(y_true_list, y_pred_list)
            rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))

            print(
                f"Epoch:{epoch_i}, Train Loss: {train_mean_loss:.4f}, Test Loss: {test_mean_loss:.4f}, " f"R_: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}",
                flush=True,
            )
