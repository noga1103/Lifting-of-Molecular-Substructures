import toponetx as tnx
import torch

import rdkit.Chem
import numpy as np

from sklearn.model_selection import train_test_split

from topomodelx.nn.cell.ccxn import CCXN
from topomodelx.utils.sparse import from_sparse

import dataset.molhiv

from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

torch.manual_seed(0)
DEVICE = torch.device("cuda")
WEIGHT_DTYPE = torch.float32
X_2_WIDTH = 1
# Define input dimensions
IN_CHANNELS_0 = 20
IN_CHANNELS_1 = 20
IN_CHANNELS_2 = 20
ONE_OUT_0_ENCODING_SIZE = 14
ONE_OUT_1_ENCODING_SIZE = 5
ONE_OUT_2_ENCODING_SIZE = 1


@dataclass
class EnhancedGraph:
    data: dataset.molhiv.MolHivData
    cell_complex: tnx.CellComplex
    x_0: torch.Tensor
    x_1: torch.Tensor
    x_2: torch.Tensor
    incidence_2_t: torch.Tensor
    adjacency_0: torch.Tensor


ALL_ATOMIC_SYMBOLS = {
    None: 0,  # None
    "C": 1,  # Carbon
    "O": 2,  # Oxygen
    "N": 3,  # Nitrogen
    "P": 4,  # Phosphorus
    "S": 5,  # Sulfur
    "Cl": 6,  # Chlorine
    "F": 7,  # Fluorine
    "I": 8,  # Iodine
    "Br": 9,  # Bromine
    "Se": 10,  # Selenium
    "Si": 11,  # Silicon
    "As": 12,  # Arsenic
    "B": 13,  # Boron
}

ALL_BOND_TYPES = {
    None: 0,  # Missing bond type
    rdkit.Chem.rdchem.BondType(1): 1,
    rdkit.Chem.rdchem.BondType(2): 2,
    rdkit.Chem.rdchem.BondType(3): 3,
    rdkit.Chem.rdchem.BondType(12): 4,
}


def load_molhiv_data() -> list[EnhancedGraph]:
    datas = dataset.molhiv.get_data()
    enhanced_graphs = []
    for data in datas:
        cell_complex = data.cell_complex
        x_0 = generate_x_0(cell_complex)
        x_1 = generate_x_1(cell_complex)
        x_2 = generate_x_2(cell_complex)

        incidence_2_t = cell_complex.incidence_matrix(rank=2).T
        adjacency_0 = cell_complex.adjacency_matrix(rank=0)
        incidence_2_t = from_sparse(incidence_2_t).to(WEIGHT_DTYPE).to(DEVICE)
        adjacency_0 = from_sparse(adjacency_0).to(WEIGHT_DTYPE).to(DEVICE)

        x_0 = x_0.to(DEVICE)
        x_1 = x_1.to(DEVICE)
        x_2 = x_2.to(DEVICE)

        enhanced_graphs.append(
            EnhancedGraph(
                data=data,
                cell_complex=cell_complex,
                x_0=x_0,
                x_1=x_1,
                x_2=x_2,
                incidence_2_t=incidence_2_t,
                adjacency_0=adjacency_0,
            )
        )

    return enhanced_graphs


def generate_x_0(complex: tnx.CellComplex) -> torch.Tensor:
    num_symbols = max(ALL_ATOMIC_SYMBOLS.values()) + 1  # Length of one-hot vector
    node_to_symbol = complex.get_node_attributes("atomic_symbol")
    x_0 = []
    for node in complex.nodes:
        symbol = node_to_symbol.get(node, None)
        index = ALL_ATOMIC_SYMBOLS.get(symbol, 0)
        one_hot = torch.zeros(num_symbols, dtype=WEIGHT_DTYPE)
        one_hot[index] = 1.0
        x_0.append(one_hot)
    return torch.stack(x_0)


def generate_x_1(complex: tnx.CellComplex) -> torch.Tensor:
    num_bond_types = max(ALL_BOND_TYPES.values()) + 1
    edge_to_bond_type = complex.get_edge_attributes("bond_type")
    x_1 = []
    for edge in complex.edges:
        bond_type = edge_to_bond_type.get(edge, None)
        index = ALL_BOND_TYPES.get(bond_type, 0)
        one_hot = torch.zeros(num_bond_types, dtype=WEIGHT_DTYPE)
        one_hot[index] = 1.0
        x_1.append(one_hot)
    if not x_1:
        return torch.zeros((0, num_bond_types), dtype=WEIGHT_DTYPE)
    return torch.stack(x_1)


def generate_x_2(complex: tnx.CellComplex) -> torch.Tensor:
    return torch.zeros((len(complex.cells), X_2_WIDTH), dtype=WEIGHT_DTYPE)


molhiv_data = load_molhiv_data()


# Create Network
class Network(torch.nn.Module):
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        n_layers=2,
        att=False,
    ):
        super().__init__()
        self.base_model = CCXN(
            in_channels_0,
            in_channels_1,
            in_channels_2,
            n_layers=n_layers,
            att=att,
        )
        self.lin_0_input = torch.nn.Linear(ONE_OUT_0_ENCODING_SIZE, in_channels_0)
        self.lin_1_input = torch.nn.Linear(ONE_OUT_1_ENCODING_SIZE, in_channels_1)
        self.lin_0 = torch.nn.Linear(in_channels_0, 1)
        self.lin_1 = torch.nn.Linear(in_channels_1, 1)
        self.lin_2 = torch.nn.Linear(in_channels_2, 1)

    def forward(self, x_0, x_1, adjacency_0, incidence_2_t):
        x_0 = self.lin_0_input(x_0)
        x_1 = self.lin_1_input(x_1)
        x_0, x_1, x_2 = self.base_model(x_0, x_1, adjacency_0, incidence_2_t)
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)

        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        return two_dimensional_cells_mean + one_dimensional_cells_mean + zero_dimensional_cells_mean


model = Network(IN_CHANNELS_0, IN_CHANNELS_1, IN_CHANNELS_2, n_layers=8)
model = model.to(DEVICE)

# Loss function and optimizer
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Split dataset
train_data, test_data = train_test_split(molhiv_data, test_size=0.2, shuffle=True)

# Training loop
test_interval = 10
num_epochs = 1000
for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    for graph in train_data:
        x_0 = graph.x_0
        x_1 = graph.x_1
        incidence_2_t = graph.incidence_2_t
        adjacency_0 = graph.adjacency_0
        y = torch.tensor(graph.data.solubility, dtype=WEIGHT_DTYPE).to(DEVICE)

        optimizer.zero_grad()
        y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
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
                x_0 = graph.x_0
                x_1 = graph.x_1
                incidence_2_t = graph.incidence_2_t
                adjacency_0 = graph.adjacency_0
                y = torch.tensor(graph.data.solubility, dtype=WEIGHT_DTYPE).to(DEVICE)

                y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
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

# Train with attention
model = Network(in_channels_0, in_channels_1, in_channels_2, n_layers=2, att=True)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop with attention
for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    for graph in train_data:
        x_0 = graph.x_0
        x_1 = graph.x_1
        incidence_2_t = graph.incidence_2_t
        adjacency_0 = graph.adjacency_0
        y = torch.tensor(graph.data.solubility, dtype=WEIGHT_DTYPE).to(DEVICE)

        optimizer.zero_grad()
        y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
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
                x_0 = graph.x_0
                x_1 = graph.x_1
                incidence_2_t = graph.incidence_2_t
                adjacency_0 = graph.adjacency_0
                y = torch.tensor(graph.data.solubility, dtype=WEIGHT_DTYPE).to(DEVICE)

                y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
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
