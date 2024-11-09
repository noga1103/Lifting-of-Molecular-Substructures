import toponetx as tnx
import torch

import rdkit.Chem

from sklearn.model_selection import train_test_split

from topomodelx.nn.cell.ccxn import CCXN
from topomodelx.utils.sparse import from_sparse

import dataset.molhiv

from dataclasses import dataclass
import torch


torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DTYPE = torch.float32
X_2_WIDTH = 5


@dataclass
class EnhancedGraph:
    data: dataset.molhiv.MolHivData
    cell_complex: tnx.CellComplex
    x_0: torch.Tensor
    x_1: torch.Tensor
    x_2: torch.Tensor


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
        enhanced_graphs.append(EnhancedGraph(cell_complex=cell_complex, x_0=x_0, x_1=x_1, x_2=x_2))

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
    # TODO: maybe also add the atomic symbol of each node to the edge features?
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
    # TODO: maybe make the feature be the number of atoms/edges in the face
    return torch.zeros((len(complex.cells), X_2_WIDTH), dtype=WEIGHT_DTYPE)


molhiv_data = load_molhiv_data()

### Use the data from molhiv_data to run the neural network below which originally used the shrec data. get rid of the shrec logic.
### add the adjacency and incidence matrices to the EnhancedGraph dataclass
### keep everything on the GPU the whole time. Do not move it back and forth with the CPU.
### regress data.solubility, which is a float. Do this with a linear layer at the end of the model.

shrec, _ = tnx.datasets.shrec_16(size="small")

shrec = {key: np.array(value) for key, value in shrec.items()}
x_0s = shrec["node_feat"]
x_1s = shrec["edge_feat"]
x_2s = shrec["face_feat"]

ys = shrec["label"]
simplexes = shrec["complexes"]


i_complex = 6
print(f"The {i_complex}th simplicial complex has {x_0s[i_complex].shape[0]} nodes with features of dimension {x_0s[i_complex].shape[1]}.")
print(f"The {i_complex}th simplicial complex has {x_1s[i_complex].shape[0]} edges with features of dimension {x_1s[i_complex].shape[1]}.")
print(f"The {i_complex}th simplicial complex has {x_2s[i_complex].shape[0]} faces with features of dimension {x_2s[i_complex].shape[1]}.")


### Get incidence / adjacency matrices
cc_list = []
incidence_2_t_list = []
adjacency_0_list = []
for simplex in simplexes:
    cell_complex = simplex.to_cell_complex()
    cc_list.append(cell_complex)

    incidence_2_t = cell_complex.incidence_matrix(rank=2).T
    adjacency_0 = cell_complex.adjacency_matrix(rank=0)
    incidence_2_t = from_sparse(incidence_2_t)
    adjacency_0 = from_sparse(adjacency_0)
    incidence_2_t_list.append(incidence_2_t)
    adjacency_0_list.append(adjacency_0)
i_complex = 6
print(f"The {i_complex}th cell complex has an incidence_2_t matrix of shape {incidence_2_t_list[i_complex].shape}.")
print(f"The {i_complex}th cell complex has an adjacency_0 matrix of shape {adjacency_0_list[i_complex].shape}.")


### Create Network
class Network(torch.nn.Module):
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        num_classes,
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
        self.lin_0 = torch.nn.Linear(in_channels_0, num_classes)
        self.lin_1 = torch.nn.Linear(in_channels_1, num_classes)
        self.lin_2 = torch.nn.Linear(in_channels_2, num_classes)

    def forward(self, x_0, x_1, adjacency_0, incidence_2_t):
        x_0, x_1, x_2 = self.base_model(x_0, x_1, adjacency_0, incidence_2_t)
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)
        # Take the average of the 2D, 1D, and 0D cell features. If they are NaN, convert them to 0.
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        # Return the sum of the averages
        return two_dimensional_cells_mean + one_dimensional_cells_mean + zero_dimensional_cells_mean


in_channels_0 = x_0s[0].shape[-1]
in_channels_1 = x_1s[0].shape[-1]
in_channels_2 = 5
num_classes = 2
print(f"The dimension of input features on nodes, edges and faces are: {in_channels_0}, {in_channels_1} and {in_channels_2}.")
model = Network(in_channels_0, in_channels_1, in_channels_2, num_classes, n_layers=2)
model = model.to(DEVICE)


### Train params
crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()


### Split dataset
test_size = 0.2
x_0_train, x_0_test = train_test_split(x_0s, test_size=test_size, shuffle=False)
x_1_train, x_1_test = train_test_split(x_1s, test_size=test_size, shuffle=False)
incidence_2_t_train, incidence_2_t_test = train_test_split(incidence_2_t_list, test_size=test_size, shuffle=False)
adjacency_0_train, adjacency_0_test = train_test_split(adjacency_0_list, test_size=test_size, shuffle=False)
y_train, y_test = train_test_split(ys, test_size=test_size, shuffle=False)


### Training loop
test_interval = 2
num_epochs = 10
for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    for x_0, x_1, incidence_2_t, adjacency_0, y in zip(
        x_0_train,
        x_1_train,
        incidence_2_t_train,
        adjacency_0_train,
        y_train,
        strict=True,
    ):
        x_0, x_1, y = (
            torch.tensor(x_0).float().to(DEVICE),
            torch.tensor(x_1).float().to(DEVICE),
            torch.tensor(y).float().to(DEVICE),
        )
        incidence_2_t, adjacency_0 = (
            incidence_2_t.float().to(DEVICE),
            adjacency_0.float().to(DEVICE),
        )
        opt.zero_grad()
        y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())

    if epoch_i % test_interval == 0:
        with torch.no_grad():
            train_mean_loss = np.mean(epoch_loss)
            for x_0, x_1, incidence_2_t, adjacency_0, y in zip(
                x_0_test,
                x_1_test,
                incidence_2_t_test,
                adjacency_0_test,
                y_test,
                strict=True,
            ):
                x_0, x_1, y = (
                    torch.tensor(x_0).float().to(DEVICE),
                    torch.tensor(x_1).float().to(DEVICE),
                    torch.tensor(y).float().to(DEVICE),
                )
                incidence_2_t, adjacency_0 = (
                    incidence_2_t.float().to(DEVICE),
                    adjacency_0.float().to(DEVICE),
                )
                y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
                test_loss = loss_fn(y_hat, y)
            print(
                f"Epoch:{epoch_i}, Train Loss: {train_mean_loss:.4f} Test Loss: {test_loss:.4f}",
                flush=True,
            )


### Train with attention
model = Network(in_channels_0, in_channels_1, in_channels_2, num_classes, n_layers=2, att=True)
model = model.to(DEVICE)
crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

### Training loop with attention
test_interval = 2
num_epochs = 10
for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    for x_0, x_1, incidence_2_t, adjacency_0, y in zip(
        x_0_train,
        x_1_train,
        incidence_2_t_train,
        adjacency_0_train,
        y_train,
        strict=True,
    ):
        x_0, x_1, y = (
            torch.tensor(x_0).float().to(DEVICE),
            torch.tensor(x_1).float().to(DEVICE),
            torch.tensor(y).float().to(DEVICE),
        )
        incidence_2_t, adjacency_0 = (
            incidence_2_t.float().to(DEVICE),
            adjacency_0.float().to(DEVICE),
        )
        opt.zero_grad()
        y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())

    if epoch_i % test_interval == 0:
        with torch.no_grad():
            train_mean_loss = np.mean(epoch_loss)
            for x_0, x_1, incidence_2_t, adjacency_0, y in zip(
                x_0_test,
                x_1_test,
                incidence_2_t_test,
                adjacency_0_test,
                y_test,
                strict=True,
            ):
                x_0, x_1, y = (
                    torch.tensor(x_0).float().to(DEVICE),
                    torch.tensor(x_1).float().to(DEVICE),
                    torch.tensor(y).float().to(DEVICE),
                )
                incidence_2_t, adjacency_0 = (
                    incidence_2_t.float().to(DEVICE),
                    adjacency_0.float().to(DEVICE),
                )
                y_hat = model(x_0, x_1, adjacency_0, incidence_2_t)
                test_loss = loss_fn(y_hat, y)
            print(
                f"Epoch:{epoch_i}, Train Loss: {train_mean_loss:.4f} Test Loss: {test_loss:.4f}",
                flush=True,
            )
