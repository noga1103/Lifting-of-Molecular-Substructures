from dataclasses import dataclass, field
import dataset.molhiv
import dataset.zinc
import rdkit.Chem
import toponetx as tnx
import torch


DEVICE = torch.device("cuda")
WEIGHT_DTYPE = torch.float32
X_2_WIDTH = 1

ONE_OUT_0_ENCODING_SIZE = 14
ONE_OUT_1_ENCODING_SIZE = 5
ONE_OUT_2_ENCODING_SIZE = 1


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


@dataclass
class EnhancedGraph:
    data: dataset.molhiv.MolHivData | dataset.zinc.ZincData
    regression_value: float
    cell_complex: tnx.CellComplex
    x_0: torch.Tensor
    x_1: torch.Tensor
    x_2: torch.Tensor
    graph_matrices: dict[str, torch.Tensor] = field(default_factory=dict)


def enhance_graphs(datas, regression_fn):
    enhanced_graphs = []
    for data in datas:
        cell_complex = data.cell_complex
        x_0 = generate_x_0(cell_complex)
        x_1 = generate_x_1(cell_complex)
        x_2 = generate_x_2(cell_complex)

        x_0 = x_0.to(DEVICE)
        x_1 = x_1.to(DEVICE)
        x_2 = x_2.to(DEVICE)

        enhanced_graphs.append(
            EnhancedGraph(
                data=data,
                cell_complex=cell_complex,
                regression_value=regression_fn(data),
                x_0=x_0,
                x_1=x_1,
                x_2=x_2,
            )
        )

    return enhanced_graphs


def load_molhiv_data() -> list[EnhancedGraph]:
    datas = dataset.molhiv.get_data()
    return enhance_graphs(datas, lambda x: x.solubility)


def load_zinc_data_small() -> list[EnhancedGraph]:
    datas = dataset.zinc.get_data_small()
    return enhance_graphs(datas, lambda x: x.reward_penalized_log_p)


def load_zinc_data() -> list[EnhancedGraph]:
    datas = dataset.zinc.get_data()
    return enhance_graphs(datas, lambda x: x.reward_penalized_log_p)


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
