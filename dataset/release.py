from dataclasses import dataclass
import dill as pickle

import toponetx


PKL_FILE = "dataset/pkl_data/release.pkl"
CSV_FILE = "dataset/csv_data/SMILES_Big_Data_Set.csv"


@dataclass
class ReleaseData:
    smiles: str
    pic_50: float  # Half maximal inhibitory concentration (log scale)
    log_p: float  # solubility
    combinatorial_complex: toponetx.CombinatorialComplex
    cell_complex: toponetx.CellComplex


DATA = None


def get_data():
    global DATA
    if DATA is None:
        DATA = read_pkl()

    return DATA


def save_pkl():
    from dataset.smiles_converter import smiles_to_combinatorial_complex, smiles_to_cell_complex
    import pandas

    csv = pandas.read_csv(CSV_FILE)
    result = []
    for row in csv.to_dict(orient="records"):
        smiles = row["SMILES"]
        pic_50 = row["pIC50"]
        log_p = row["logP"]
        combinatorial_complex = smiles_to_combinatorial_complex(smiles)
        cell_complex = smiles_to_cell_complex(smiles)
        result.append(
            ReleaseData(
                smiles=smiles,
                log_p=log_p,
                pic_50=pic_50,
                combinatorial_complex=combinatorial_complex,
                cell_complex=cell_complex,
            )
        )

    with open(PKL_FILE, "wb") as f:
        pickle.dump(result, f)

    return result


def read_pkl():
    with open(PKL_FILE, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    save_pkl()
    print(len(get_data()))
    print(get_data()[0].combinatorial_complex)
    print("======================")
    print(get_data()[0].combinatorial_complex.get_node_attributes("atomic_symbol"))
    print("======================")
    print(get_data()[0].combinatorial_complex.get_cell_attributes("bond_type"))
    print("======================")
    print(get_data()[0].cell_complex)
    print("======================")
    print(get_data()[0].cell_complex.get_node_attributes("atomic_symbol"))
    print("======================")
    print(get_data()[0].cell_complex.get_cell_attributes(rank=1, name="bond_type"))
    print("======================")
    print(len(get_data()[0].cell_complex.get_cell_attributes(rank=1, name="bond_type")))
