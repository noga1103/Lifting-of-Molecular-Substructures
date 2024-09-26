from dataclasses import dataclass
import pickle

import toponetx

from smiles_converter import smiles_to_complex

PKL_FILE = "dataset/pkl_data/molhiv.pkl"
CSV_FILE = "dataset/csv_data/molhiv_clean_smallsample.csv"


@dataclass
class MolHivData:
    smiles: str
    solubility: float  # measured log solubility in mols per litre
    name: str  # human readable name
    complex: toponetx.CombinatorialComplex


MOLHIV_DATA = None


def get_data():
    global MOLHIV_DATA
    if MOLHIV_DATA is None:
        MOLHIV_DATA = read_pkl()

    return MOLHIV_DATA


def save_pkl():
    import pandas

    csv = pandas.read_csv(CSV_FILE)
    result = []
    for row in csv.to_dict(orient="records"):
        solubility = row["measured log solubility in mols per litre"]
        smiles = row["smiles"]
        name = row["mol_id"]
        complex = smiles_to_complex(smiles)
        result.append(MolHivData(smiles=smiles, solubility=solubility, name=name, complex=complex))

    with open(PKL_FILE, "wb") as f:
        pickle.dump(result, f)

    return result


def read_pkl():
    with open(PKL_FILE, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    save_pkl()
    print(len(get_data()))
    print(get_data()[0])
