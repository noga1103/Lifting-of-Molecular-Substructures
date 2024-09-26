from dataclasses import dataclass
import pickle

import toponetx

from smiles_converter import smiles_to_complex

PKL_FILE = "dataset/pkl_data/release.pkl"
CSV_FILE = "dataset/csv_data/SMILES_Big_Data_Set.csv"


@dataclass
class ReleaseData:
    smiles: str
    pic_50: float  # Half maximal inhibitory concentration (log scale)
    log_p: float  # solubility
    complex: toponetx.CombinatorialComplex


DATA = None


def get_data():
    global DATA
    if DATA is None:
        DATA = read_pkl()

    return DATA


def save_pkl():
    import pandas

    csv = pandas.read_csv(CSV_FILE)
    result = []
    for row in csv.to_dict(orient="records"):
        smiles = row["SMILES"]
        pic_50 = row["pIC50"]
        log_p = row["logP"]
        complex = smiles_to_complex(smiles)
        result.append(ReleaseData(smiles=smiles, log_p=log_p, pic_50=pic_50, complex=complex))

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
