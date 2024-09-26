import math
import os
import pickle

from dataclasses import dataclass

import toponetx

from smiles_converter import smiles_to_complex

# Chunk the pkl files b/c github has a 100MB limit on filesizes.
NUM_CHUNKS = 5
PKL_FILE_BASE = "dataset/pkl_data/zinc"
CSV_FILE = "dataset/csv_data/250k_rndm_zinc_drugs_clean_3.csv"


@dataclass
class ZincData:
    smiles: str
    log_p: float  # water–octanol partition coefficient
    qed: float  # Quantitative Estimation of Drug-likeness
    sas: float  # synthetic accessibility score
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
        smiles = row["smiles"]
        log_p = row["logP"]
        qed = row["qed"]
        sas = row["SAS"]
        complex = smiles_to_complex(smiles)
        result.append(ZincData(smiles=smiles, log_p=log_p, qed=qed, sas=sas, complex=complex))

    save_pkl_chunks(result)

    return result


def save_pkl_chunks(data, base_path=PKL_FILE_BASE, num_chunks=NUM_CHUNKS):
    chunk_size = math.ceil(len(data) / num_chunks)

    # Iterate over each chunk and save it as a separate pickle file
    for i in range(num_chunks):
        # Get the current chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))  # Don't go beyond the list's length
        chunk = data[start_idx:end_idx]

        # Define the output file name
        output_file = f"{base_path}_{i}.pkl"

        # Save the current chunk as a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(chunk, f)


def read_pkl(base_path=PKL_FILE_BASE, num_chunks=NUM_CHUNKS):
    data = []

    # Iterate over the number of chunks (files) and load them
    for i in range(num_chunks):
        input_file = f"{base_path}_{i}.pkl"

        with open(input_file, "rb") as f:
            chunk = pickle.load(f)
        data.extend(chunk)  # Add the loaded chunk to the combined list

    return data


if __name__ == "__main__":
    save_pkl()
    print(len(get_data()))
    print(get_data()[0])
