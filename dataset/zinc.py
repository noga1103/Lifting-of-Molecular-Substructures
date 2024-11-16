import marshal
import cPickle as pickle
import gc
from dataclasses import dataclass
import toponetx
import networkx as nx
from rdkit import Chem

# Chunk the pkl files b/c github has a 100MB limit on filesizes.
NUM_CHUNKS = 12
PKL_FILE_BASE = "dataset/pkl_data/zinc"
CSV_FILE = "dataset/csv_data/250k_rndm_zinc_drugs_clean_3.csv"

@dataclass
class ZincData:
    smiles: str
    reward_penalized_log_p: float
    log_p: float
    qed: float
    sas: float
    combinatorial_complex: toponetx.CombinatorialComplex
    cell_complex: toponetx.CellComplex

DATA = None
DATA_SMALL = None

def calculate_reward_penalized_log_p(smiles, log_p, sas):
    # Constants remain unchanged
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    mol = Chem.MolFromSmiles(smiles)
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    cycle_length = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_length = 0 if cycle_length <= 6 else cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (sas - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle

def get_data():
    global DATA
    if DATA is None:
        # Disable garbage collection during data loading
        gc.disable()
        try:
            DATA = read_pkl()
        finally:
            # Re-enable garbage collection after loading
            gc.enable()
    return DATA

def get_data_small():
    global DATA_SMALL
    if DATA_SMALL is None:
        DATA_SMALL = read_pkl()
    return DATA_SMALL

def save_pkl():
    import pandas
    from dataset.smiles_converter import smiles_to_combinatorial_complex, smiles_to_cell_complex

    csv = pandas.read_csv(CSV_FILE)
    result = []
    
    # Disable garbage collection during processing
    gc.disable()
    try:
        for row in csv.to_dict(orient="records"):
            smiles = row["smiles"].rstrip()
            log_p = row["logP"]
            qed = row["qed"]
            sas = row["SAS"]
            reward_penalized_log_p = calculate_reward_penalized_log_p(smiles, log_p, sas)
            combinatorial_complex = smiles_to_combinatorial_complex(smiles)
            cell_complex = smiles_to_cell_complex(smiles)
            result.append(
                ZincData(
                    smiles=smiles,
                    reward_penalized_log_p=reward_penalized_log_p,
                    log_p=log_p,
                    qed=qed,
                    sas=sas,
                    combinatorial_complex=combinatorial_complex,
                    cell_complex=cell_complex,
                )
            )
    finally:
        gc.enable()

    save_pkl_chunks(result)
    return result

def save_pkl_chunks(data, base_path=PKL_FILE_BASE, num_chunks=NUM_CHUNKS):
    import math

    chunk_size = math.ceil(len(data) / num_chunks)
    
    # Configure pickle for faster processing
    pickle.Pickler.fast = 1
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        chunk = data[start_idx:end_idx]
        
        output_file = f"{base_path}_{i}.pkl"
        
        with open(output_file, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_small():
    return read_pkl(base_path=PKL_FILE_BASE, num_chunks=1)

def read_pkl(base_path=PKL_FILE_BASE, num_chunks=NUM_CHUNKS):
    data = []
    
    # Disable garbage collection during loading
    gc.disable()
    try:
        for i in range(num_chunks):
            input_file = f"{base_path}_{i}.pkl"
            with open(input_file, "rb") as f:
                chunk = pickle.load(f)
            data.extend(chunk)
    finally:
        gc.enable()
    
    return data

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
