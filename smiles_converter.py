from rdkit import Chem
from rdkit.Chem import BRICS
import toponetx


def smiles_to_complex(smiles, add_hydrogens=False):
    mol = Chem.MolFromSmiles(smiles)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    broken_mols = BRICS.BreakBRICSBonds(mol)
    frags = Chem.GetMolFrags(broken_mols)
    print(f"atoms: {mol.GetNumAtoms()}")
    print(f"bonds: {mol.GetNumBonds()}")
    print(f"frags: {frags}")

    complex = toponetx.CombinatorialComplex()

    # add nodes (atoms)
    for atom in mol.GetAtoms():
        complex.add_cell([atom.GetIdx()], rank=0, symbol=atom.GetSymbol())

    # add edges (bonds)
    for bond in mol.GetBonds():
        complex.add_cell([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()], rank=1, bond_type=bond.GetBondType())

    # add faces (brics fragments)
    for frag in frags:
        face = []
        for idx in frag:
            if idx >= mol.GetNumAtoms():
                # fake nodes are added to the frags at the ends of the broken bonds. Ignore them.
                continue
            face.append(idx)
        if len(face) < 3:
            # functional group is too small to be its own face
            continue
        complex.add_cell(face, rank=2)

    return complex


if __name__ == "__main__":
    smiles = "CC(=O)O"
    smiles = "CC(NCC(O)COC1=CC=CC2=CC=CC=C21)C"
    print(smiles_to_complex(smiles))
