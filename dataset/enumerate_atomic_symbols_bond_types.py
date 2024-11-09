# Print out all the unique Symbols and BondTypes across all the datasets to enumerate
# the options for the one-hot encoding.

from dataset.molhiv import get_data as molhiv_get_data
from dataset.release import get_data as release_get_data
from dataset.zinc import get_data as zinc_get_data

m_data = molhiv_get_data()
r_data = release_get_data()
# z_data = zinc_get_data()

all_atomic_symbols = set()
all_bond_types = set()


# for d in m_data + r_data + z_data:
for d in m_data + r_data:
    atomic_symbols = d.combinatorial_complex.get_node_attributes("atomic_symbol").values()
    bond_types = d.combinatorial_complex.get_cell_attributes("bond_type").values()
    all_atomic_symbols.update(atomic_symbols)
    all_bond_types.update(bond_types)


print(all_atomic_symbols)
print(all_bond_types)
