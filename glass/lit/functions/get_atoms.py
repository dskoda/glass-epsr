import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


def initialize_atoms(init_atoms):
    node_encoder = OneHotEncoder(sparse_output=False)
    x = node_encoder.fit_transform(init_atoms.numbers.reshape(-1, 1))
    init_species = torch.tensor(x, dtype=torch.float)
    init_pos = torch.tensor(init_atoms.positions, dtype=torch.float)
    init_cell = torch.tensor(init_atoms.cell.tolist(), dtype=torch.float)
    Z_list = node_encoder.categories_[0].tolist()
    return Z_list, init_species, init_pos, init_cell


def get_dH(ref, test):
    from quests.descriptor import get_descriptors
    from quests.entropy import approx_delta_entropy

    k, cutoff = 32, 5.0
    x1 = get_descriptors([test], k=k, cutoff=cutoff)
    x2 = get_descriptors([ref], k=k, cutoff=cutoff)
    dH = approx_delta_entropy(x1, x2, h=0.015, n=5, graph_neighbors=10)
    return np.mean([x for x in dH if x < 5000])
