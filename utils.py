'''this file is just for general function'''
import rdkit.Chem as Chem
import networkx as nx
from config import Config
import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

config = Config()

# mask the matrix to find valid row to take mutation or crossover operation
def mask(expand_adj):
    node_num = np.count_nonzero(expand_adj.diagonal())
    row_sum = np.sum(expand_adj[:node_num, :node_num], axis=0)
    mask_row = np.argwhere(row_sum < config.full_valence).squeeze(axis=1).tolist()
    return mask_row

# adj2mol is to convert adjacent matrix into mol object in rdkit
def adj2mol(nodes, adj, possible_bonds):
    mol = Chem.RWMol()

    for i in range(len(nodes)):
        #print(nodes[i])
        atom = Chem.Atom(nodes[i])
        mol.AddAtom(atom)

    for i in range(len(nodes)-1):
        for j in range(i + 1, len(nodes)):
            if adj[i, j]:
                mol.AddBond(i, j, possible_bonds[adj[i, j] - 1])

    return mol

# mol2nx is to convert mol object in rdkit into network object
def mol2nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def diversity_scores(mols, data):
    rand_mols = np.random.choice(data.data, 100)
    fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

    scores = np.array(
        list(map(lambda x: __compute_diversity(x, fps) if x is not None else 0, mols)))
    scores = np.clip(remap(scores, 0.9, 0.945), 0.0, 1.0)

    return scores

def __compute_diversity(mol, fps):
    ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
    score = np.mean(dist)
    return score

def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def canonicalize_matrix(matrix, node_list):
    atom_num = len(node_list)
    exp_mat = matrix

    for i in range(atom_num):
        exp_mat[i, i] = config.vocab_nodes_encode[node_list[i]]
    row_sum = np.sum(exp_mat[:atom_num, :atom_num], axis=0)
    error_row = np.argwhere(row_sum > config.full_valence).squeeze(axis=1).tolist()
    if len(error_row) > 0:
        return False
    return True

def _smilarity_between_two_mols(mol1, mol2):
    # mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    vec1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 4, nBits=512)
    vec2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 4, nBits=512)

    tani = DataStructs.TanimotoSimilarity(vec1, vec2)
    return tani