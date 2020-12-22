'''This file is for unit of Graph based Chemical Evolutionary Algorithm(GCEA)'''
from rdkit.Chem.QED import qed
from score_util import calc_score
from utils import *
import numpy as np
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import Draw


### constant parameters for molecule class
possible_bonds = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE
]
table_of_elements = {
    6 : 'C',
    7 : 'N',
    8 : 'O',
    9 : 'F',
    16 : 'S',
    17 : 'Cl',
    35 : 'Br',
    53 : 'I',
}
vocab_nodes_encode = {
    'C': 1, 'N': 2, 'O': 3, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4, 'I': 4
}


'''Molecule's object class be an individual in population for evolutionary algorithm'''
class Molecule(object):
    def __init__(self, smiles):
        self.smiles = smiles

        self.possible_bonds = possible_bonds
        self.table_of_elements = table_of_elements
        self.vocab_nodes_encode = vocab_nodes_encode
        self.mol = Chem.MolFromSmiles(smiles)

        self.adj = self._get_adj_mat(smiles)
        self.node_list = self._get_node_list(smiles)
        self.num_atom = len(self.node_list)
        self.expand_mat = self._get_expand_mat(self.adj, self.node_list)
        self.life_time = 0
        self.pool_life_time = 0
        self.similarity = -1

        self.property = {
            'qed': qed(self.mol),
            'J_score': calc_score(self.mol),
            'MW' : ExactMolWt(self.mol)
        }
        self.prior_flag = False

    def __hash__(self):
        return hash(self.smiles)

    def __eq__(self, other):
        self_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.smiles))
        other_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(other.smiles))
        return self_smiles == other_smiles

    def _get_adj_mat(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        G = mol2nx(mol)
        atomic_nums = nx.get_node_attributes(G, 'atomic_num')
        adj = np.zeros([len(atomic_nums), len(atomic_nums)])
        bond_list = nx.get_edge_attributes(G, 'bond_type')
        for edge in G.edges():
            first, second = edge
            adj[[first], [second]] = self.possible_bonds.index(bond_list[first, second]) + 1
            adj[[second], [first]] = self.possible_bonds.index(bond_list[first, second]) + 1
        return adj

    def _get_node_list(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        G = mol2nx(mol)
        atomic_nums = nx.get_node_attributes(G, 'atomic_num')
        node_list = []
        for i in range(len(atomic_nums)):
            try:
                node_list.append(self.table_of_elements[atomic_nums[i]])
            except KeyError:
                pass
        return node_list

    def _get_expand_mat(self, adj, node_list):
        def _get_diag_mat(node_list):
            length = len(node_list)
            diag_mat = np.zeros([length, length])
            for i in range(length):
                diag_mat[[i], [i]] = self.vocab_nodes_encode[node_list[i]]
            return diag_mat

        diag_mat = _get_diag_mat(node_list)
        return adj + diag_mat

    def get_image(self, filename):
        Draw.MolToFile(self.mol, filename)

    def _smilarity_between_two_smiles(self, smi2):
        smi1 = self.smiles
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)

        vec1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
        vec2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)

        tani = DataStructs.TanimotoSimilarity(vec1, vec2)
        return tani