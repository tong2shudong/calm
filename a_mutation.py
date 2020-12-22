from score_util import *
from utils import *
from a_molecule import Molecule
from numpy.random import choice
from a_config import Configuration
import copy


benzene = Molecule('C1=CC=CC=C1')

class Mutation():
    def __init__(self, config):
        self.config = config

    def _add_bond(self, molecule):
        temp_mol = copy.deepcopy(molecule)

        if temp_mol.num_atom < 2:
            return molecule

        temp_expand_adj = temp_mol.expand_mat
        temp_adj = temp_mol.adj
        mask_row = mask(temp_expand_adj)

        goal_mol = None
        goal_smiles = None

        for i in mask_row:
            row = temp_adj[i]
            for j in range(len(row)):
                if row[j] > 0 and j in mask_row:
                    temp_adj[i][j] += 1
                    temp_adj[j][i] += 1
                    goal_adj = temp_adj
                    goal_node_list = temp_mol.node_list
                    goal_mol = adj2mol(goal_node_list, goal_adj.astype(int), self.config.possible_bonds)
                    goal_smiles = Chem.MolToSmiles(goal_mol)
                    break
            if goal_mol != None:
                break

        if goal_mol != None:
            return Molecule(goal_smiles)
        else:
            return molecule

    def _add_atom(self, molecule):
        temp_mol = copy.deepcopy(molecule)

        temp_node_list = copy.deepcopy(temp_mol.node_list)
        temp_adj = copy.deepcopy(temp_mol.adj)
        temp_expand_adj = copy.deepcopy(temp_mol.expand_mat)

        temp_elements = self.config.temp_elements

        atom_index = np.random.choice(self.config.length_elements, 1)[0]#, p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])[0]
        atom = temp_elements[atom_index]
        mask_row = mask(temp_expand_adj)
        if len(mask_row) < 1:
            return molecule
        mask_index = np.random.choice(mask_row, 1)[0]

        goal_length = len(temp_node_list) + 1
        goal_adj = np.zeros([goal_length, goal_length])
        goal_adj[:goal_length - 1, :goal_length - 1] = temp_adj
        goal_adj[goal_length - 1, mask_index] = goal_adj[mask_index, goal_length - 1] = 1

        temp_node_list.append(atom)
        goal_node_list = temp_node_list

        goal_mol = adj2mol(goal_node_list, goal_adj.astype(int), self.config.possible_bonds)
        goal_smiles = Chem.MolToSmiles(goal_mol)

        return Molecule(goal_smiles)

    def _add_atom_between_bond(self, molecule):
        temp_mol = copy.deepcopy(molecule)

        temp_elements = self.config.temp_elements
        atom_index = np.random.choice(4, 1)[0]#, p=[0.7, 0.1, 0.1, 0.1])[0]
        atom = temp_elements[atom_index]

        temp_adj = temp_mol.adj

        length = temp_mol.num_atom
        insert_index1 = np.random.choice(length, 1)
        insert_row = temp_adj[insert_index1][0]

        insert_index2 = 0
        for i in range(len(insert_row)):
            if insert_row[i] > 0:
                insert_index2 = i

        temp_adj[insert_index1, insert_index2] = temp_adj[insert_index2, insert_index1] = 0

        goal_adj = np.zeros([length + 1, length + 1])
        goal_adj[:length, :length] = temp_adj
        goal_adj[length, insert_index1] = goal_adj[insert_index1, length] = 1
        goal_adj[insert_index2, length] = goal_adj[length, insert_index2] = 1

        temp_node_list = temp_mol.node_list
        temp_node_list.append(atom)
        goal_node_list = temp_node_list

        goal_mol = adj2mol(goal_node_list, goal_adj.astype(int), self.config.possible_bonds)
        goal_smiles = Chem.MolToSmiles(goal_mol)

        return Molecule(goal_smiles)

    def _change_atom(self, molecule):
        temp_mol = copy.deepcopy(molecule)
        temp_node_list = temp_mol.node_list
        temp_adj = temp_mol.adj

        length_molecule = temp_mol.num_atom

        sum_row = np.sum(temp_adj, axis=0)
        sorted_index = np.argsort(sum_row)

        flag = False
        for idx in range(int(length_molecule * 0.3)):
            if flag == True:
                break
            now_index = sorted_index[idx]
            original_atom_type = temp_node_list[now_index]
            bond_value = sum_row[now_index]

            for k in range(5):
                atom_index = np.random.randint(0, 7)
                atom_type = self.config.temp_elements[atom_index]
                if atom_type != original_atom_type and self.config.vocab_nodes_encode[atom_type] + bond_value <= 5:
                    flag = True
                    temp_node_list[now_index] = self.config.inverse_table_of_elements[atom_type]
                    break

        goal_mol = adj2mol(temp_node_list, temp_adj.astype(int), self.config.possible_bonds)
        goal_smiles = Chem.MolToSmiles(goal_mol)

        return Molecule(goal_smiles)

    def _combine_two_subgraph(self, subg1, subg2):
        len1, len2 = subg1.shape[0], subg2.shape[0]
        goal_adj = np.zeros([len1 + len2, len1 + len2])
        goal_adj[:len1, :len1] = subg1
        goal_adj[len1:, len1:] = subg2
        ''' this part can be modified '''
        row1, row2 = mask(subg1), mask(subg2)
        row2 = [i + len1 for i in row2]
        index1, index2 = choice(row1), choice(row2)
        goal_adj[index1, index2] = 1
        goal_adj[index2, index1] = 1
        return goal_adj

    def add_funtional_group(self, molecule):
        adj1, adj2 = molecule.adj, benzene.adj
        goal_adj = self._combine_two_subgraph(adj1, adj2)
        goal_node_list = molecule.node_list + benzene.node_list
        goal_mol = adj2mol(goal_node_list, goal_adj.astype(int), self.config.possible_bonds)
        goal_smiles = Chem.MolToSmiles(goal_mol)
        return Molecule(goal_smiles)

    def mutate(self, molecule):
        molecule_pool = []
        molecule_pool.append(molecule)

        num_mutation_max, num_mutation_min = self.config.num_mutation_max, self.config.num_mutation_min
        num_mutation = np.random.choice([number for number in range(num_mutation_min, num_mutation_max+1)], 1, replace=False)[0]

        for iteration in range(num_mutation):
            choice = np.random.choice(4, 1, p=[0.3, 0.3, 0.1, 0.3])[0]
            if choice == 0:
                temp_mol = self._add_atom(molecule_pool[-1])
                molecule_pool.append(temp_mol)
            elif choice == 1:
                temp_mol = self._add_atom_between_bond(molecule_pool[-1])
                molecule_pool.append(temp_mol)
            elif choice == 2:
                temp_mol = self._add_bond(molecule_pool[-1])
                molecule_pool.append(temp_mol)
            elif choice == 3:
                temp_mol = self._change_atom(molecule_pool[-1])
                molecule_pool.append(temp_mol)
        return molecule_pool[1:]

from a_config import Configuration
config = Configuration()

from rdkit.Chem import Draw
if __name__ == '__main__':
    mutate_op = Mutation(config)
    smiles = 'CCCCCCCCCCCCCCCCC'
    mol = Molecule(smiles)

    test = mutate_op._change_atom(mol)
    Draw.MolToFile(test.mol, 'change_atom.png')