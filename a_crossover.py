from queue import Queue
from random import *
from utils import *
from a_molecule import Molecule

class Crossover():
    def __init__(self, config):
        self.crossover_mu = config.crossover_mu
        self.graph_size = config.graph_size
        self.vocab_nodes_encode = config.vocab_nodes_encode
        self.possible_bonds = config.possible_bonds

    def _bfs_molecule(self, molecule):
        length = molecule.num_atom

        # sigma = self.crossover_sigma * length
        mu = self.crossover_mu * length
        '''using normal distribution to select the number of molecules'''
        '''this mu and sigma of normal distribution is decided by the above paramters'''
        num_sample = (int)(np.random.poisson(mu))
        while num_sample >= length:
            num_sample = (int)(np.random.poisson(mu))

        '''we choose the start node from the set who has the most degrees for BFS algorithm '''
        adj = molecule.adj
        np.putmask(adj, adj >= 1, 1)
        row_sum = np.sum(adj, axis=0)
        max_ = np.max(row_sum)
        index_arr = [i for i in range(len(row_sum)) if row_sum[i] == max_]
        index_start = np.random.choice(index_arr, 1)[0]

        hash = np.zeros(length)
        res = []
        q = Queue(self.graph_size)
        q.put(index_start)
        res.append(index_start)
        hash[index_start] = 1

        while len(res) < num_sample:
            node = q.get()
            node_list = list(np.squeeze(np.argwhere(adj[node] >= 1), axis=1))
            for n in node_list:
                if hash[n] != 1:
                    hash[n] = 1
                    q.put(n)
                    res.append(n)

        temp_node_list = []
        '''res store the index of atom in the molecules'''
        for i in res:
            temp_node_list.append(molecule.node_list[i])

        '''reconstruct the subgraph in adj form'''
        num_atom = len(res)
        temp_mat = np.zeros([num_atom, num_atom])
        # print(res)
        for i in range(num_atom - 1):
            for j in range(1, num_atom):
                temp_mat[i, j] = molecule.expand_mat[res[i], res[j]]
                temp_mat[j, i] = molecule.expand_mat[res[j], res[i]]

        temp_mat = temp_mat.astype(int)
        length = len(temp_node_list)

        for i in range(length):
            temp_mat[i, i] = self.vocab_nodes_encode[temp_node_list[i]]
        # mol = adj2mol(temp_node_list, temp_mat, possible_bonds)
        # Draw.MolToFile(mol, 'test2.png')
        '''temp_mat for expand_mat, temp_node_list for node_list'''
        return temp_mat, temp_node_list

    # combine two subgraph to get an new graph represented as a molecule
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

    # we sample from two molecule and get an new molecule after mutation
    def crossover(self, mol1, mol2):
        temp_mat1, temp_node_list1 = self._bfs_molecule(mol1)
        temp_mat2, temp_node_list2 = self._bfs_molecule(mol2)

        goal_mat = self._combine_two_subgraph(temp_mat1, temp_mat2).astype(int)
        goal_list = temp_node_list1 + temp_node_list2
        adj = goal_mat
        for i in range(len(goal_list)):
            adj[i, i] = 0
        mol_temp = adj2mol(goal_list, adj, self.possible_bonds)

        return [Molecule(Chem.MolToSmiles(mol_temp))]