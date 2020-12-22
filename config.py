'''This file is for hypeparameters and configuration'''
import rdkit.Chem as Chem

'''restore paramters using a class'''
class Config(object):
    def __init__(self):
        self.poplution_size = 50
        self.crossover_rate = 0.8
        self.init_poplution_file_name = './randomv2.sdf'
        self.crossover_mu = 0.5
        self.crossover_sigma = 0.1
        self.graph_size = 80
        self.full_valence = 5
        self.mutation_rate = [0.45, 0.45, 0.1]
        self.temp_elements = {
            0: 'C',
            1: 'N',
            2: 'O',
            3: 'S',
        }
        self.vocab_nodes_encode = {
            'C': 1, 'N': 2, 'O': 3, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4, 'I': 4
        }

        self.possible_bonds = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE
        ]

        self.length_elements = len(self.temp_elements)

        self.mutate_rate = 0.03
