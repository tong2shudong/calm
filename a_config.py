import rdkit.Chem as Chem


class Configuration(object):
    def __init__(self,
                 population_size=50,
                 crossover_mu=0.5,
                 graph_size=800,
                 crossover_rate=1,
                 mutate_rate=0.3,
                 alpha=0,
                 num_mutation_max=1,
                 num_mutation_min=1,
                 n_layers=3,
                 replace_hp=0.01,
                 replace_rate=0.25,
                 property_name='J_score'):

        self.n_layers = n_layers
        # parameters for population
        self.population_size = population_size
        self.init_poplution_file_name = '/home/jeffzhu/aaai_ga/data/randomv2.sdf'

        # parameters for crossover
        self.crossover_mu = crossover_mu
        self.graph_size = graph_size

        # parameters for constants
        self.possible_bonds = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE
        ]
        self.table_of_elements = {
            6: 'C',
            7: 'N',
            8: 'O',
            9: 'F',
            16: 'S',
            17: 'Cl',
            35: 'Br',
            53: 'I',
        }
        self.inverse_table_of_elements = {
            'C' : 6,
            'N' : 7,
            'O' : 8,
            'F' : 9,
            'S' : 16,
            'Cl' : 17,
            'Br' : 35,
            'I' : 53
        }
        self.vocab_nodes_encode = {
            'C': 1, 'N': 2, 'O': 3, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4, 'I': 4
        }

        #parameters for mutation
        self.mutation_rate = [0.45, 0.45, 0.1]
        self.temp_elements = {
            0: 'C',
            1: 'N',
            2: 'O',
            3: 'S',
            4: 'Br',
            5: 'Cl',
            6: 'F'
        }
        self.length_elements = len(self.temp_elements)
        self.between_atom_length = 4

        #parameters for trainer
        self.crossover_rate = crossover_rate
        self.alpha = alpha
        self.crossover_sigma = 0.1
        self.full_valence = 5

        #parameters for mutation
        self.mutate_rate = mutate_rate
        self.num_mutation_max = num_mutation_max
        self.num_mutation_min = num_mutation_min
        self.replace_hp = replace_hp
        self.replace_rate = replace_rate

        self.property_name = property_name

        self.init_file_style = 'sdf'
        self.property_name = 'J_score'