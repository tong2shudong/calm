'''this file is to design a Population class for evolutionary algorithm'''
from a_molecule import Molecule
from utils import *
import pickle

class Population(object):
    def __init__(self, config):
        self.population_pool = []
        self.population_size = config.population_size
        self.init_poplution_file_name = config.init_poplution_file_name
        self.num_members = 0
        self.config = config
        self.init_file_style = config.init_file_style

    #get the length of the population_pool
    def get_length(self):
        return len(self.population_pool)

    # init population using 500 random molecules from zinc dataset
    def _init_population(self):
        if self.init_file_style is 'sdf':
            init_data = Chem.SDMolSupplier(self.init_poplution_file_name)

            for i in range(self.population_size):
                self.population_pool.append(Molecule(Chem.MolToSmiles(init_data[i])))
        elif self.init_file_style is 'pkl':
            init_data = pickle.load(open(self.init_poplution_file_name, 'rb'))

            for i in range(self.population_size):
                self.population_pool.append(Molecule(init_data[i]))
        self.num_members = self.config.population_size
