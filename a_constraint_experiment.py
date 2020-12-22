from a_algorithm import *
from a_config import Configuration

config = Configuration()
config.init_file_style = 'pkl'
config.init_poplution_file_name = 'data/800_mols_for_constraint.pkl'

if __name__ == '__main__':
    trainer = Trainer(config)