import pickle
import time
import copy
from a_population import Population
from a_crossover import Crossover
from a_mutation import Mutation
import numpy as np
from a_config import Configuration
from utils import _smilarity_between_two_mols

import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

class Trainer():
    def __init__(self, configfile, time_counts=40, property_name='J_score', is_mutate=True, delta_time=120):
        self.configfile = configfile

        self.pop = Population(self.configfile)
        self.operator_cro = Crossover(self.configfile)
        self.operator_mut = Mutation(self.configfile)

        self.property_name = property_name
        self.is_mutate = is_mutate
        self.delta_time = delta_time
        self.time_counts = time_counts
        self.alpha = self.configfile.alpha

        self.best_res = -10000

    def _init_similarity_population(self):
        standard_mol = self.pop.population_pool[0]
        for idx in range(self.pop.population_size):
            sim = self._calculate_similarity(standard_mol, self.pop.population_pool[idx])
            self.pop.population_pool[idx].similarity = sim

    def _calculate_similarity(self, standard_mol, molecule):
        if molecule.property[self.property_name] > standard_mol.property[self.property_name]:
            sim = 0
        else:
            _sim = _smilarity_between_two_mols(standard_mol.mol, molecule.mol)
            sim = 0 if _sim == 1 else _sim
        return sim

    def _sort_function(self, molecule):
        return molecule.property[self.property_name] + self.alpha * molecule.similarity

    def train(self):
        #print(self.configfile.n_layers)
        self.pop._init_population()
        time_count = 0
        starttime = (int)(time.time())

        for iteration in range(100000000):
            if time_count >= self.time_counts:
                final_pool = self.pop.population_pool
                pickle.dump(final_pool, open('qed_max_pool.pkl', 'wb'))
                break
            num_new_atom = self.pop.population_size
            temp_pool = []
            for j in range(int(num_new_atom * self.configfile.crossover_rate)):
                index_ = np.random.choice(self.pop.population_size, 2, replace=False)
                index1, index2 = index_[0], index_[1]
                mol1, mol2 = self.pop.population_pool[index1], self.pop.population_pool[index2]
                new_mol = self.operator_cro.crossover(mol1, mol2)
                temp_pool = temp_pool + new_mol

            if self.is_mutate:
                mutate_index = np.random.choice(num_new_atom, int(num_new_atom * config.mutate_rate), replace=False)
                for j in mutate_index:
                    temp_molecule = copy.deepcopy(self.pop.population_pool[j])
                    new_molecule_pool = self.operator_mut.mutate(temp_molecule)
                    temp_pool = temp_pool + new_molecule_pool

            # for j in range(num_new_atom):
            #     index_ = np.random.choice(num_new_atom, 1, replace=False)
            #     index = index_[0]
            #     mol = self.pop.population_pool[index]
            #     mol1 = self.operator_mut.add_funtional_group(mol)
            #     temp_pool.append(mol1)
            # standard_mol= self.pop.population_pool[0]
            # for idx in range(len(temp_pool)):
            #     sim = self._calculate_similarity(standard_mol, temp_pool[idx])
            #     temp_pool[idx].similarity = sim

            total = temp_pool + self.pop.population_pool
            total = set(total)
            #res = sorted(total, key=lambda x: x.property[self.property_name], reverse=True)
            res = sorted(total, key=lambda x: self._sort_function(x), reverse=True)

            res_score = [res[idx].property[self.property_name] for idx in range(self.pop.population_size)]
            print('mean is : ' + str(np.mean(res_score)) + ' std is : ' + str(np.std(res_score)) + ' max is : ' + str(
                np.max(res_score)))
            print(res[0].smiles)

            self.pop.population_pool = []
            for j in range(self.pop.population_size):
                self.pop.population_pool.append(res[j])
            # max_pool.append(res[0])

            currenttime = (int)(time.time())
            if (currenttime - starttime) > self.delta_time:
                starttime = currenttime
                print('ok')
                # logger.info('mean is : '+ str(np.mean(res_score)) + ' std is : '+ str(np.std(res_score)) + ' max is : ' + str(np.max(res_score)))
                time_count += 1
                if self.pop.population_pool[0].property[self.property_name] > self.best_res:
                    pickle.dump(self.pop.population_pool[0].smiles, open('final_smiles','wb'))
                    print('write into ok!')
                # if np.std(res_score) < 1e-6:
                #     break

if __name__ == '__main__':
    config = Configuration()
    trainer = Trainer(config)
    #trainer._init_similarity_population()
    trainer.train()