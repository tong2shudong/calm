import pickle
import time
import copy
from a_population import Population
from a_crossover import Crossover
from a_mutation import Mutation
import numpy as np
from a_config import Configuration
from a_candidate_pool import CandidatePool
from a_molecule import *

import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from utils import *

class Record():
    def __init__(self,
                 record_size=15,
                 delta=1e-5):
        self.record_pool = []
        self.record_size = record_size
        self.delta = delta

    def add_record(self, record):
        if len(self.record_pool) < self.record_size:
            self.record_pool.append(record)
        elif len(self.record_pool) == self.record_size:
            self.record_pool.pop(0)
            self.record_pool.append(record)
        elif len(self.record_pool) > self.record_size:
            self.record_pool = self.record_pool[-self.record_size:]

    def print_situation(self):
        print(np.std(self.record_pool))

    def is_converse(self):
        if len(self.record_pool) > self.record_size:
            self.record_pool = self.record_pool[-self.record_size:]
        if self.record_size == len(self.record_pool):
            pool_std = np.std(self.record_pool)
            if self.delta >= pool_std:
                return True
            else:
                return False
        else:
            return False

class PropertyRecord():
    def __init__(self,
                 property_reocord_size=100,
                 property_name='qed'):
        self.property_mol_pool = []
        self.property_reocord_size = property_reocord_size
        self.property_name = property_name

    def get_length(self):
        return len(self.property_mol_pool)

    def add_molecule(self, molecule):
        length = self.get_length()
        if length < self.property_reocord_size:
            self.property_mol_pool.append(molecule)
        else:
            self.property_mol_pool.pop(self.property_reocord_size-1)
        temp_mol_pool = self.property_mol_pool
        temp_mol_pool = sorted(temp_mol_pool, key=lambda x: x.property[self.property_name], reverse=True)
        self.property_mol_pool = temp_mol_pool

    def get_top_property_value(self):
        res = [self.property_mol_pool[i].property[self.property_name] for i in range(self.get_length())]
        return res

    def get_top_property_smiles(self):
        res = [self.property_mol_pool[i].smiles for i in range(self.get_length())]
        return res

    def get_top_property_molecule(self):
        return self.property_mol_pool




class Trainer():
    def __init__(self, configfile, time_counts=40, is_mutate=True, delta_time=120):
        self.configfile = configfile

        self.record_pool = []
        self.pop_pool = []
        self.operator_cro = Crossover(self.configfile)
        self.operator_mut = Mutation(self.configfile)
        #self.pop = None

        self.property_name = 'J_score'
        self.delta_time = delta_time
        self.time_counts = time_counts
        self.alpha = self.configfile.alpha
        self.candidate_pool_list = []

        self.best_res = -10000
        self.property_record = PropertyRecord()
        self.buff_pool = []

    def _init_pop_pool(self):
        for i in range(self.configfile.n_layers):
            temp_pop = Population(self.configfile)
            self.pop_pool.append(temp_pop)
        self.pop_pool[0]._init_population()

        first_candidate = CandidatePool(candidate_pool_size=2000)
        self.candidate_pool_list.append(first_candidate)
        for i in range(1, self.configfile.n_layers):
            temp_candidate_pool = CandidatePool()
            self.candidate_pool_list.append(temp_candidate_pool)

        for i in range(self.configfile.n_layers):
            temp_record = Record()
            self.record_pool.append(temp_record)

    def _sort_function(self, molecule):
        return molecule.property[self.property_name] + self.alpha * molecule.similarity

    def _crossover(self, this_pop):
        new_mol_pool = []
        for j in range(int(this_pop.population_size * self.configfile.crossover_rate)):
            index_ = np.random.choice(this_pop.get_length(), 2, replace=False)
            index1, index2 = index_[0], index_[1]
            mol1, mol2 = this_pop.population_pool[index1], this_pop.population_pool[index2]
            new_mol = self.operator_cro.crossover(mol1, mol2)
            new_mol_pool = new_mol_pool +new_mol
        return new_mol_pool

    def _mutate(self, this_pop):
        new_mol_pool = []

        num_new_atom = this_pop.get_length()
        mutate_index = np.random.choice(num_new_atom, int(num_new_atom * config.mutate_rate), replace=False)
        for j in mutate_index:
            temp_molecule = copy.deepcopy(this_pop.population_pool[j])
            new_molecule_pool = self.operator_mut.mutate(temp_molecule)
            new_mol_pool = new_mol_pool + new_molecule_pool
        return new_mol_pool

    def _smilarity_between_two_smiles(self, smi1, smi2):
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)

        vec1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
        vec2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)

        tani = DataStructs.TanimotoSimilarity(vec1, vec2)
        return tani

    def print_population(self, pop_index = 0):
        pop = self.pop_pool[pop_index]
        pop_size = self.configfile.population_size

        for i in range(pop_size):
            print('SMILES : ' + pop.population_pool[i].smiles + ' J_score : ' + str(pop.population_pool[i].property['J_score']))

    # def train_without_constraint(self):
    #     #code for processing pool
    #     self._init_pop_pool()
    #
    #     time_count = 0
    #     starttime = (int)(time.time())
    #
    #
    #     multi_level_flag = False
    #     break_flag = False
    #     multi_mutation_flag = False
    #
    #     self.configfile.num_mutation_max = 1
    #     self.configfile.num_mutation_min = 1
    #     is_crossover = True
    #     is_mutate = True
    #
    #     half_min_flag = True
    #
    #     this_pop = self.pop_pool[0]
    #
    #     count = 0
    #
    #     for iteration in range(100000000):
    #         # if flag == True:
    #         #     print('break')
    #         #     break
    #         # print('start')
    #         now_time = int(time.time())
    #         if now_time - starttime > 30 and half_min_flag == True:
    #             print('---------------------'+str(this_pop.population_pool[0].property[self.property_name]))
    #             half_min_flag = False
    #             print(count)
    #             break
    #
    #         if break_flag == True:
    #             break
    #
    #         if time_count >= self.time_counts:
    #             final_pool = this_pop.population_pool
    #             pickle.dump(final_pool, open('qed_max_pool.pkl', 'wb'))
    #             break
    #
    #         this_record = self.record_pool[0]
    #         this_candidate = self.candidate_pool_list[0]
    #
    #         #num_new_atom = this_pop.population_size
    #         temp_pool = []
    #
    #         #the operation of mutation and crossover create new molecules
    #         if is_crossover == True:
    #             temp_pool = temp_pool + self._crossover(this_pop)
    #         if is_mutate == True:
    #             temp_pool = temp_pool + self._mutate(this_pop)
    #
    #         # for j in range(num_new_atom):
    #         #     index_ = np.random.choice(num_new_atom, 1, replace=False)
    #         #     index = index_[0]
    #         #     mol = self.pop.population_pool[index]
    #         #     mol1 = self.operator_mut.add_funtional_group(mol)
    #         #     temp_pool.append(mol1)
    #         # standard_mol= self.pop.population_pool[0]
    #         # for idx in range(len(temp_pool)):
    #         #     sim = self._calculate_similarity(standard_mol, temp_pool[idx])
    #         #     temp_pool[idx].similarity = sim
    #
    #         total = temp_pool + this_pop.population_pool
    #         total = set(total)
    #         count += len(total)
    #
    #         #total = set(total)
    #         #res = sorted(total, key=lambda x: x.property[self.property_name], reverse=True)
    #         res = sorted(total, key=lambda x: self._sort_function(x), reverse=True)
    #         self.property_record.add_molecule(res[0])
    #
    #         # add molecules generated by crossover and mutation into candidate_pool
    #         res_candidate = res[this_pop.population_size:]
    #         index = np.random.choice(len(res_candidate), 2, replace=False)
    #         index1, index2 = index[0], index[1]
    #         res_candidate[index1].life_time = 0
    #         res_candidate[index2].life_time = 0
    #         this_candidate.add_molecule(res_candidate[index1])
    #         this_candidate.add_molecule(res_candidate[index2])
    #
    #         res_score = [res[idx].property[self.property_name] for idx in range(this_pop.population_size)]
    #         print('mean is : ' + str(np.mean(res_score)) + ' std is : ' + str(np.std(res_score)) + ' max is : ' + str(
    #             np.max(res_score)))
    #
    #         this_pop.population_pool = []
    #         for j in range(this_pop.population_size):
    #             res[j].life_time += 1
    #             this_pop.population_pool.append(res[j])
    #
    #         if multi_level_flag == False:
    #             # this_record.add_record(np.max(res_score))
    #             # if this_record.is_converse():
    #             #     for idx_top in range(3):
    #             #         print(str(res[idx_top].property[self.property_name]) + '-------' + str(res[idx_top].smiles))
    #             #     break_flag = True
    #             pass
    #         # print(len(this_pop.population_pool))
    #         # update the record and process the convergence
    #         if multi_level_flag == True:
    #             this_record.add_record(np.max(res_score))
    #             this_record.print_situation()
    #             if this_record.is_converse():
    #                 multi_mutation_flag = True
    #                 num_of_replace = int(0.25 * this_pop.population_size)
    #                 replace_pool = this_pop.population_pool[-num_of_replace:]
    #                 print(len(replace_pool))
    #
    #                 this_pop.population_pool = [x for x in this_pop.population_pool if x not in replace_pool]
    #                 print('the length of this_pop is : ' + str(len(this_pop.population_pool)))
    #
    #                 extra_mols = this_candidate.extract_molecules(replace_pool)
    #                 this_pop.population_pool = this_pop.population_pool + extra_mols
    #                 print('the length of this_pop is : ' + str(len(this_pop.population_pool)))
    #
    #                 num_of_next = int(0.2 * this_pop.population_size)
    #                 next_pool = this_pop.population_pool[:num_of_next]
    #                 for idx_next in range(len(next_pool)):
    #                     self.candidate_pool_list[1].add_molecule(next_pool[idx_next])
    #
    #                     # flag = True
    #                     # pass
    #                 # else:
    #                 #     print('std is : ')
    #                 #     print(this_record.record_pool)
    #                 # max_pool.append
    #         if multi_mutation_flag == True:
    #             multi_mutation_flag = False
    #
    #             size = int(this_pop.population_size * 0.25)
    #             temp_pool = this_pop.population_pool[:size]
    #             pass


    def train_without_constraint(self):

        population = Population(self.configfile)
        self.configfile.init_file_style = 'pkl'
        self.configfile.init_poplution_file_name = '/home/jeffzhu/aaai_ga/data/randomv2.sdf'
        population._init_population()
        self.configfile.num_mutation_max = 5
        self.configfile.num_mutation_min = 5

        molecule_pool = population.population_pool
        pop_size = len(molecule_pool)

        crossover_op = Crossover(self.configfile)
        mutate_op = Mutation(self.configfile)

        temp_pool = []
        candidate_pool = []
        max_mol = None

        counter = Count()

        log_file_prefix = 'candidate'
        log_file_suffix = '.pkl'

        for iteration in range(100000000):
            #print(iteration)
            temp_pool = []

            for idx in range(int(pop_size * 0.8)):
                index = np.random.randint(0, pop_size, 2)
                mol1, mol2 = molecule_pool[index[0]], molecule_pool[index[1]]
                new_mols = crossover_op.crossover(mol1, mol2)
                temp_pool += new_mols

            for idx in range(pop_size):
                temp_mol = molecule_pool[idx]
                new_mols = mutate_op.mutate(temp_mol)
                temp_pool += new_mols

            total = temp_pool + molecule_pool
            total = set(total)
            total = list(total)
            #print(len(total))

            sorted_total = sorted(total, key=lambda x:x.property[self.property_name], reverse=True)
            molecule_pool = sorted_total[:pop_size]

            value_pool = [molecule_pool[i].property[self.property_name] for i in range(pop_size)]
            print('max value is : ' + str(value_pool[0]) + ' mean value is : ' + str(np.mean(value_pool)) + ' std value is : ' + str(np.std(value_pool)))
            print(str(value_pool[0]) + ' ' + str(value_pool[1]) + ' ' + str(value_pool[2]) + ' ' + str(iteration))
            print(molecule_pool[0].smiles + ' ' + molecule_pool[1].smiles + ' ' + molecule_pool[2].smiles)

            # if len(candidate_pool) > 2000:
            #     candidate_pool = candidate_pool[-2000:]

            # counter.add_record(np.mean(value_pool))
            # if counter.is_converge():
            #     print('convergence-----------------------------')
            #     similarity_pool = [self._smilarity_between_two_smiles(molecule_pool[0].smiles, candidate_pool[i].smiles) for i in range(len(candidate_pool))]
            #     index_sim_pool = np.argsort(similarity_pool)
            #     #print(index_sim_pool)
            #     index_pool = index_sim_pool[:12]
            #
            #     top_pool = molecule_pool[:12]
            #
            #     #'muti-mutate'
            #     for idx in range(12):
            #         print('mutate starting')
            #         self.configfile.num_mutation_max = 8
            #         self.configfile.num_mutation_min = 8
            #         each_pool = []
            #         for idx_1 in range(10):
            #             each_pool += mutate_op.mutate(molecule_pool[idx])
            #         each_pool = sorted(each_pool, key=lambda x: x.property[self.property_name], reverse=True)
            #         if each_pool[0].property[self.property_name] > molecule_pool[idx].property[self.property_name]:
            #             print('find new higher molecule')
            #             molecule_pool[idx] = each_pool[0]
            #
            #     self.configfile.num_mutation_max = 3
            #     self.configfile.num_mutation_min = 3
            #
            #     for idx in range(38, 50):
            #         c_index = index_pool[idx-38]
            #         molecule_pool[idx] = candidate_pool[c_index]
            #     counter.count = 0
            #
            #     temp_pool = candidate_pool
            #     for idx in range(2000):
            #         if idx not in index_pool:
            #             candidate_pool.append(temp_pool[idx])
            #     print('update ok')

    def train_with_constraint(self, property_delta = 0.6, start_index = 0):
        self.configfile.init_poplution_file_name = '/home/jeffzhu/aaai_ga/data/800_mols_for_constraint_v5'
        target_pool = pickle.load(open(self.configfile.init_poplution_file_name, 'rb'))
        self.configfile.init_file_style = 'pkl'
        self.configfile.crossover_mu = 0.75
        self._init_pop_pool()

        self.configfile.num_mutation_max = 3
        self.configfile.num_mutation_min = 3

        # this_pop = self.pop_pool[0]
        # original_pop = this_pop
        # this_mol = original_pop.population_pool[0]
        # original_value = this_mol.property[self.property_name]
        #
        # max_value = original_value
        # max_smiles = this_mol.smiles
        # max_smilarity = None
        #
        # print('original is : '+ str(original_value))
        # standard_mol = this_mol
        #
        # this_value = this_mol.property[self.property_name]
        # #print('this value is : ' + str(this_value))
        # mutate_op = Mutation(self.configfile)
        # crossover_op = Crossover(self.configfile)
        # contraint_recorder = ConstraintRecorder(self.configfile)
        #
        # molecule_pool = []
        # count_op = Count()

        for mol_idx in range(start_index*50, (start_index+1)*50):
            #mol_idx = 0
            this_mol = Molecule(target_pool[mol_idx])
            original_value = this_mol.property[self.property_name]

            max_value = original_value
            max_smiles = this_mol.smiles
            max_smilarity = None

            print('original is : ' + str(original_value))
            standard_mol = this_mol

            mutate_op = Mutation(self.configfile)
            crossover_op = Crossover(self.configfile)

            molecule_pool = []
            for iteration in range(50):
                temp_mat, temp_node_list = crossover_op._bfs_molecule(this_mol)
                temp_mol = adj2mol(temp_node_list, temp_mat, self.configfile.possible_bonds)
                new_mol = Molecule(Chem.MolToSmiles(temp_mol))

                molecule_pool.append(new_mol)


            start_time = int(time.time())
            for iteration in range(100000000):

                temp_pool = []
                for idx in range(50):
                    temp_pool += mutate_op.mutate(molecule_pool[idx])

                total = temp_pool + molecule_pool
                total = set(total)
                total = list(total)

                #print(len(total))

                similarity_pool = [self._smilarity_between_two_smiles(total[idx].smiles, standard_mol.smiles) for idx in range(len(total))]
                sorted_similarity_pool = sorted(similarity_pool, reverse=True)
                # get the max value
                for idx in range(len(similarity_pool)):
                    if similarity_pool[idx] > property_delta and total[idx].property[self.property_name] > max_value:
                        max_value = total[idx].property[self.property_name]
                        max_smiles = total[idx].smiles
                        max_smilarity = similarity_pool[idx]

                # if sorted_similarity_pool[0] < property_delta + 0.05:
                #     loss_pool = similarity_pool
                # else:
                rate = 13 + (max_value - original_value)
                loss_pool = [similarity_pool[idx] * rate + total[idx].property[self.property_name]-original_value for idx in range(len(total))]

                #print(sorted_similarity_pool)
                index_loss_pool = np.argsort(loss_pool)[-50:]

                molecule_pool = []
                for idx in range(50):
                    index_now = index_loss_pool[idx]
                    molecule_pool.append(total[index_now])

                res = sorted(molecule_pool, key=lambda x:x.property[self.property_name], reverse=True)
                print('the maximum is : ' + str(max_value) + ' original is : ' + str(original_value) + ' improvement is : ' + str(max_value-original_value))
                now_time = int(time.time())
                if now_time - start_time > 30 * 60:
                    str_obj = 'index_is: ' + str(mol_idx) + ' original_mol_is: ' + str(standard_mol.smiles) + ' max_mol_is: ' + str(max_smiles) + ' max_sim_is: ' + str(max_smilarity) + ' impro_is: ' + str(max_value-original_value) +'\n'
                    pickle.dump(str_obj, open('test_constraint'+ '_6' + str(start_index) + '.pkl', 'ab'))
                    print(str_obj)
                    print('dump ok')
                    break

    # the constraint_flag decides whether to execute the constraint property experiment
    def train(self, constraint_flag = True):
        if constraint_flag == False:
            self.train_without_constraint()
        else:
            self.train_with_constraint()

class Count():
    def __init__(self,
                 size = 8):
        self.max_size = size
        self.standard = None
        self.count = 0

    def add_record(self, record):
        if self.standard == None:
            self.standard = record
            self.count = 1
        elif self.standard != record:
            self.standard = record
            self.count = 1
        else:
            self.count += 1

    def is_converge(self):
        if self.count > self.max_size:
            return True
        else:
            return False

if __name__ == '__main__':
    config = Configuration()
    trainer = Trainer(config)
    #trainer.train_with_constraint(start_index=15)
    trainer.train_without_constraint()
    # simles1 = 'CCCc1cc(NC(=O)C(=O)N2CC3CCC2C3)n(C)n1'
    # simles2 = 'CCCCC(CCCCCCCCCCCCCC1=NN(C)C(NC(=O)C(=O)N2CCCCC2CCCCCCCCC(Br)CCCCCCC(Br)CCCCCBr)=C1)(SBr)SC(Br)Br'
    # #
    # print(trainer._smilarity_between_two_smiles(simles1, simles2))