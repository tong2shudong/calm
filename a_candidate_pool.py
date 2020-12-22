import numpy as np
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
import copy

class CandidatePool():
    def __init__(self,
                 candidate_pool_size=50):
        self.candidate_pool_size = candidate_pool_size
        self.pool = []

    def _calculate_softmax(self, pool_life_time):
        exp_sum = 0
        for i in pool_life_time:
            exp_sum += np.exp(i)
        res = [i/exp_sum for i in pool_life_time]
        return res

    def get_length(self):
        return len(self.pool)

    def add_molecule(self, molecule):
        pool_length = len(self.pool)
        if pool_length < self.candidate_pool_size:
            self.pool.append(molecule)
        elif pool_length == self.candidate_pool_size:
            pool_life_time = [i.pool_life_time for i in self.pool if i.prior_flag == False]
            sorted_life_time = np.argsort(pool_life_time)
            index = sorted_life_time[0]
            self.pool.pop(index)
            molecule.pool_life_time = 0
            self.pool.append(molecule)

    def _smilarity_between_two_smiles(self, smi1, smi2):
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)

        vec1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 4, nBits=512)
        vec2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 4, nBits=512)

        tani = DataStructs.TanimotoSimilarity(vec1, vec2)
        return tani

    def _calculate_similarity(self, molecule):
        similarity = []
        for i in self.pool:
            similarity.append(self._smilarity_between_two_smiles(molecule.smiles, i.smiles))
        return similarity

    def extract_molecules(self, molecule_list):
        res_mols = []
        length = len(molecule_list)
        if length > len(self.pool):
            res_mols = self.pool
            return res_mols
        else:
            for mol in molecule_list:
                similarity = self._calculate_similarity(mol)
                max_index = np.argmax(similarity)
                self.pool[max_index].pool_life_time = 0
                res_mols.append(self.pool[max_index])
                self.pool.pop(max_index)
            return res_mols

    def unload_molecules(self, num_molecules):
        pool = self.pool
        pool = sorted(pool, key=lambda x: x.pool_life_time, reverse=True)
        temp_mols = pool[:num_molecules]
        for i in range(num_molecules):
            pool.pop(0)
        return temp_mols

    def update_candidate_pool(self):
        for molecule in self.pool:
            molecule.pool_life_time += 1

from a_molecule import *

if __name__ == '__main__':
    candidate_pool = CandidatePool()

    smiles_pool = ['C']

    for i in range(10):
        smiles_pool.append(smiles_pool[-1]+'C')

    for i in range(10):
        candidate_pool.add_molecule(Molecule(smiles_pool[i]))

    standard_mol = Molecule(smiles_pool[-1])
    print(len(candidate_pool.pool))
    print(candidate_pool.extract_molecules([standard_mol])[0].smiles)