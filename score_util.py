import networkx as nx
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops
from rdkit import rdBase
import rdkit.Chem as Chem

import sascorer

def calc_score(mol):
    logP_mean = 2.457  # np.mean(logP_values)
    logP_std = 1.434  # np.std(logP_values)
    SA_mean = -3.053  # np.mean(SA_scores)
    SA_std = 0.834  # np.std(SA_scores)
    cycle_mean = -0.048  # np.mean(cycle_scores)
    cycle_std = 0.287  # np.std(cycle_scores)

    molecule = mol
    smiles = Chem.MolToSmiles(molecule)
    # if Descriptors.MolWt(molecule) > 500:
    #     return -1e10
    # if len(smiles) > 81:
    #     return -1e10
    # if mol.GetNumAtoms() > 200:
    #     return -1e10

    current_log_P_value = Descriptors.MolLogP(molecule)
    current_SA_score = -sascorer.calculateScore(molecule)
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(molecule)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    current_cycle_score = -cycle_length

    current_SA_score_normalized = (current_SA_score - SA_mean) / SA_std
    current_log_P_value_normalized = (current_log_P_value - logP_mean) / logP_std
    current_cycle_score_normalized = (current_cycle_score - cycle_mean) / cycle_std

    score = (current_SA_score_normalized
             + current_log_P_value_normalized
             + current_cycle_score_normalized)
    return score

