# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py

import math
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import networkx as nx

from ...common.chem import standardize_smiles
from . import sa_scorer, kinase_scorer, sim_scorer#, drd2_scorer, chemprop_scorer
from .adtgpu.get_reward import get_dock_score

CONSTRAIN_FACTOR = 100.0
DELTA = 0.6
class Args:
    def __init__(self, obabel_path='', adt_path='', receptor_file='', run_id=''):
        self.obabel_path = obabel_path
        self.adt_path = adt_path
        self.receptor_file = receptor_file
        self.run_id = run_id
ARGS = Args(run_id='006')

### get scores
def get_scores(objective, mols, old_mols, init_mols):
    mols = [standardize_smiles(mol) for mol in mols]
    mols_valid = [mol for mol in mols if mol is not None]
    old_mols_valid = [old_mol for mol, old_mol in zip(mols, old_mols) if mol is not None]
    init_mols_valid = [init_mol for mol, init_mol in zip(mols, init_mols) if mol is not None]
    
    if objective == 'drd2':
        scores = drd2_scorer.get_scores(mols_valid)
    #elif objective == 'dock':
    #    scores = get_dock_score(mols_valid, ARGS)
    #    scores = [s / 40.0 for s in scores]
    elif objective == 'jnk3' or objective == 'gsk3b':
        scores = kinase_scorer.get_scores(objective, mols_valid)
    elif objective.startswith('chemprop'):
        scores = chemprop_scorer.get_scores(objective, mols_valid)
    else: scores = [get_score(objective, mol, old_mol, init_mol)
                    for mol, old_mol, init_mol in zip(mols_valid, old_mols_valid, init_mols_valid)]
        
    scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
    return scores

def get_score(objective, mol, old_mol, init_mols):
    try:
        if objective == 'sim': 
            return sim_scorer.sim_score(mol, init_mols, CONSTRAIN_FACTOR, DELTA) / 40.0
        elif objective == 'sim_actual': 
            return sim_scorer.sim_actual(mol, init_mols)
        elif objective == 'dock':
            return get_dock_score(mol, ARGS)[0] / 40.0
        elif objective == 'qed': 
            return QED.qed(mol)
        elif objective == 'sa': 
            x = sa_scorer.calculateScore(mol)
            return (10. - x) / 9. # normalized to [0, 1]
        elif objective == 'mw': # molecular weight
            return mw(mol)
        elif objective == 'logp': # real number
            return Descriptors.MolLogP(mol)
        elif objective == 'penalized_logp':
            return penalized_logp(mol)
        elif 'rand' in objective:
            raise NotImplementedError
            # return rand_scorer.get_score(objective, mol)
        else: raise NotImplementedError
    except ValueError:
        return 0.

    
### molecular properties
def mw(mol):
    '''
    molecular weight estimation from qed
    '''
    x = Descriptors.MolWt(mol)
    a, b, c, d, e, f = 2.817, 392.575, 290.749, 2.420, 49.223, 65.371
    g = math.exp(-(x - c + d/2) / e)
    h = math.exp(-(x - c - d/2) / f)
    x = a + b / (1 + g) * (1 - 1 / (1 + h))
    return x / 104.981
    
def penalized_logp(mol):
    # Modified from https://github.com/bowenliu16/rl_graph_generation
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sa_scorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle
