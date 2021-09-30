from rdkit.Chem import *
from rdkit import DataStructs

CONSTRAIN_FACTOR = 10.0
DELTA = 0.2

def constrained_score(mol, old_mol, main_reward):
    try:
        curr_fp = AllChem.GetMorganFingerprint(mol, radius=2)
        target_fp = AllChem.GetMorganFingerprint(old_mol, radius=2)
        sim = DataStructs.TanimotoSimilarity(target_fp, curr_fp)
    except Exception as e:
        sim = 0.0

    reward = main_reward - CONSTRAIN_FACTOR * max(0, DELTA - sim)
    return reward
