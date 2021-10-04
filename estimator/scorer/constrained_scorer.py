import rdkit.Chem as Chem
from rdkit.Chem import *
from rdkit import DataStructs

CONSTRAIN_FACTOR = 10.0
DELTA = 0.2

def constrained_score(mol, old_mol, new_reward, old_reward):
    try:
        curr_fp = AllChem.GetMorganFingerprint(mol, radius=2)
        target_fp = AllChem.GetMorganFingerprint(old_mol, radius=2)
        sim = DataStructs.TanimotoSimilarity(target_fp, curr_fp)
    except Exception as e:
        sim = 0.0

    improvement = new_reward - old_reward
    new_smiles = Chem.MolToSmiles(mol)
    old_smiles = Chem.MolToSmiles(old_mol)
    reward = new_reward - CONSTRAIN_FACTOR * max(0, DELTA - sim)
    fields = '{}, {}, {}, {}, {}\n'.format(new_reward, improvement, sim, new_smiles, old_smiles)
    with open('constrain_mols.csv', 'a') as f:
        f.write(fields)
    return reward
