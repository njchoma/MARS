from rdkit.Chem import AllChem
from rdkit import DataStructs

def sim_score(mol, old_mol, constrain_factor=100.0, delta=0.2):
    try:
        curr_fp = AllChem.GetMorganFingerprint(mol, radius=2)
        target_fp = AllChem.GetMorganFingerprint(old_mol, radius=2)
        sim = DataStructs.TanimotoSimilarity(target_fp, curr_fp)
    except Exception as e:
        sim = 0.0

    score = constrain_factor * (1 - max(0, delta - sim))
    return score

def sim_actual(mol, old_mol):
    try:
        curr_fp = AllChem.GetMorganFingerprint(mol, radius=2)
        target_fp = AllChem.GetMorganFingerprint(old_mol, radius=2)
        sim = DataStructs.TanimotoSimilarity(target_fp, curr_fp)
    except Exception as e:
        sim = 0.0

    return sim
