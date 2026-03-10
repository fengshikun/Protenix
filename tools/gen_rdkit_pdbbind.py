import os
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import re
import sys


PARENT_DIR = "/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected"
N_CONFS = 10


def load_failed_pdbids(error_file: Path) -> set[str]:
    """
    从 error log 中提取 Error processing XXXX: 里的 XXXX
    """
    failed = set()
    pattern = re.compile(r'Error processing (\S+):')

    with open(error_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                failed.add(m.group(1))

    return failed

def load_ligand_preserve_order(lig_mol2, lig_sdf):
    """
    Load ligand molecule while preserving atom order exactly as in the input file.
    """
    mol = None

    
    if lig_mol2.exists():
        # For mol2: disable sanitize first to preserve atom order
        mol = Chem.MolFromMol2File(str(lig_mol2), sanitize=False, removeHs=False)
        # if mol is not None:
        #     Chem.SanitizeMol(mol)
        return mol
    if lig_sdf.exists():
        # SDF preserves atom order naturally; disable removeHs to avoid reordering
        supplier = Chem.SDMolSupplier(str(lig_sdf), removeHs=False, sanitize=False)
        mol = supplier[0]
        # if mol is not None:
        #     Chem.SanitizeMol(mol)
        return mol

    return None



def generate_conformers(mol, num_confs=10):
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=42, clearConfs=True, numThreads=8 # maxAttempts=5000,
    )
    sz = len(allconformers)
    
    for i in range(sz):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
        except:
            continue
    
    if not sz:
        ps = AllChem.ETKDGv2()
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # for conf_id in conf_ids:
    #     AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
    
    return mol  # Returns the molecule with multiple conformers;  mol_rdkit = copy.deepcopy(mol_)


def generate_confs_preserve_order(mol, num_confs=10, remove_hs=False):
    """
    Generate conformers without changing atom order.
    """
    # mol = Chem.AddHs(mol, addCoords=True)  # add hydrogens but preserve order

    mol_rdkit = copy.deepcopy(mol)
    
    mol_rdkit.RemoveAllConformers()
    mol_rdkit = AllChem.AddHs(mol_rdkit)
    # generate_conformer(mol_rdkit)
    
    generate_conformers(mol_rdkit)
    if remove_hs:
        mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
    # mol_rdkit = copy.deepcopy(mol_rdkit)
    return mol_rdkit


def main(error_file: Path | None = None):
    parent = Path(PARENT_DIR)
    
    if error_file is not None:
        failed_pdbids = load_failed_pdbids(error_file)
        print(f"Loaded {len(failed_pdbids)} failed pdbids from {error_file}")

        all_dirs = sorted(
            d for d in parent.iterdir()
            if d.is_dir() and d.name in failed_pdbids
        )
    else:
        all_dirs = sorted(
            d for d in parent.iterdir() if d.is_dir()
        )
    
    # all_dirs = sorted([d for d in parent.iterdir() if d.is_dir()])

    print(f"Total pdb directories: {len(all_dirs)}\n")

    for d in tqdm(all_dirs, desc="Processing folders"):
        pdbid = d.name
        lig_mol2 = d / f"{pdbid}_ligand.mol2"
        lig_sdf  = d / f"{pdbid}_ligand.sdf"
        out_sdf  = d / f"{pdbid}_ligand_rdkit.sdf"
        
        
        # pdbid = '5eg4'
        # d = f'/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected/{pdbid}'
        # lig_mol2 = Path(f"{d}/{pdbid}_ligand.mol2")
        # lig_sdf  = Path(f"{d}/{pdbid}_ligand.sdf")
        # out_sdf  = Path(f"{d}/{pdbid}_ligand_rdkit.sdf")

        # already done
        # if out_sdf.exists():
        #     print(f"{d} already processed, skipping.")
        #     continue

        mol = load_ligand_preserve_order(lig_mol2, lig_sdf)
        if mol is None:
            continue
        
        # mol_conf = generate_confs_preserve_order(mol, N_CONFS)

        try:
            mol_conf = generate_confs_preserve_order(mol, N_CONFS)

            writer = Chem.SDWriter(str(out_sdf))
            for cid in range(mol_conf.GetNumConformers()):
                writer.write(mol_conf, confId=cid)
            writer.close()

        except Exception as e:
            print(f"Error processing {pdbid}: {e}")
            continue


if __name__ == "__main__":
    error_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(error_file)
