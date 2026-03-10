from Bio.PDB import PDBParser
biopython_parser = PDBParser()
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import os
from tqdm import tqdm


def parse_pdb_from_path(path):
    return parse_pdb_structure_from_path(path)[0]


def parse_pdb_structure_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure(os.path.basename(path), path)
        return structure

comp_protein_path = '/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected/13gs/13gs_protein_esmfold_aligned_tr_fix.pdb'
exp_protein_path = '/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected/13gs/13gs_protein_processed_fix.pdb'



experimental_receptor = parse_pdb_from_path(exp_protein_path)
computational_receptor = parse_pdb_from_path(comp_protein_path) if comp_protein_path else None
complex_name = f'{os.path.basename(exp_protein_path)}'

SORTING_DICT = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],  
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],  
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],  
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],  
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"], 
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}


def order_atoms_in_residue(res, atom):
        """
        An order function that sorts atoms of a residue.
        Atoms N, CA, C, O always come first, thereafter the rest of the atoms are sorted according
        to how they appear in the chemical components. Hydrogens come last and are not sorted.
        """

        if atom.name == "OXT":
            return 999
        elif atom.element == "H":
            return 1000

        if res.resname in SORTING_DICT:
            if atom.name in SORTING_DICT[res.resname]:
                return SORTING_DICT[res.resname].index(atom.name)
        else:
            raise Exception("Unknown residue", res.resname)
        raise Exception(f"Could not find atom {atom.name} in {res.resname}")

def _sort_atoms_by_element(_protein):
    for res in _protein.get_residues():
        res.child_list.sort(key=lambda atom: order_atoms_in_residue(res, atom))

def _remove_hs(_protein):
    for res in _protein.get_residues():
        atoms_to_remove = []
        for atom in res:
            if atom.element == 'H':
                atoms_to_remove.append(atom)
        for atom in atoms_to_remove:
            res.detach_child(atom.id)

remove_hs_and_sort = True
if remove_hs_and_sort:
    _remove_hs(experimental_receptor)
    _sort_atoms_by_element(experimental_receptor)

if computational_receptor is not None and remove_hs_and_sort:
    # In the case that we are flexible (or conformer matching) we sort the atoms,
    # so that we can compare them in the inference script
    _remove_hs(computational_receptor)
    _sort_atoms_by_element(computational_receptor)

if computational_receptor is not None and experimental_receptor is not None:
    len_comp = len(list(computational_receptor.get_atoms()))
    len_exp = len(list(experimental_receptor.get_atoms()))
    assert len_comp == len_exp, \
        (f"The length of the experimental structure ({exp_protein_path}, {len_exp}) "
            f"does not match the length of the computational structure ({comp_protein_path}, {len_comp})")

    # Check if we have 100% atom identity (hydrogens were ignored in loading already)
    assert [a.name for a in computational_receptor.get_atoms()] == [a.name for a in experimental_receptor.get_atoms()], \
        "The proteins do not have 100% sequence identity (excluding hydrogens)"


from Bio.PDB import PDBIO

io = PDBIO()

# 写 experimental
exp_out_path = exp_protein_path.replace(".pdb", "_processed_sorted.pdb")
io.set_structure(experimental_receptor)
io.save(exp_out_path)

# 写 computational
if computational_receptor is not None:
    comp_out_path = comp_protein_path.replace(".pdb", "_processed_sorted.pdb")
    io.set_structure(computational_receptor)
    io.save(comp_out_path)

print("Saved:")
print("  Experimental receptor →", exp_out_path)
if computational_receptor is not None:
    print("  Computational receptor →", comp_out_path)



ROOT = "/vepfs-mlp2/mlp-public/shikunfeng/Datas/posebusters"

# =======================================================
# 主处理函数（对一个 target）
# =======================================================
def process_one_complex(folder):
    subdir = os.path.join(ROOT, folder)

    exp = None
    comp = None

    exp_path = None
    comp_path = None

    # 自动找文件
    for f in os.listdir(subdir):
        if f.endswith("_protein_fixed.pdb"):
            exp_path = os.path.join(subdir, f)
        if f.endswith("_esmfold_fixed.pdb"):
            comp_path = os.path.join(subdir, f)

    if exp_path is None:
        return f"{folder}: ⚠️ No experimental protein found."

    # 读取 experimental
    experimental = parse_pdb_structure_from_path(exp_path)[0]

    # 删除 H + 排序
    _remove_hs(experimental)
    _sort_atoms_by_element(experimental)

    # 保存 experimental
    io = PDBIO()
    out_exp = exp_path.replace(".pdb", "_processed_sorted.pdb")
    io.set_structure(experimental)
    io.save(out_exp)

    msg = f"{folder}: ✓ saved experimental"

    # 如果没有 computational 就结束
    if comp_path is None:
        return msg + " (no computational)"

    # 读取 computational
    computational = parse_pdb_structure_from_path(comp_path)[0]

    _remove_hs(computational)
    _sort_atoms_by_element(computational)

    # 保存 comp
    out_comp = comp_path.replace(".pdb", "_processed_sorted.pdb")
    io.set_structure(computational)
    io.save(out_comp)

    # 数量一致性检查
    len_exp = len(list(experimental.get_atoms()))
    len_comp = len(list(computational.get_atoms()))

    if len_exp != len_comp:
        return f"{folder}: ❌ Atom count mismatch EXP={len_exp}, COMP={len_comp}"

    return f"{folder}: ✓ OK"


# =======================================================
# 主遍历
# =======================================================
if __name__ == "__main__":
    subfolders = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

    results = []

    for folder in tqdm(subfolders, desc="Processing complexes"):
        try:
            msg = process_one_complex(folder)
            print(msg)
        except Exception as e:
            msg = f"{folder}: ERROR {str(e)}"
        results.append(msg)

    # print("\n==== Summary ====")
    # for r in results:
    #     print(r)