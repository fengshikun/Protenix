#!/usr/bin/env python3
"""
merge_to_full_mmcif.py

Merge a protein PDB and a ligand MOL2, produce a complete mmCIF with:
 - atom_site (from MMCIFIO)
 - _atom_site.label_entity_id (inserted if missing)
 - _entity, _struct_asym
 - _entity_poly, _entity_poly_seq (for polymer entities)
 - simple _pdbx_struct_assembly and _pdbx_struct_assembly_gen

This output is suitable for get_poly_res_names(), entity_poly_type(), num_assembly_poly_chains(), etc.
"""
import os
import string
import tempfile
import shlex
from io import StringIO
from typing import Dict, Tuple, List
from tqdm import tqdm

from Bio.PDB import PDBParser, MMCIFIO
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from rdkit import Chem
import gemmi


# -------------------------
# Helpers
# -------------------------
def three_to_one(res3: str) -> str:
    if not res3:
        return "X"
    key = res3.capitalize()
    return protein_letters_3to1.get(key, "X")

def detect_chain_is_polymer(resnames: List[str]) -> bool:
    """
    Heuristic: treat chain as polymer (polypeptide(L)) if:
      - chain length >= 2
      - proportion of standard AAs (three->one known) > 0.6
    """
    if len(resnames) < 2:
        return False
    mapped = [three_to_one(r) for r in resnames]
    known = sum(1 for x in mapped if x != "X")
    return (known / len(mapped)) >= 0.6

def unique_chain_id(existing: List[str]) -> str:
    for c in string.ascii_uppercase:
        if c not in existing:
            return c
    raise RuntimeError("No available uppercase chain id")

def insert_label_entity_id_in_atom_site(cif_text: str, chain_to_entity: Dict[str, str]) -> str:
    """
    Ensure _atom_site.label_entity_id exists and update its values according to chain_to_entity.
    If the column exists, overwrite the old values (including '?').
    If not, insert the column after label_asym_id or auth_asym_id.
    """
    lines = cif_text.splitlines()
    out_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        out_lines.append(line)

        if line.strip() == "loop_":
            j = i + 1
            headers = []
            while j < n and lines[j].strip().startswith("_atom_site."):
                headers.append(lines[j].strip())
                j += 1

            if not headers:
                i += 1
                continue

            # atom_site header detected
            header_names = [h.split()[0] for h in headers]
            has_entity_col = "_atom_site.label_entity_id" in header_names

            # find data block
            data_lines = []
            k = j
            while (
                k < n
                and lines[k].strip()
                and not lines[k].strip().startswith("loop_")
                and not lines[k].strip().startswith("_")
            ):
                data_lines.append(lines[k])
                k += 1

            # Determine asym_col
            if "_atom_site.label_asym_id" in header_names:
                asym_col = header_names.index("_atom_site.label_asym_id")
            elif "_atom_site.auth_asym_id" in header_names:
                asym_col = header_names.index("_atom_site.auth_asym_id")
            else:
                asym_col = None

            # If entity column doesn't exist → insert it
            if not has_entity_col:
                out_lines.pop()   # remove the last loop_ or previous append
                out_lines.append("loop_")

                insert_idx = None
                if "_atom_site.label_asym_id" in header_names:
                    insert_idx = header_names.index("_atom_site.label_asym_id") + 1
                elif "_atom_site.auth_asym_id" in header_names:
                    insert_idx = header_names.index("_atom_site.auth_asym_id") + 1
                else:
                    insert_idx = len(headers)

                # rebuild header row
                new_header_names = header_names[:insert_idx] + ["_atom_site.label_entity_id"] + header_names[insert_idx:]

                for h in new_header_names:
                    out_lines.append(h)

                entity_col = insert_idx  # new column index

            else:
                # entity_id column exists
                entity_col = header_names.index("_atom_site.label_entity_id")
                # rebuild header block exactly once
                for h in headers:
                    out_lines.append(h)

            # Process data lines
            for data in data_lines:
                try:
                    tokens = shlex.split(data)
                except:
                    tokens = data.split()

                if asym_col is None or asym_col >= len(tokens):
                    ent_id = "?"
                else:
                    chain_id = tokens[asym_col].strip("'\"")
                    ent_id = chain_to_entity.get(chain_id, "?")

                # If entity column exists → overwrite
                if has_entity_col:
                    if entity_col < len(tokens):
                        tokens[entity_col] = ent_id
                    else:
                        tokens.append(ent_id)
                else:
                    # Insert new entity column
                    tokens.insert(entity_col, ent_id)

                rebuilt = []
                for t in tokens:
                    if " " in t or t == "":
                        rebuilt.append("'" + t + "'")
                    else:
                        rebuilt.append(t)
                out_lines.append(" ".join(rebuilt))

            i = k - 1

        i += 1

    return "\n".join(out_lines)


# parse mmCIF loop header and data and insert a new header+column for label_entity_id
def insert_label_entity_id_in_atom_site2(cif_text: str, chain_to_entity: Dict[str, str]) -> str:
    """
    Find the atom_site loop in cif_text, and if _atom_site.label_entity_id is missing,
    insert it after _atom_site.label_asym_id (or auth_asym_id if label_asym_id missing).
    For each atom_site data row, insert the corresponding entity id based on asym_id.
    """
    lines = cif_text.splitlines()
    out_lines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        out_lines.append(line)
        # detect atom_site loop start
        if line.strip() == "loop_":
            # look ahead to see if this loop is atom_site (check following header lines)
            j = i + 1
            headers = []
            while j < n and lines[j].strip().startswith("_atom_site."):
                headers.append(lines[j].strip())
                j += 1
            if headers and any(h.startswith("_atom_site.") for h in headers):
                # This is an atom_site loop. Find data start index j
                data_start = j
                # gather data lines until next blank or next loop_ or next category line starting with "_"
                k = data_start
                data_lines = []
                while k < n and lines[k].strip() != "" and not lines[k].strip().startswith("loop_") and not lines[k].strip().startswith("_"):
                    data_lines.append(lines[k])
                    k += 1

                # Check whether label_entity_id header exists
                if any(h.split()[0] == "_atom_site.label_entity_id" for h in headers):
                    # already present — nothing to insert, just advance
                    # append the remaining data lines as-is (they are already in out_lines via out_lines append earlier)
                    # but we've already appended line at i; need to append headers and data lines as we pass them in loop.
                    # To avoid duplication, skip ahead to k-1
                    # (we already appended the first "loop_" line)
                    # now we need to append the header lines and data lines into out_lines and move i forward
                    # but we've already appended the first header line in out_lines, so we need to append the rest headers
                    # To make logic simpler, rebuild: rewind out_lines back to before i and re-add the block properly.
                    # Safer approach: do nothing special here — just continue normal iteration (we already appended a single line).
                    pass

                else:
                    # Insert a new header into headers list after _atom_site.label_asym_id or auth_asym_id
                    header_names = [h.split()[0] for h in headers]
                    # determine insertion index
                    insert_after = None
                    if "_atom_site.label_asym_id" in header_names:
                        insert_after = header_names.index("_atom_site.label_asym_id")
                    elif "_atom_site.auth_asym_id" in header_names:
                        insert_after = header_names.index("_atom_site.auth_asym_id")
                    else:
                        # fallback: append at end
                        insert_after = len(headers) - 1

                    new_header = "_atom_site.label_entity_id"
                    # rebuild header block in out_lines (we already added "loop_" earlier)
                    # remove the header lines already appended to out_lines (we appended only the "loop_" line so far),
                    # now append the full header block with inserted header
                    # remove the last appended line (the "loop_") to reconstruct correctly
                    out_lines.pop()  # removed the loop_ line appended above
                    out_lines.append("loop_")
                    for idx, h in enumerate(headers):
                        out_lines.append(h)
                        if idx == insert_after:
                            out_lines.append(new_header)
                    # now process data lines: parse each line tokens and insert entity id token
                    # find which header tells chain id
                    header_names = [h.split()[0] for h in headers]
                    if "_atom_site.label_asym_id" in header_names:
                        asym_col = header_names.index("_atom_site.label_asym_id")
                    elif "_atom_site.auth_asym_id" in header_names:
                        asym_col = header_names.index("_atom_site.auth_asym_id")
                    else:
                        asym_col = None

                    for data in data_lines:
                        # split respecting quotes
                        try:
                            tokens = shlex.split(data)
                        except Exception:
                            # fallback naive split
                            tokens = data.split()
                        # If number tokens < headers, maybe multi-line; best-effort: skip
                        if asym_col is None or asym_col >= len(tokens):
                            # cannot determine chain -> insert '?' as placeholder
                            ent_token = "?"
                        else:
                            chain_token = tokens[asym_col]
                            # remove surrounding quotes if any
                            if chain_token.startswith("'") and chain_token.endswith("'"):
                                chain_token_unq = chain_token[1:-1]
                            elif chain_token.startswith('"') and chain_token.endswith('"'):
                                chain_token_unq = chain_token[1:-1]
                            else:
                                chain_token_unq = chain_token
                            ent_id = chain_to_entity.get(chain_token_unq, "?")
                            ent_token = ent_id
                        # insert ent_token after asym_col index (i.e., position asym_col+1)
                        if len(tokens) >= asym_col + 1:
                            tokens.insert(asym_col + 1, ent_token)
                        else:
                            tokens.append(ent_token)
                        # reconstruct line
                        # ensure tokens with spaces are quoted (rare)
                        rebuilt = []
                        for t in tokens:
                            if " " in str(t) or t == "":
                                rebuilt.append("'" + str(t) + "'")
                            else:
                                rebuilt.append(str(t))
                        out_lines.append(" ".join(rebuilt))
                    # advance i to k-1 (we've consumed headers and data)
                    i = k - 1
        i += 1

    return "\n".join(out_lines)


def fix_cif_entity_id(cif_text: str, asym_to_entity: dict[str, str]) -> str:
    lines = cif_text.splitlines()
    out_lines = []

    in_atom_site_loop = False
    atom_site_columns = []
    col_asym = -1
    col_entity = -1

    for line in lines:
        stripped = line.strip()

        # 检查进入 atom_site 表
        if stripped.startswith("loop_"):
            in_atom_site_loop = False
            atom_site_columns = []
            out_lines.append(line)
            continue

        # 收集 atom_site 列定义
        if stripped.startswith("_atom_site."):
            atom_site_columns.append(stripped)
            out_lines.append(line)

            # 记录列位置
            # if stripped == "_atom_site.label_asym_id":
            if stripped == "_atom_site.auth_asym_id":
                col_asym = len(atom_site_columns) - 1
            if stripped == "_atom_site.label_entity_id":
                col_entity = len(atom_site_columns) - 1

            continue

        # 如果已经收集到 atom_site 列，且下一行不是以 "_" 开头，说明进入数据行
        if atom_site_columns and not stripped.startswith("_"):
            in_atom_site_loop = True

        # 处理 atom_site 的数据行
        if in_atom_site_loop:
            # 将一行拆成列（mmCIF 使用空格分隔）
            parts = line.split()

            # 如果长度不一致，不处理
            if len(parts) != len(atom_site_columns):
                out_lines.append(line)
                continue

            asym = parts[col_asym]
            if asym in asym_to_entity:
                parts[col_entity] = asym_to_entity[asym]
            else:
                parts[col_entity] = "?"  # 或者保持原值

            out_lines.append(" ".join(parts))
            continue

        # 其它行原样输出
        out_lines.append(line)

    return "\n".join(out_lines)

def build_entity_blocks(structure) -> Tuple[str, Dict[str, str]]:
    """
    Construct entity, struct_asym, entity_poly, entity_poly_seq, assembly text blocks.
    Returns (text_to_append, chain_to_entity mapping)
    """
    # 1) build chain -> resname list
    chain_to_resnames = {}
    for chain in structure[0]:
        resnames = []
        for res in chain.get_residues():
            rn = res.resname.strip() if getattr(res, "resname", None) else ""
            if rn:
                resnames.append(rn)
        chain_to_resnames[chain.id] = resnames

    # 2) map sequences to entity ids (same sequence => same entity)
    seq_to_eid = {}
    chain_to_entity = {}
    next_eid = 1
    for ch, seq in chain_to_resnames.items():
        key = tuple(seq)
        if key not in seq_to_eid:
            seq_to_eid[key] = str(next_eid)
            next_eid += 1
        chain_to_entity[ch] = seq_to_eid[key]

    # Build text blocks
    parts = []

    # _entity
    lines = ["loop_", "_entity.id", "_entity.type", "_entity.pdbx_description"]
    # We must list each entity only once (iterate seq_to_eid)
    # Decide entity.type based on is polymer
    for seq_tuple, eid in seq_to_eid.items():
        seq_list = list(seq_tuple)
        if detect_chain_is_polymer(seq_list):
            etype = "polymer"
            desc = f"polypeptide_entity_{eid}"
        else:
            etype = "non-polymer"
            desc = f"nonpoly_entity_{eid}"
        lines.append(f"{eid} {etype} '{desc}'")
    parts.append("\n".join(lines))

    # _struct_asym: chain -> entity
    lines = ["loop_", "_struct_asym.id", "_struct_asym.entity_id"]
    for ch, eid in chain_to_entity.items():
        lines.append(f"{ch} {eid}")
    parts.append("\n".join(lines))

    # _entity_poly (for polymer entities)
    lines = ["loop_", "_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.pdbx_seq_one_letter_code"]
    for seq_tuple, eid in seq_to_eid.items():
        seq_list = list(seq_tuple)
        if detect_chain_is_polymer(seq_list):
            one_letter = "".join(three_to_one(r) for r in seq_list)
            # assume L stereochemistry for normal amino acids
            lines.append(f"{eid} polypeptide(L) '{one_letter}'")
    parts.append("\n".join(lines))

    # _entity_poly_seq
    lines = ["loop_", "_entity_poly_seq.entity_id", "_entity_poly_seq.num", "_entity_poly_seq.mon_id"]
    for seq_tuple, eid in seq_to_eid.items():
        seq_list = list(seq_tuple)
        if detect_chain_is_polymer(seq_list):
            for idx, mono in enumerate(seq_list, start=1):
                lines.append(f"{eid} {idx} {mono}")
    parts.append("\n".join(lines))

    # simple assembly: define assembly 1 as containing all chains
    lines = ["loop_", "_pdbx_struct_assembly.id", "_pdbx_struct_assembly.details", "_pdbx_struct_assembly.oligomeric_count"]
    oligomeric_count = len(chain_to_entity)  # number of chain instances
    lines.append(f"1 'generated_by_merge_script' {oligomeric_count}")
    parts.append("\n".join(lines))

    # assembly_gen: list all chains in asym_id_list
    asym_list = ",".join(chain_to_resnames.keys())
    lines = ["loop_", "_pdbx_struct_assembly_gen.assembly_id", "_pdbx_struct_assembly_gen.oper_expression", "_pdbx_struct_assembly_gen.asym_id_list"]
    lines.append(f"1 1 {asym_list}")
    parts.append("\n".join(lines))

    text_to_append = "\n\n".join(parts)
    return text_to_append, chain_to_entity


def load_ligand_preserve_order(lig_mol2, lig_sdf):
    """
    Load ligand molecule while preserving atom order exactly as in the input file.
    """
    mol = None

    if lig_sdf.exists():
        # SDF preserves atom order naturally; disable removeHs to avoid reordering
        supplier = Chem.SDMolSupplier(str(lig_sdf), removeHs=False, sanitize=False)
        mol = supplier[0]
        # if mol is not None:
        #     Chem.SanitizeMol(mol)
        return mol

    if lig_mol2.exists():
        # For mol2: disable sanitize first to preserve atom order
        mol = Chem.MolFromMol2File(str(lig_mol2), sanitize=False, removeHs=False)
        # if mol is not None:
        #     Chem.SanitizeMol(mol)
        return mol

    return None


# -------------------------
# Main pipeline
# -------------------------
def merge_protein_and_ligand_to_full_mmcif(protein_pdb: str, ligand_mol2: str,  lig_sdf: str, output_cif: str, ligand_chain_hint: str = "L", apo_protein_pdb_path: str = "", rdkit_mol_path: str = ""):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_pdb)

    # read ligand with RDKit and convert to PDB block
    ligand = load_ligand_preserve_order(Path(ligand_mol2), Path(lig_sdf))
    # ligand = Chem.MolFromMol2File(ligand_mol2, removeHs=False)
    if ligand is None:
        raise RuntimeError("RDKit cannot parse ligand mol2 file")
    ligand_pdb = Chem.MolToPDBBlock(ligand)
    ligand_struct = parser.get_structure("lig", StringIO(ligand_pdb))

    model = structure[0]
    lig_model = ligand_struct[0]
    ligand_chains = list(lig_model.get_chains())
    if not ligand_chains:
        raise RuntimeError("No chain found in ligand conversion")
    ligand_chain = ligand_chains[0]

    existing = [c.id for c in model]
    new_chain = ligand_chain_hint
    if new_chain in existing:
        new_chain = unique_chain_id(existing)
    ligand_chain.id = new_chain
    model.add(ligand_chain)

    # 1) Write a base mmCIF to a temp file using MMCIFIO
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cif")
    tmp_name = tmp.name
    tmp.close()
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(tmp_name)

    # 2) Read temp mmCIF text
    with open(tmp_name, "r") as f:
        cif_text = f.read()

    # 3) Build entity blocks & chain->entity map
    append_text, chain_to_entity = build_entity_blocks(structure)

    # 4) Insert label_entity_id into atom_site (if missing)
    cif_with_entity_col = fix_cif_entity_id(cif_text, chain_to_entity)

    # 5) Append entity blocks (avoid duplicate insertion if already present)
    final_text = cif_with_entity_col
    # add blocks only if not already present
    for block_snippet in append_text.split("\n\n"):
        if block_snippet.strip() and block_snippet not in final_text:
            final_text += "\n\n" + block_snippet


    # add the custom fields for apo protein and rdkit mol path
    
    path_text = f"loop_\n_my_category.rdkit_ligand\n_my_category.apo_protein\n{rdkit_mol_path}\n{apo_protein_pdb_path}\n"
    final_text += "\n\n" + path_text
    

    # 6) Write final CIF
    with open(output_cif, "w") as f:
        f.write(final_text)

    # cleanup
    try:
        os.unlink(tmp_name)
    except Exception:
        pass

    print(f"Full mmCIF written to: {output_cif}")
    return output_cif
# -------------------------
# CLI example usage
# -------------------------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Merge PDB + MOL2 => full mmCIF with entity/poly info.")
#     # parser.add_argument("--pdb", required=False, default="/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected/13gs/13gs_protein_processed_fix.pdb", help="Protein PDB file")
#     parser.add_argument("--pdb", required=False, default="/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected/13gs/13gs_protein_processed_fix_processed_sorted.pdb", help="Protein PDB file")
#     parser.add_argument("--mol2", required=False, default="/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected/13gs/13gs_ligand.mol2", help="Ligand MOL2 file")
#     parser.add_argument("--out", required=False, default="complex.full.cif", help="Output mmCIF path")
#     parser.add_argument("--ligchain", required=False, default="L", help="Preferred ligand chain id (default L)")
#     args = parser.parse_args()

#     merge_protein_and_ligand_to_full_mmcif(args.pdb, args.mol2, args.out, args.ligchain)




root_dir = '/vepfs-mlp2/mlp-public/shikunfeng/Datas/posebusters/'

from pathlib import Path

# 假定你已经定义了 merge_protein_and_ligand_to_full_mmcif
# 如果函数在另一个文件，请 from xxx import merge_protein_and_ligand_to_full_mmcif

def check_folder_valid(folder_path, tag):
    """
    判断一个子文件夹是否包含全部 6 个必要文件
    """
    required_files = [
        f"{tag}_ligand.sdf",
        f"{tag}_ligand_start_conf.sdf",
        f"{tag}_protein_fixed_processed_sorted.pdb",
        f"{tag}_esmfold_fixed_processed_sorted.pdb",
    ]
    for rf in required_files:
        if not os.path.isfile(os.path.join(folder_path, rf)):
            return False
    return True


def batch_convert_to_mmcif(root_dir, chain_hint="L"):
    """
    遍历 root_dir 下全部子文件夹：
    - 判断是否满足要求
    - 若满足，生成 mmCIF
    """
    root = Path(root_dir)
    subfolders = sorted([p for p in root.iterdir() if p.is_dir()])

    print(f"找到 {len(subfolders)} 个子文件夹.\n")

    for folder in tqdm(subfolders, desc="Processing folders"):
        tag = folder.name  # 如 5opc

        if not check_folder_valid(folder, tag):
            continue   # 跳过不完整的文件夹

        # 构造文件路径
        ligand_sdf = folder / f"{tag}_ligand.sdf"
        ligand_rdkit = folder / f"{tag}_ligand_start_conf.sdf"
        protein_processed = folder / f"{tag}_protein_fixed_processed_sorted.pdb"
        apo_protein = folder / f"{tag}_esmfold_fixed_processed_sorted.pdb"

        # 输出 mmCIF
        output_cif = folder / f"{tag}_merged.cif"
        output_cif2 = folder / f"{tag}_merged_fake.cif"

        # 调用你的合并函数（你之前的脚本已经实现）
        try:
            merge_protein_and_ligand_to_full_mmcif(
                protein_pdb=str(protein_processed),
                ligand_mol2=str(ligand_sdf),
                lig_sdf=str(ligand_sdf),
                output_cif=str(output_cif),
                ligand_chain_hint=chain_hint,
                apo_protein_pdb_path=str(apo_protein),# 如需要可设置
                rdkit_mol_path=str(ligand_rdkit)
            )
            # merge_protein_and_ligand_to_full_mmcif(
            #     protein_pdb=str(apo_protein),
            #     ligand_mol2=str(ligand_sdf),
            #     lig_sdf=str(ligand_sdf),
            #     output_cif=str(output_cif2),
            #     ligand_chain_hint=chain_hint,
            #     apo_protein_pdb_path=str(apo_protein),# 如需要可设置
            #     rdkit_mol_path=str(ligand_rdkit)
            # )
        except Exception as e:
            print(f"处理文件夹 {folder} 时出错: {e}")
            continue


if __name__ == "__main__":
    root_dir = "/vepfs-mlp2/mlp-public/shikunfeng/Datas/posebusters/"
    batch_convert_to_mmcif(root_dir)