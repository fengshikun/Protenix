import os
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

PDB_DIR = "/vepfs-mlp2/mlp-public/shikunfeng/Datas/PDBBIND_atomCorrected"
biopython_parser = PDBParser(QUIET=True)


def three_to_one(res3: str) -> str:
    return protein_letters_3to1.get(res3.capitalize(), "X")


def get_chain_sequences(pdb_path):
    """Return dict: chain_id -> sequence"""
    structure = biopython_parser.get_structure('x', pdb_path)[0]
    chain_seqs = {}

    for chain in structure:
        seq = ""
        for residue in chain:
            if residue.get_resname() == "HOH":
                continue

            atoms = {atom.name for atom in residue}
            if not {"CA", "N", "C"}.issubset(atoms):
                continue  # not a real amino acid residue

            seq += three_to_one(residue.get_resname())

        chain_seqs[chain.id] = seq

    return chain_seqs


def assign_entity_ids(chain_seqs):
    """Group chains by identical sequence → assign entity_id"""
    entity_map = {}     # chain → entity_id
    entities = {}       # entity_id → seq
    next_eid = 1

    for chain_id, seq in chain_seqs.items():
        found = False
        for eid, eseq in entities.items():
            if seq == eseq:
                entity_map[chain_id] = eid
                found = True
                break
        if not found:
            entities[next_eid] = seq
            entity_map[chain_id] = next_eid
            next_eid += 1

    return entities, entity_map


print("Scanning directory:", PDB_DIR)

results = []   # store (pdb_id, num_chains, num_entities, chain_entity_map)

for folder in sorted(os.listdir(PDB_DIR)):
    folder_path = os.path.join(PDB_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    # Look for protein PDB file
    pdb_file = None
    for file in os.listdir(folder_path):
        if file.endswith("_protein_processed_fix.pdb"):
            pdb_file = os.path.join(folder_path, file)
            break
        if file.endswith("_protein_esmfold_aligned_tr_fix.pdb"):
            pdb_file = os.path.join(folder_path, file)
            break

    if pdb_file is None:
        # skip if no matching protein pdb found
        continue

    try:
        chain_seqs = get_chain_sequences(pdb_file)
        entities, chain2entity = assign_entity_ids(chain_seqs)
        if len(entities) > 1:
            print(f"[MULTI-ENTITY] PDB {folder}: {len(chain_seqs)} chains, {len(entities)} entities, mapping = {chain2entity}")
    except Exception as e:
        print(f"[ERROR] Failed reading {pdb_file}: {e}")
        continue

    results.append((folder, len(chain_seqs), len(entities), chain2entity))


# 输出最终结果
print("\n===== SUMMARY =====\n")

for pdb_id, nchains, nentities, mapping in results:
    print(f"PDB {pdb_id}: {nchains} chains, {nentities} entities, mapping = {mapping}")

print("\n===== MULTIPLE ENTITY STRUCTURES =====\n")
for pdb_id, nchains, nentities, mapping in results:
    if nentities > 1:
        print(f"PDB {pdb_id}: {nentities} entities, chains → {mapping}")
