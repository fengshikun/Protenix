#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Q-BioLiP nonredund receptor/ligand mmCIF into Protenix-friendly *complex* mmCIFs.

Input (from Q-BioLiP "Non-redundant (nr-sequence)" mmCIF downloads):
  - nonredund_rec_cif/*.cif      (receptor/assembly structures, filename: {pdb}_{assembly}.cif)
  - nonredund_lig_cif/*.cif      (ligand-only structures, filename: {pdb}_{assembly}_{ligid}_{site}.cif)
  - ligand.json                 (optional; used for filtering by SMILES)

Output:
  <out_root>/
    complexes/xx/<sample_id>.cif     (merged complexes; sharded by first 2 chars)
    manifest.tsv                     (success + skipped-as-filtered are NOT included; only success)
    failed.tsv                       (errors)
    stats.txt

Notes / design choices:
  - We generate *one* complex CIF per ligand CIF (one ligand per sample).
  - We remove hydrogens + waters.
  - In the receptor, we keep polymers only (drop existing hetero residues) and normalize atom names
    for residues present in Protenix RES_ATOMS_DICT to avoid KeyError like 'CG', 'CD1', 'OG1'...
  - We also normalize atom names for ligand residues *if* their resname exists in RES_ATOMS_DICT,
    to avoid crashes for ligands whose CCD code overlaps with residue dictionaries.
  - Ligand heavy-atom filter is applied AFTER reading ligand CIF but BEFORE loading receptor CIF
    (to save IO when filtering out many ligands).

Typical usage (from your Q-BioLiP workdir that contains extracted folders):
  python3 clean_qbiolip_nonredund_mmcif.py \
    --rec_root nonredund_rec_cif \
    --lig_root nonredund_lig_cif \
    --ligand_json ligand.json \
    --out_root /path/to/out \
    --workers 32

Then run Protenix preprocessing:
  find /path/to/out/complexes -name "*.cif" | sort > /path/to/out/complexes_cif_paths.txt
  python3 $PROTENIX_ROOT/scripts/prepare_training_data.py -i /path/to/out/complexes_cif_paths.txt \
    -o /path/to/out/indices.csv -b /path/to/out/bioassembly -n 32 -d
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gemmi

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Try to use Protenix's residue-atom dictionary (best).
_HAS_PROTENIX_DICT = False
RES_ATOMS_DICT: Dict[str, Dict[str, int]] = {}
try:
    from protenix.data.constants import RES_ATOMS_DICT as _P_RES_ATOMS_DICT  # type: ignore

    RES_ATOMS_DICT = {k.upper(): {ak.strip().upper(): int(av) for ak, av in v.items()} for k, v in _P_RES_ATOMS_DICT.items()}
    _HAS_PROTENIX_DICT = True
except Exception:
    RES_ATOMS_DICT = {}
    _HAS_PROTENIX_DICT = False

# ligand filename: 4x9d_1_U5P_J
_LIG_RE = re.compile(r"^(?P<pdb>[0-9a-z]{4})_(?P<assembly>\d+)_(?P<ligid>[A-Za-z0-9]{1,3})_(?P<site>.+)$", re.IGNORECASE)

# Some very common non-druglike / ions / crystallization additives (tune as you like)
_DEFAULT_EXCLUDE_LIGIDS = {
    # waters / simple ions / buffers
    "HOH", "WAT", "H2O",
    "NA", "K", "LI", "MG", "MN", "CA", "ZN", "CU", "CO", "NI", "FE", "FE2", "FE3",
    "CL", "BR", "I", "CS", "SR", "BA",
    "SO4", "PO4", "NO3", "CO3", "SCN",
    # common cryo / solvents
    "GOL", "EDO", "PEG", "PG4", "PGO", "MPD", "DMS", "ACT", "FMT", "BME", "EOH", "IPA",
    # common small sugars / trivial (you may remove these exclusions if you want)
    "GLC", "MAN", "NAG", "BMA",
    # frequent non-organic / metal clusters
    "SF4", "FES", "F3S", "F4S", "FS4",
}

_ALLOWED_ELEMS_DEFAULT = {"C", "N", "O", "S", "P", "F", "CL", "BR", "I", "B", "SE", "SI"}  # reasonable organic set


@dataclass(frozen=True)
class LigTask:
    lig_path: str
    rec_key: str
    lig_id: str
    site_tag: str
    sample_id: str


# Globals for mp workers
_REC_MAP: Dict[str, str] = {}
_ARGS: dict = {}


def _is_hydrogen(atom: gemmi.Atom) -> bool:
    # gemmi element.name is uppercase string for known elements, '' for unknown.
    e = atom.element.name.upper() if atom.element is not None else ""
    if e in ("H", "D"):
        return True
    # fallback for messy CIFs
    nm = atom.name.strip().upper()
    return nm.startswith("H") or nm.startswith("D")


def _remove_waters_inplace(st: gemmi.Structure) -> None:
    for model in st:
        for chain in model:
            for ri in range(len(chain) - 1, -1, -1):
                if chain[ri].is_water():
                    del chain[ri]


def _keep_only_polymers_inplace(st: gemmi.Structure) -> None:
    """
    Drop all hetero residues in-place (keeps polymer residues only).
    gemmi uses res.het_flag:
      'A' => atom/standard residue in polymer
      'H' => hetero
      other values can appear; we conservatively keep only 'A'.
    """
    for model in st:
        for chain in model:
            for ri in range(len(chain) - 1, -1, -1):
                res = chain[ri]
                if getattr(res, "het_flag", "A") != "A":
                    del chain[ri]
        # drop empty chains
        for ci in range(len(model) - 1, -1, -1):
            if len(model[ci]) == 0:
                del model[ci]


def _normalize_atoms_for_residue_inplace(res: gemmi.Residue, valid_atoms: Set[str]) -> None:
    """
    For residues covered by RES_ATOMS_DICT, rename ambiguous atom names (CG->CG1, CD1->CD, OG->OG1, etc)
    and DROP atoms that are still unknown. This prevents Protenix KeyError.
    """
    used: Set[str] = set()

    def choose_name(raw: str) -> Optional[str]:
        nm = raw.strip().upper()
        if not nm:
            return None
        # exact
        if nm in valid_atoms and nm not in used:
            return nm

        # If has trailing digits, try base
        base = re.sub(r"\d+$", "", nm)
        if base != nm and base in valid_atoms and base not in used:
            return base

        # If missing digit, try append 1..4
        if base == nm:
            for d in ("1", "2", "3", "4"):
                cand = nm + d
                if cand in valid_atoms and cand not in used:
                    return cand

        # Try base + digit (covers cases like OG->OG1 when base != nm, and CG->CG1)
        for d in ("1", "2", "3", "4"):
            cand = base + d
            if cand in valid_atoms and cand not in used:
                return cand

        # Last resort: if nm is valid but already used, drop (avoid duplicates)
        return None

    for ai in range(len(res) - 1, -1, -1):
        atom = res[ai]
        # Drop unknown elements
        if atom.element is not None and atom.element.name.upper() == "X":
            del res[ai]
            continue
        # Drop hydrogens (normally already removed by st.remove_hydrogens())
        if _is_hydrogen(atom):
            del res[ai]
            continue

        new_name = choose_name(atom.name)
        if new_name is None:
            del res[ai]
            continue
        atom.name = new_name
        used.add(new_name)


def _normalize_atom_names_inplace(st: gemmi.Structure) -> None:
    """
    Apply residue atom-name normalization for ANY residue whose resname exists in RES_ATOMS_DICT,
    regardless of polymer/hetero status. This maximizes compatibility with Protenix parser.
    """
    if not RES_ATOMS_DICT:
        return
    for model in st:
        for chain in model:
            for res in chain:
                resname = res.name.strip().upper()
                if resname in RES_ATOMS_DICT:
                    valid = set(RES_ATOMS_DICT[resname].keys())
                    if valid:
                        _normalize_atoms_for_residue_inplace(res, valid)


def _keep_only_first_model(st: gemmi.Structure) -> None:
    # many CIFs contain only one model, but keep safe
    while len(st) > 1:
        del st[1]


def _clean_structure_inplace(
    st: gemmi.Structure,
    *,
    keep_only_polymers: bool,
    normalize_atoms: bool,
) -> None:
    _keep_only_first_model(st)
    try:
        st.remove_alternative_conformations()
    except Exception:
        # older gemmi builds might not have it
        pass
    try:
        st.remove_hydrogens()
    except Exception:
        pass

    _remove_waters_inplace(st)

    if keep_only_polymers:
        _keep_only_polymers_inplace(st)

    if normalize_atoms:
        _normalize_atom_names_inplace(st)

    # Drop empty chains after all ops
    for model in st:
        for ci in range(len(model) - 1, -1, -1):
            if len(model[ci]) == 0:
                del model[ci]


def _pick_free_chain_id(existing: Set[str]) -> str:
    # Prefer single-char chain IDs, last letters first to avoid clashing with common A/B/C...
    for ch in "ZYXWVUTSRQPONMLKJIHGFEDCBA":
        if ch not in existing:
            return ch
    # fallback to multi-char (mmCIF allows it)
    i = 1
    while True:
        cand = f"Z{i}"
        if cand not in existing:
            return cand
        i += 1


def _extract_best_ligand_atoms(st_lig: gemmi.Structure, lig_id: str) -> List[gemmi.Atom]:
    """
    Choose a "best" ligand residue (prefer resname == lig_id, else max heavy atom count),
    return its atoms (heavy only) as clones.
    """
    lig_id_u = lig_id.strip().upper()
    best: Optional[Tuple[int, int, gemmi.Residue]] = None  # (match, heavy_count, residue)
    if len(st_lig) == 0:
        return []
    m = st_lig[0]
    for chain in m:
        for res in chain:
            if res.is_water():
                continue
            resname_u = res.name.strip().upper()
            match = 1 if (resname_u == lig_id_u) else 0
            heavy = 0
            for atom in res:
                if _is_hydrogen(atom):
                    continue
                if atom.element is not None and atom.element.name.upper() == "X":
                    continue
                heavy += 1
            if heavy <= 0:
                continue
            key = (match, heavy, res)
            if best is None or (match, heavy) > (best[0], best[1]):
                best = key
    if best is None:
        return []
    res = best[2]
    atoms: List[gemmi.Atom] = []
    for atom in res:
        if _is_hydrogen(atom):
            continue
        if atom.element is not None and atom.element.name.upper() == "X":
            continue
        atoms.append(atom.clone())
    return atoms


def _write_complex_cif(
    rec_path: Path,
    lig_path: Path,
    out_cif: Path,
    *,
    sample_id: str,
    lig_id: str,
    min_lig_heavy_atoms: int,
    max_lig_heavy_atoms: int,
    require_allowed_elements: bool,
    allowed_elements: Set[str],
    normalize_atoms: bool,
) -> Tuple[str, str]:
    """
    Returns (status, message)
      status: "OK" | "SKIP" | "FAIL"
    """
    try:
        # load + clean ligand first (so we can filter before reading receptor)
        lig = gemmi.read_structure(str(lig_path))
        _clean_structure_inplace(lig, keep_only_polymers=False, normalize_atoms=normalize_atoms)

        lig_atoms = _extract_best_ligand_atoms(lig, lig_id)
        n_heavy = len(lig_atoms)
        if min_lig_heavy_atoms > 0 and n_heavy < min_lig_heavy_atoms:
            return "SKIP", f"lig_heavy_atoms<{min_lig_heavy_atoms} ({n_heavy})"
        if max_lig_heavy_atoms > 0 and n_heavy > max_lig_heavy_atoms:
            return "SKIP", f"lig_heavy_atoms>{max_lig_heavy_atoms} ({n_heavy})"

        if require_allowed_elements:
            elems = {a.element.name.upper() for a in lig_atoms if a.element is not None and a.element.name}
            # If ligand has no element info, skip this filter
            if elems and any(e not in allowed_elements for e in elems):
                bad = sorted([e for e in elems if e not in allowed_elements])[:10]
                return "SKIP", f"lig_has_disallowed_elements:{','.join(bad)}"

        # resume check (after filter, before heavy IO)
        if out_cif.exists() and out_cif.stat().st_size > 0:
            return "OK", "exists"

        # load + clean receptor
        rec = gemmi.read_structure(str(rec_path))
        _clean_structure_inplace(rec, keep_only_polymers=True, normalize_atoms=normalize_atoms)

        if len(rec) == 0 or len(rec[0]) == 0:
            return "FAIL", "empty_receptor_after_clean"

        # Add ligand as new chain
        existing_chain_ids = {ch.name for ch in rec[0]}
        lig_chain_id = _pick_free_chain_id(existing_chain_ids)

        chain = gemmi.Chain(lig_chain_id)
        res = gemmi.Residue()
        res.name = lig_id.strip().upper()
        res.het_flag = "H"
        res.seqid = gemmi.SeqId(1, " ")
        for a in lig_atoms:
            chain_res_atom_name = a.name.strip().upper()
            if chain_res_atom_name:
                a.name = chain_res_atom_name
            chain_res_elem = a.element.name.upper() if a.element is not None else ""
            if chain_res_elem == "X":
                continue
            res.add_atom(a)
        if len(res) == 0:
            return "FAIL", "ligand_has_no_atoms_after_clean"

        chain.add_residue(res)
        rec[0].add_chain(chain)

        # Set entry id for Protenix
        rec.name = sample_id

        out_cif.parent.mkdir(parents=True, exist_ok=True)
        doc = rec.make_mmcif_document()
        doc.write_file(str(out_cif))
        return "OK", ""
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {e}"


def _load_ligand_json(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, dict] = {}
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict) and "ligid" in obj:
                out[str(obj["ligid"]).strip().upper()] = obj
    elif isinstance(data, dict):
        # some versions could be dict[ligid]=...
        for k, v in data.items():
            out[str(k).strip().upper()] = v if isinstance(v, dict) else {"value": v}
    return out


def _ligid_passes_filters(
    lig_id: str,
    *,
    keep_only_small_molecule: bool,
    require_smiles_in_ligand_json: bool,
    ligand_json_map: Dict[str, dict],
    exclude_ligids: Set[str],
) -> bool:
    lig_u = lig_id.strip().upper()
    if lig_u in exclude_ligids:
        return False
    if not keep_only_small_molecule:
        return True

    # crude: exclude single/di-atom ions etc by ID list; rest keep
    if require_smiles_in_ligand_json:
        meta = ligand_json_map.get(lig_u)
        if meta is None:
            return False
        smiles = str(meta.get("smiles", "")).strip()
        if not smiles:
            return False
    return True


def _iter_lig_tasks(
    lig_root: Path,
    rec_map: Dict[str, str],
    ligand_json_map: Dict[str, dict],
    *,
    keep_only_small_molecule: bool,
    require_smiles_in_ligand_json: bool,
    exclude_ligids: Set[str],
    limit: int,
) -> List[LigTask]:
    tasks: List[LigTask] = []
    n = 0
    for p in lig_root.glob("*.cif"):
        if not p.is_file():
            continue
        m = _LIG_RE.match(p.stem)
        if m is None:
            continue
        pdb = m.group("pdb").lower()
        assembly = m.group("assembly")
        lig_id = m.group("ligid").upper()
        site = m.group("site")
        rec_key = f"{pdb}_{assembly}"
        if rec_key not in rec_map:
            continue
        if not _ligid_passes_filters(
            lig_id,
            keep_only_small_molecule=keep_only_small_molecule,
            require_smiles_in_ligand_json=require_smiles_in_ligand_json,
            ligand_json_map=ligand_json_map,
            exclude_ligids=exclude_ligids,
        ):
            continue
        tasks.append(
            LigTask(
                lig_path=str(p),
                rec_key=rec_key,
                lig_id=lig_id,
                site_tag=site,
                sample_id=p.stem,
            )
        )
        n += 1
        if limit > 0 and n >= limit:
            break
    return tasks


def _init_worker(rec_map: Dict[str, str], args_dict: dict) -> None:
    global _REC_MAP, _ARGS
    _REC_MAP = rec_map
    _ARGS = args_dict


def _worker(task: LigTask) -> Tuple[str, str, str, str, str]:
    """
    Returns (status, sample_id, rec_path, lig_path, message)
      status: OK / SKIP / FAIL
    """
    rec_path = _REC_MAP.get(task.rec_key, "")
    if not rec_path:
        return "FAIL", task.sample_id, "", task.lig_path, "receptor_not_found"

    out_root = Path(_ARGS["out_root"])
    # shard by first 2 chars (reduces dir fanout for 100k+ files)
    shard = (task.sample_id[:2] or "xx").lower()
    out_cif = out_root / "complexes" / shard / f"{task.sample_id}.cif"

    status, msg = _write_complex_cif(
        rec_path=Path(rec_path),
        lig_path=Path(task.lig_path),
        out_cif=out_cif,
        sample_id=task.sample_id,
        lig_id=task.lig_id,
        min_lig_heavy_atoms=int(_ARGS["min_lig_heavy_atoms"]),
        max_lig_heavy_atoms=int(_ARGS["max_lig_heavy_atoms"]),
        require_allowed_elements=bool(_ARGS["require_allowed_elements"]),
        allowed_elements=set(_ARGS["allowed_elements"]),
        normalize_atoms=bool(_ARGS["normalize_atoms"]),
    )
    return status, task.sample_id, rec_path, task.lig_path, msg


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec_root", type=str, required=True, help="Directory containing receptor CIFs: nonredund_rec_cif/*.cif")
    ap.add_argument("--lig_root", type=str, required=True, help="Directory containing ligand CIFs: nonredund_lig_cif/*.cif")
    ap.add_argument("--out_root", type=str, required=True, help="Output root directory")

    ap.add_argument("--ligand_json", type=str, default="", help="Optional ligand.json (used for filtering)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunksize", type=int, default=200)

    ap.add_argument("--limit", type=int, default=0, help="Debug: only process first N ligand CIFs (0 = all)")

    # filtering knobs
    ap.add_argument("--keep_only_small_molecule", action="store_true", help="Use ligand-id based filters (default on)")
    ap.set_defaults(keep_only_small_molecule=True)

    ap.add_argument("--require_smiles_in_ligand_json", action="store_true", help="Keep only ligids that exist in ligand.json with non-empty SMILES")
    ap.set_defaults(require_smiles_in_ligand_json=False)

    ap.add_argument("--exclude_ligids", type=str, default="", help="Comma-separated extra ligids to exclude (e.g., 'ATP,ADP,GTP')")
    ap.add_argument("--min_lig_heavy_atoms", type=int, default=5)
    ap.add_argument("--max_lig_heavy_atoms", type=int, default=120)

    ap.add_argument("--require_allowed_elements", action="store_true", help="Drop ligands containing elements outside --allowed_elements")
    ap.set_defaults(require_allowed_elements=False)
    ap.add_argument("--allowed_elements", type=str, default=",".join(sorted(_ALLOWED_ELEMS_DEFAULT)))

    # cleaning / compatibility
    ap.add_argument("--normalize_atoms", action="store_true", help="Normalize residue atom names using Protenix RES_ATOMS_DICT (default on)")
    ap.set_defaults(normalize_atoms=True)

    args = ap.parse_args(argv)

    rec_root = Path(args.rec_root).expanduser().resolve()
    lig_root = Path(args.lig_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not rec_root.exists():
        raise SystemExit(f"[FATAL] rec_root not found: {rec_root}")
    if not lig_root.exists():
        raise SystemExit(f"[FATAL] lig_root not found: {lig_root}")

    ligand_json_map = _load_ligand_json(Path(args.ligand_json).expanduser().resolve() if args.ligand_json else None)

    # receptor map
    rec_map: Dict[str, str] = {}
    for p in rec_root.glob("*.cif"):
        if p.is_file():
            rec_map[p.stem] = str(p)
    if not rec_map:
        raise SystemExit(f"[FATAL] no receptor CIFs found under: {rec_root}")

    exclude = set(_DEFAULT_EXCLUDE_LIGIDS)
    if args.exclude_ligids.strip():
        exclude |= {x.strip().upper() for x in args.exclude_ligids.split(",") if x.strip()}

    # build tasks WITHOUT reading ligand CIF contents (fast)
    tasks = _iter_lig_tasks(
        lig_root=lig_root,
        rec_map=rec_map,
        ligand_json_map=ligand_json_map,
        keep_only_small_molecule=bool(args.keep_only_small_molecule),
        require_smiles_in_ligand_json=bool(args.require_smiles_in_ligand_json),
        exclude_ligids=exclude,
        limit=int(args.limit),
    )
    if not tasks:
        raise SystemExit(
            "[FATAL] No ligand CIF matched expected filename pattern or passed filters.\n"
            f"  lig_root={lig_root}\n"
            "  expected example: 4x9d_1_U5P_J.cif\n"
        )

    # mp ctx
    try:
        ctx = mp.get_context("fork")
    except Exception:
        ctx = mp

    args_dict = {
        "out_root": str(out_root),
        "min_lig_heavy_atoms": int(args.min_lig_heavy_atoms),
        "max_lig_heavy_atoms": int(args.max_lig_heavy_atoms),
        "require_allowed_elements": bool(args.require_allowed_elements),
        "allowed_elements": [x.strip().upper() for x in args.allowed_elements.split(",") if x.strip()],
        "normalize_atoms": bool(args.normalize_atoms),
    }

    manifest_path = out_root / "manifest.tsv"
    failed_path = out_root / "failed.tsv"
    stats_path = out_root / "stats.txt"

    ok = 0
    skip = 0
    fail = 0

    with open(manifest_path, "w", encoding="utf-8", newline="") as f_ok, open(failed_path, "w", encoding="utf-8", newline="") as f_fail:
        ok_w = csv.writer(f_ok, delimiter="\t")
        ff_w = csv.writer(f_fail, delimiter="\t")
        ok_w.writerow(["sample_id", "rec_key", "lig_id", "site_tag", "receptor_path", "ligand_path", "complex_cif"])
        ff_w.writerow(["sample_id", "rec_key", "lig_id", "site_tag", "receptor_path", "ligand_path", "reason"])

        with ctx.Pool(processes=int(args.workers), initializer=_init_worker, initargs=(rec_map, args_dict)) as pool:
            it = pool.imap_unordered(_worker, tasks, chunksize=max(1, int(args.chunksize)))
            if tqdm is not None:
                it = tqdm(it, total=len(tasks), desc="build_complexes", unit="lig")

            for status, sample_id, rec_path, lig_path, msg in it:
                # derive rec_key + lig_id/site from sample_id quickly
                m = _LIG_RE.match(Path(lig_path).stem)
                if m is None:
                    rec_key = ""
                    lig_id = ""
                    site = ""
                else:
                    rec_key = f"{m.group('pdb').lower()}_{m.group('assembly')}"
                    lig_id = m.group("ligid").upper()
                    site = m.group("site")

                shard = (sample_id[:2] or "xx").lower()
                out_cif = out_root / "complexes" / shard / f"{sample_id}.cif"

                if status == "OK":
                    ok += 1
                    ok_w.writerow([sample_id, rec_key, lig_id, site, rec_path, lig_path, str(out_cif)])
                elif status == "SKIP":
                    skip += 1
                    # We do NOT write SKIP into failed.tsv (keeps it for true errors only)
                else:
                    fail += 1
                    ff_w.writerow([sample_id, rec_key, lig_id, site, rec_path, lig_path, msg])

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"rec_root\t{rec_root}\n")
        f.write(f"lig_root\t{lig_root}\n")
        f.write(f"out_root\t{out_root}\n")
        f.write(f"receptor_cif_count\t{len(rec_map)}\n")
        f.write(f"ligand_tasks\t{len(tasks)}\n")
        f.write(f"success\t{ok}\n")
        f.write(f"skipped_by_filters\t{skip}\n")
        f.write(f"failed\t{fail}\n")
        f.write(f"normalize_atoms\t{bool(args.normalize_atoms)}\n")
        f.write(f"has_protenix_RES_ATOMS_DICT\t{_HAS_PROTENIX_DICT}\n")
        f.write(f"min_lig_heavy_atoms\t{int(args.min_lig_heavy_atoms)}\n")
        f.write(f"max_lig_heavy_atoms\t{int(args.max_lig_heavy_atoms)}\n")
        f.write(f"keep_only_small_molecule\t{bool(args.keep_only_small_molecule)}\n")
        f.write(f"require_smiles_in_ligand_json\t{bool(args.require_smiles_in_ligand_json)}\n")
        f.write(f"exclude_ligids_count\t{len(exclude)}\n")

    print("[DONE]")
    print("  complexes_dir:", str(out_root / "complexes"))
    print("  manifest:", str(manifest_path))
    print("  failed:", str(failed_path))
    print("  stats:", str(stats_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
