"""
BiHC (Bidirectional Hierarchical Clustering) - protein domain segmentation.

Structure reading, the STRIDE wrapper, the per-chain driver around bihc_core / bihc_hm, output helpers (ranges, PyMOL,
structure files) and the partition-comparison metrics.
"""
import os
import sys
import json
import subprocess
import tempfile
import itertools
from io import StringIO
from typing import List, Tuple, Sequence, Union, Optional, Iterable

import numpy as np
from scipy.spatial import distance_matrix

from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.PDBIO import Select
from Bio.PDB.Polypeptide import is_aa

SS_BY_CHAIN = None
RES_LABELS_BY_CHAIN = None


def _is_polymer_residue(residues, i):
    """bihc_core.is_polymer_residue (loaded lazily, as everywhere in this module)"""
    return _load_core().is_polymer_residue(residues, i)


def _load_core():
    """Import the pipeline (bihc_core) however this file was loaded (package, script, or importlib)."""
    try:
        from src import bihc_core as _m
        return _m
    except Exception:
        pass
    try:
        import bihc_core as _m
        return _m
    except Exception:
        pass
    import importlib.util as _ilu
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bihc_core.py')
    _spec = _ilu.spec_from_file_location('bihc_core', _p)
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    return _m


bihc_core = _load_core()

OPT_T = None          # None -> predict() uses t = bihc_hm.T_HM (fallback graphs: their own F)
OPT_POST = True       # post-processing (MINSEGMENT -> min2ss), on by default


def set_options(t=None, post=True):
    """threshold t of the partition selection (None -> bihc_hm.T_HM) and the post-processing switch."""
    global OPT_T, OPT_POST
    OPT_T = t
    OPT_POST = bool(post)


def set_structure_path(path):
    """Give the pipeline (bihc_core) the input file, which it needs for the all-atom tessellation."""
    bihc_core.set_structure_path(path)


def set_ss_by_chain(ss_by_chain: dict):
    """
    Register the STRIDE secondary structure of each chain (used to build the SSE units).
    ss_by_chain format:
      { 'A': ss_info, 'B': ss_info, ... }

    Where ss_info can be:
      - ['H','E','C', ...] length N
      - [(res_label,'H'), (res_label,'C'), ...] length N
    """
    global SS_BY_CHAIN
    SS_BY_CHAIN = ss_by_chain


def set_res_labels_by_chain(res_labels_by_chain: dict):
    """Original residue labels per chain, e.g. {'A_': ['1','2','11A', ...]}.

    The pipeline (bihc_core) needs them to align the all-atom tessellation to the CA coordinates.
    """
    global RES_LABELS_BY_CHAIN
    RES_LABELS_BY_CHAIN = res_labels_by_chain


class CustomNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check for NumPy data types
        if isinstance(obj, (np.ndarray, np.generic)):  
            return obj.tolist()  # Convert NumPy arrays and scalars to lists
            
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)  # Convert NumPy floats to Python floats
            
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)  # Convert NumPy ints to Python ints
            
        # Add any other NumPy types you want to handle here
        return super(CustomNumpyEncoder, self).default(obj)


def decorate_message(mess, cover_by = '='):
    print(cover_by*len(mess))
    print(mess)
    print(cover_by*len(mess))


def process_structure(input_data, atom_type="CA", models=True, get_res=False, filename=None, chain_filter=None):
    """
    Parser for PDB and mmCIF protein chains (nucleic-acid chains are skipped: BiHC is a protein method).

    Returns:
      - result: list of (xs, ys, zs) tuples per chain
      - chains: list of chain IDs
      - res_num: list of CONTIGUOUS integers [1,2,3,...] per chain (for clustering)
      - res_labels: list of ORIGINAL residue labels ["1","2","11A","11B",...] per chain (for output)
    """

    if filename is None:
        raise ValueError("filename must be provided to detect .pdb vs .cif")

    is_list = isinstance(input_data, list)
    is_str = isinstance(input_data, str)
    if not (is_list or is_str):
        raise TypeError("input_data must be a list of lines or a file path string")

    fname = filename.lower()

    # Parse structure
    if fname.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
        if is_list:
            structure = parser.get_structure("model", StringIO("\n".join(input_data)))
        else:
            structure = parser.get_structure("model", input_data)
    elif fname.endswith((".cif", ".mmcif")):
        parser = MMCIFParser(QUIET=True)
        if is_list:
            structure = parser.get_structure("model", StringIO("\n".join(input_data)))
        else:
            structure = parser.get_structure("model", input_data)
    else:
        raise ValueError("Unknown file type: expected .pdb or .cif")

    # Helpers
    def _is_nucleotide_resname(resname: str) -> bool:
        r = (resname or "").strip().upper()
        return r in {"A", "C", "G", "U", "I", "T"} or r in {"DA", "DC", "DG", "DT", "DU"} or r.startswith("D")

    def _detect_chain_type(residues) -> str:
        aa_count = 0
        nt_count = 0
        for res in residues:
            hetflag, _, _ = res.id
            if hetflag == "W":
                continue
            if is_aa(res, standard=True):
                aa_count += 1
                continue
            try:
                if _is_nucleotide_resname(res.get_resname()):
                    nt_count += 1
            except Exception:
                pass
        return "rna" if nt_count > aa_count else "protein"

    # Extract coordinates
    result = []
    chains = []
    res_num = []
    res_labels = []

    # Handle models
    model_iter = structure
    if models is False:
        model_iter = []
        for m in structure:
            model_iter = [m]
            break

    for model in model_iter:
        for chain in model:
            if chain_filter is not None and chain.id not in set(chain_filter):
                continue

            residues = list(chain.get_residues())
            if _detect_chain_type(residues) != "protein":
                continue                       # nucleic-acid chain: not handled by BiHC
            atoms_to_try = [atom_type] if atom_type == "CA" else [atom_type, "CA"]

            xs = ys = zs = None
            labels = None
            chosen = None

            for at in atoms_to_try:
                _xs, _ys, _zs = [], [], []
                _labels = []

                for ri, residue in enumerate(residues):
                    hetflag, seq_id, icode = residue.id
                    if hetflag == "W":
                        continue
                    # HETATM residues (modified residues such as MSE written as HETATM) stay out of the graph, as
                    # they stay out of the Voronota tessellation and of STRIDE, which read ATOM records only; they
                    # are put back into the domains at output (hetero_polymer_order / fill_hetero). Kept in the
                    # graph they become zero-area coil units that cut their helix or strand in two.
                    if hetflag.strip():
                        continue

                    # Polymer amino-acid residues: standard ones, plus any other residue peptide-bonded into the
                    # chain (UNK and modified residues written as ATOM), but not ligands carrying an amino-acid
                    # moiety (SAM / SAH), which are not bonded into the chain.
                    if not (is_aa(residue, standard=True) or _is_polymer_residue(residues, ri)):
                        continue

                    if at not in residue:
                        continue

                    atom = residue[at]
                    x, y, z = atom.get_coord()
                    _xs.append(float(x))
                    _ys.append(float(y))
                    _zs.append(float(z))

                    # Store ORIGINAL label with insertion code
                    ic = (icode.strip() if isinstance(icode, str) else "")
                    _labels.append(f"{int(seq_id)}{ic}")

                if len(_xs) > 0:
                    xs, ys, zs = _xs, _ys, _zs
                    labels = _labels
                    chosen = at
                    break

            if chosen is None:
                continue

            chains.append(chain.id)
            result.append((xs, ys, zs))
            
            # CRITICAL: res_num = contiguous 1..N for clustering
            res_num.append(list(range(1, len(xs) + 1)))
            
            # CRITICAL: res_labels = original labels for output
            res_labels.append(labels)

    if not chains:
        return False

    if get_res:
        return result, chains, res_num, res_labels
    else:
        return result, chains, res_num


def hetero_polymer_order(path, chain, graph_labels, atom_type="CA"):
    """Every polymer residue of one chain in file order, HETATM ones included: [(label, is_hetero, (x, y, z))].

    The graph holds the ATOM residues only (process_structure); this is the same residue rule without that restriction,
    taken from the model whose ATOM residues are exactly graph_labels. None if no model matches or the chain has no
    HETATM polymer residue (nothing to put back)."""
    fname = path.lower()
    if fname.endswith(".pdb") or fname.endswith(".ent"):
        structure = PDBParser(QUIET=True).get_structure("model", path)
    elif fname.endswith((".cif", ".mmcif")):
        structure = MMCIFParser(QUIET=True).get_structure("model", path)
    else:
        return None
    want = [str(x) for x in graph_labels]
    for model in structure:
        for ch in model:
            if ch.id != chain:
                continue
            residues = list(ch.get_residues())
            order = []
            for ri, residue in enumerate(residues):
                hetflag, seq_id, icode = residue.id
                if hetflag == "W" or not (is_aa(residue, standard=True) or _is_polymer_residue(residues, ri)):
                    continue
                at = atom_type if atom_type in residue else "CA"
                if at not in residue:
                    continue
                ic = (icode.strip() if isinstance(icode, str) else "")
                order.append((f"{int(seq_id)}{ic}", bool(hetflag.strip()), tuple(float(v) for v in residue[at].get_coord())))
            if [l for l, h, _ in order if not h] == want:
                return order if any(h for _, h, _ in order) else None
    return None


def fill_hetero(pred, graph_labels, order):
    """Labels of every residue of `order` (hetero_polymer_order): graph residues keep their label; each run of HETATM
    residues takes the domain of its neighbours when both sides agree or only one side exists; between two domains a
    single residue goes to the larger domain and a longer run is split half / half. So a domain is not cut into pieces
    by the one- or two-residue gaps the HETATM residues leave in the graph."""
    lab_of = {str(l): int(p) for l, p in zip(graph_labels, pred)}
    labels = [l for l, _, _ in order]
    lab = [lab_of.get(l) if not h else None for l, h, _ in order]
    size = {}
    for v in lab_of.values():
        size[v] = size.get(v, 0) + 1
    n, i = len(lab), 0
    while i < n:
        if lab[i] is not None:
            i += 1; continue
        j = i
        while j < n and lab[j] is None:
            j += 1
        L = lab[i - 1] if i > 0 else None
        R = lab[j] if j < n else None
        if L is None or R is None or L == R:
            fill = [L if L is not None else R] * (j - i)
        elif j - i == 1:
            fill = [L if size.get(L, 0) >= size.get(R, 0) else R]
        else:
            h = (j - i) // 2
            fill = [L] * h + [R] * (j - i - h)
        lab[i:j] = fill
        i = j
    return lab, labels, [xyz for _, _, xyz in order]


def list_to_range(l):
    l = sorted(set(l))  # Sort and remove duplicates, but keep order
    l2 = []
    s = l[0]  # Start of the first range

    for p, v in enumerate(l):
        if p >= 1:
            # If current number is consecutive with the previous one
            if v == l[p-1] + 1:
                # If it's the last element, append the final range
                if p == len(l) - 1:
                    l2.append(range(s, v+1))
                continue
            
            # If the sequence breaks, append the current range
            e = l[p-1] + 1
            l2.append(range(s, e))
            s = v  # Start a new range with the current element

        # If it's the last element and not part of a consecutive sequence
        if p == len(l) - 1:
            l2.append(range(s, v+1))
    
    return l2


def pymol_process(pred, res_num, name=None, color=None, verbose=False, res_labels=None, chain=None, obj=None):
    """
    Generate PyMOL commands for coloring clusters.
    
    Args:
        pred: cluster labels
        res_num: contiguous residue numbers [1,2,3,...]
        name: prefix for selection names
        color: list of colors
        verbose: print commands
        res_labels: original PDB residue labels (optional)
        chain: chain id; the selections are restricted to it (residue numbers repeat across chains)
        obj: PyMOL object name; the selections are restricted to it as well
    """
    if color is None:
        color = ['blue', 'yellow', 'magenta', 'orange', 'green', 'pink', 'cyan', 
                 'purple', 'red', 'white', 'brown', 'lightblue', 'lightorange', 
                 'lightpink', 'gold']

    label_set = list(set(pred))

    if len(label_set) > len(color):
        additional_colors_names = ['{:02d}'.format(i) for i in range(len(color), len(label_set))]
        color.extend(additional_colors_names)

    cmd = []
    if verbose:
        msg = 'Command for PyMOL:'
        decorate_message(msg)
    
    for num, label in enumerate(label_set):
        # Get positions for this label
        positions = [p for p, v in enumerate(pred) if v == label]
        
        # Use original labels if available, otherwise use res_num
        if res_labels is not None:
            label_list = [res_labels[p] for p in positions]
        else:
            label_list = [res_num[p] for p in positions]
        
        if label == -1:
            clust_name = name + f'_outlier' if name is not None else f'outlier'
            cmd.append(command_pymol_with_labels(label_list, clust_name, 'grey', verbose, res_labels, chain, obj))
        else:
            clust_name = name + f'_cluster_{num+1}' if name is not None else f'cluster_{num+1}'
            cmd.append(command_pymol_with_labels(label_list, clust_name, color[num], verbose, res_labels, chain, obj))

    return cmd


def command_pymol_with_labels(label_list, name, color, verbose=False, chain_labels=None, chain=None, obj=None):
    """
    Generate PyMOL command using original residue labels (handles insertions).
    
    Args:
        label_list: list of residue labels like ["1", "2", "11A", "11B", "12"]
        name: selection name
        color: color name
        verbose: print command
    """
    # Group consecutive ranges (over the chain's own residues when they are given: domain_ranges)
    ranges = domain_ranges(label_list, chain_labels) if chain_labels is not None else list_to_range_with_insertions(label_list)
    esc = lambda x: f'\\{x}' if str(x).startswith('-') else f'{x}'      # PyMOL reads '-5' as a range operator

    scope = ' and '.join(([obj] if obj else []) + ([f'chain {chain}'] if chain else []))
    mess = f'select {name}, {scope} and resi ' if scope else f'select {name}, res '
    for p, (start, end) in enumerate(ranges):
        if start == end:
            mess += esc(start)
        else:
            mess += f'{esc(start)}-{esc(end)}'
        
        if p != len(ranges) - 1:
            mess += '+'
    
    mess += f'; color {color}, {name}'
    
    if verbose:
        print(mess)
    
    return mess


def cluster_algo(data, algo, min_length, chain_id, chain_type):
    """BiHC on one protein chain: SSE super-nodes + Voronoi contact areas + self-containment ratio (bihc_core / bihc_hm).
    Returns one domain label per residue of `data`."""
    print('-' * 40)
    bihc_core.LAST_GRAPH = None                   # never let a skipped chain inherit the previous chain's graph
    if algo != 'B':
        sys.exit(f"Unrecognized algorithm: {algo}")
    if chain_type != 'protein':                 # BiHC is a protein method; process_structure already skips the rest
        print(f"  chain {chain_id} is not a protein chain; skipped")
        return np.zeros(len(data), dtype=int)
    print(f"Executing BiHC on chain {chain_id}...")
    ss_info = SS_BY_CHAIN.get(chain_id) if isinstance(SS_BY_CHAIN, dict) else None
    res_labels = RES_LABELS_BY_CHAIN.get(chain_id) if isinstance(RES_LABELS_BY_CHAIN, dict) else None
    if res_labels is None:
        print('  WARNING: residue labels were not registered for this chain; assuming 1..N, '
              'which will not match the structure if it does not start at residue 1')
        res_labels = list(range(1, len(data) + 1))
    t = OPT_T                          # None -> t = bihc_hm.T_HM (fallback graphs: their own F)
    print(f"  BiHC ({bihc_core.backend()} contact areas): harmonic-mean partition selection over bottom-up / "
          f"top-down partitions, t={t if t is not None else 0.365}, beta={bihc_core.BETA}, post-processing: "
          + ("MINSEGMENT 40 -> min2ss, adjacent highest-F merges" if OPT_POST else "OFF"))
    labels = bihc_core.predict(data, ss_info, res_labels, chain_id, F=t, post=OPT_POST, verbose=True)
    return np.asarray(labels, dtype=int)


def last_graph():
    """the unit graph of the chain cluster_algo() just processed (None if it was not built)"""
    return bihc_core.LAST_GRAPH


def cluster_from_graph(graph, chain_id):
    """BiHC on one chain from its saved unit graph (BiHC.py -g): only the partition selection and clean-up are run."""
    print('-' * 40)
    print(f"Executing BiHC on chain {chain_id} from its saved graph...")
    t = OPT_T
    print(f"  BiHC (saved {graph.get('graph_kind', 'trivial')} graph): harmonic-mean partition selection over bottom-up / "
          f"top-down partitions, t={t if t is not None else 0.365}, beta={bihc_core.BETA}, post-processing: "
          + ("MINSEGMENT 40 -> min2ss, adjacent highest-F merges" if OPT_POST else "OFF"))
    labels = bihc_core.predict_from_graph(graph, t=t, post=OPT_POST, verbose=True)
    return np.asarray(labels, dtype=int)


def check_C(result, threshold):
    """
    Check and filter chains based on threshold.
    
    Returns:
      - data: list of coordinate arrays
      - res_num_array: list of residue number arrays (contiguous integers)
      - removed_chain_index: list of removed chain indices
      - res_labels_array: list of original residue labels
    """
    data = []   
    removed_chain_index = []
    res_labels_array = []
    
    if result == False or len(result) == 0:
        return False
    
    # Unpack result - check if it has res_labels (4 elements) or not (3 elements)
    if len(result) == 4:  # with res_labels
        coords_list, chain_ids, res_num_list, res_labels_list = result
    else:  # without res_labels
        coords_list, chain_ids, res_num_list = result
        # Create fallback res_labels as strings of res_num
        res_labels_list = [[str(x) for x in rn] for rn in res_num_list]
    
    # Filter chains by threshold
    for t in range(len(coords_list)):
        if len(coords_list[t][0]) < threshold:
            removed_chain_index += [t]
            continue
        
        # Build coordinate array
        l = [[coords_list[t][0][i], coords_list[t][1][i], coords_list[t][2][i]] 
             for i in range(len(coords_list[t][0]))]
        data += [np.array(l)]
        
        # Keep corresponding res_labels
        res_labels_array += [res_labels_list[t]]
    
    # Get valid res_num (not removed)
    valid_res_num = [res_num_list[i] for i in range(len(res_num_list)) 
                     if i not in removed_chain_index]
    
    return data, valid_res_num, removed_chain_index, res_labels_array


def find_missing_res(file):
    """
    Missing residues of the protein chains, for PDB and mmCIF.
    
    Returns: dict {chain_id: list of missing residue labels with insertions}
             Example: {'A': ['10', '11', '11A'], 'B': ['50']}
    """
    from collections import defaultdict
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa

    missing = defaultdict(list)  # lists, not sets: residue order is kept
    fname = file.lower()

    def _is_nt(resname: str) -> bool:
        r = (resname or "").strip().upper()
        return r in {"A", "C", "G", "U", "I", "T"} or r in {"DA", "DC", "DG", "DT", "DU"} or r.startswith("D")

    def _looks_like_protein_resname(resname: str) -> bool:
        r = (resname or "").strip().upper()
        return len(r) == 3 or r == "MSE"

    # Step 1: Explicit missing annotations
    if fname.endswith(".pdb"):
        is_remark_465 = False
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("REMARK 465   M RES C SSSEQI"):
                    is_remark_465 = True
                    continue
                if is_remark_465 and line.startswith("REMARK 465"):
                    if len(line.strip()) < 20:
                        continue
                    try:
                        res_name = line[15:18].strip()
                        chain_id = line[19:20].strip()
                        res_num = line[22:27].strip()
                        if chain_id == "":
                            continue
                        if _is_nt(res_name) or _looks_like_protein_resname(res_name):
                            missing[chain_id].append(res_num)
                    except Exception:
                        continue
                elif is_remark_465 and not line.startswith("REMARK"):
                    break

    elif fname.endswith((".cif", ".mmcif")):
        in_block = False
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "_pdbx_unobs_or_zero_occ_residues.polymer_flag" in line:
                    in_block = True
                    continue
                if in_block:
                    if line.startswith("#") or line.strip() == "" or line.startswith("_"):
                        break
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        chain_id = parts[1]
                        res_num = parts[2]
                        comp_id = parts[3]
                        if chain_id:
                            if _is_nt(comp_id) or _looks_like_protein_resname(comp_id):
                                try:
                                    missing[chain_id].append(res_num)
                                except Exception:
                                    pass
    else:
        return {}

    # Step 2: Infer gaps from observed residues
    try:
        if fname.endswith(".pdb"):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("model", file)
        else:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("model", file)

        for model in structure:
            for chain in model:
                chain_id = chain.id
                observed_labels = []
                
                for res in chain.get_residues():
                    hetflag, resseq, icode = res.id
                    if hetflag == "W":
                        continue
                    if is_aa(res, standard=True) or _is_nt(res.get_resname()):
                        ic = (icode.strip() if isinstance(icode, str) else "")
                        observed_labels.append((int(resseq), f"{int(resseq)}{ic}"))

                if not observed_labels:
                    continue

                # Sort by integer part
                observed_labels.sort(key=lambda x: x[0])
                
                # Find gaps in integer sequence
                seen_nums = set(x[0] for x in observed_labels)
                min_num = min(seen_nums)
                max_num = max(seen_nums)
                
                for num in range(min_num, max_num + 1):
                    if num not in seen_nums:
                        missing[chain_id].append(str(num))

    except Exception:
        pass

    # Finalize: remove duplicates while preserving order
    result = {}
    for ch, vals in missing.items():
        if len(vals) > 0:
            # Remove duplicates preserving order
            seen = set()
            unique_vals = []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)
            result[ch] = unique_vals
    
    return result


def write_chain_structure(structure_file, chain, labels, out_path):
    """Write one chain of the first model of `structure_file` to `out_path` (.pdb as PDB, .cif as mmCIF), keeping
    only the residues `labels` (author labels: the residues of the output, modified residues included) in file order.
    Water and ligands are left out."""
    is_cif = structure_file.lower().endswith(('.cif', '.mmcif'))
    structure = (MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)).get_structure('s', structure_file)
    first = next(iter(structure)).id
    keep = {str(l) for l in labels}

    class _Chain(Select):
        def accept_model(self, model):
            return model.id == first

        def accept_chain(self, ch):
            return ch.id == chain

        def accept_residue(self, residue):
            return f'{residue.id[1]}{residue.id[2].strip()}' in keep

    if out_path.lower().endswith('.cif'):
        from Bio.PDB import MMCIFIO
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, _Chain())


def extend_missing_res(list_label, list_residue, res_labels=None):
    """
    Extend labels to include missing residues.
    
    Args:
        list_label: cluster labels (same length as list_residue)
        list_residue: contiguous integers [1,2,3,...]
        res_labels: original PDB labels ["1","2","11A","11B",...] (optional)
    
    Returns:
        list_label_ext: extended labels
        list_residue_ext: extended residue numbers
        res_labels_ext: extended original labels (if provided)
    """
    max_res = max(list_residue)
    min_res = min(list_residue)
    list_residue_ext = list(range(min_res, max_res + 1))
    
    # Use a dictionary for fast lookups
    residue_to_label = dict(zip(list_residue, list_label))
    
    # Initialize the output label list
    list_label_ext = []
    last_label = list_label[0]

    for i in list_residue_ext:
        if i in residue_to_label:
            last_label = residue_to_label[i]
        list_label_ext.append(last_label)
    
    # Handle res_labels if provided
    if res_labels is not None:
        residue_to_orig_label = dict(zip(list_residue, res_labels))
        res_labels_ext = []
        last_orig_label = res_labels[0]
        
        for i in list_residue_ext:
            if i in residue_to_orig_label:
                last_orig_label = residue_to_orig_label[i]
            res_labels_ext.append(last_orig_label)
        
        return list_label_ext, list_residue_ext, res_labels_ext
    
    return list_label_ext, list_residue_ext


def domain_overlap_matrix(lists_label, list_residue = None): #Order in lists_label: ground_truth, prediction 
    if list_residue == None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}

    group_residue = {'pred': {}, 'true': {}}

    for key in group_label.keys():
        for label in set(group_label[key]):
            group_residue[key][label] = [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label]

    domain_matrix = []
    for label in sorted(set(group_label['pred'])): 
        row = []
        for label2 in sorted(set(group_label['true'])):
            row += [len([intersect for intersect in group_residue['pred'][label] if intersect in group_residue['true'][label2]])]

        domain_matrix += [row]


    min_labels = [min(set(group_label['pred'])), min(set(group_label['true']))]
    return domain_matrix, min_labels


def NDO(domain_matrix, len_chain, min_labels=[0, 0]):
    domain_matrix_no_linker = np.array([row[(min_labels[1] == -1):] for row in domain_matrix[(min_labels[0] == -1):]])
    domain_matrix = np.array(domain_matrix)
    
    sum_col = np.sum(domain_matrix, axis=0)
    sum_row = np.sum(domain_matrix, axis=1)
    try:
        max_col = np.amax(domain_matrix_no_linker, axis=0)
        max_row = np.amax(domain_matrix_no_linker, axis=1)
    except:
        max_col = 0
        max_row = 0

    Y = sum(2 * max_row - sum_row[(min_labels[0] == -1):]) + sum(2 * max_col - sum_col[(min_labels[1] == -1):])
    score = Y / (2 * (len_chain - sum_col[0] * (min_labels[1] == -1)))
    return score


def domain_distance(segment1, segment2):
    return (abs(min(segment1) - min(segment2)) + abs(max(segment1) - max(segment2))) / 2


def csd_distance_matrix(lists_label, list_residue=None):
    """Distance matrix between predicted (rows) and true (columns) domain segments for CSD: the mean offset of the two
    segment ends, where an end next to a linker (label -1) may also be placed at the far side of that linker (the
    smaller distance is kept).
    lists_label = [true_labels, pred_labels]."""
    if list_residue is None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}
    pred_outlier = [list_residue[i] for i in range(len(list_residue)) if group_label['pred'][i] == -1]
    true_outlier = [list_residue[i] for i in range(len(list_residue)) if group_label['true'][i] == -1]
    
    outlier_pos = {'pred': [], 'true': []}
    if pred_outlier:
        outlier_pos['pred'] = list_to_range(pred_outlier)
    if true_outlier:
        outlier_pos['true'] = list_to_range(true_outlier)
    
    for key in outlier_pos:
        if not bool(outlier_pos[key]):
            outlier_pos[key] = [[-9999]]
    
    group_residue = {key: {label: [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label] for label in set(group_label[key])} for key in group_label}
    
    domain_distance_mtx = []
    for label1 in sorted(set(group_label['pred']) - {-1}):
        for segment1 in list_to_range(group_residue['pred'][label1]):
            row = []
            for label2 in sorted(set(group_label['true']) - {-1}):
                for segment2 in list_to_range(group_residue['true'][label2]):
                    score = domain_distance(segment2, segment1)
                    # Check if segment boundaries are adjacent to a linker
                    lst1_min = [min(segment1)]; lst2_min = [min(segment2)]
                    lst1_max = [max(segment1)]; lst2_max = [max(segment2)]
                    for i, j in itertools.product(outlier_pos['pred'], outlier_pos['true']):
                        # Adjust start position if next to a linkers
                        if min(segment1) == i[-1] + 1:
                            lst1_min += [i[0]]
                        if min(segment2) == j[-1] + 1:
                            lst2_min += [j[0]]
                        # Adjust end position if next to a linker
                        if max(segment1) == i[0] - 1:
                            lst1_max += [i[-1]]
                        if max(segment2) == j[0] - 1:
                            lst2_max += [j[-1]]

                    for min_pos in itertools.product(lst1_min, lst2_min):
                        for max_pos in itertools.product(lst1_max, lst2_max):
                            new_score = domain_distance(range(min_pos[1], max_pos[1] + 1),
                                                        range(min_pos[0], max_pos[0] + 1))
                        if new_score < score:
                            score = new_score

                    row.append(score)

            domain_distance_mtx.append(row)
    
    if len(domain_distance_mtx) == 0:
        return [[999999]]
    
    return domain_distance_mtx


def dbd_distance_matrix(lists_label, list_residue=None):
    """
    Distance matrix between predicted and true boundary entities, for DBD.
    - Entities include:
        * point boundaries at domain↔domain changes (both sides != -1)
        * internal linker ranges (label == -1), excluding terminal linkers
    - Matrix shape: [#pred_entities x #true_entities]
    - Distance: 0 if overlap (point in range or ranges overlap), else minimal absolute residue gap.

    Args
    ----
    lists_label : [true_labels, pred_labels]
        Each is a list of ints per residue; -1 denotes linker.
    list_residue : optional list/range of residue indices (same length as labels).
        Defaults to range(len(true_labels)).

    Returns
    -------
    list[list[int]] : distance matrix (rows=pred entities, cols=true entities)
    """
    true_labels, pred_labels = lists_label
    
    # Count distinct domain IDs, ignoring -1 (linker/outlier)
    num_dom_true = len(set(true_labels) - {-1})
    num_dom_pred = len(set(pred_labels) - {-1})

    # If exactly one side is single-domain and the other is multi-domain, penalize
    if (num_dom_true == 1) ^ (num_dom_pred == 1):
        return [[9999999]]
    
    if num_dom_true == 1 and num_dom_pred == 1:
        return [[0]]
    
    if num_dom_pred == 0 or num_dom_true == 0:
        if num_dom_pred != num_dom_true:
            return [[0]]

    if list_residue is None:
        list_residue = list(range(len(true_labels)))

    # ---- helpers ----
    def runs(labels, residues):
        """Yield inclusive runs as (a, b, val) in residue coordinates."""
        if not labels:
            return
        s = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                yield residues[s], residues[i-1], labels[i-1]
                s = i
        yield residues[s], residues[len(labels)-1], labels[-1]

    def internal_linker_ranges(labels, residues):
        """All non-terminal linker runs (label == -1)."""
        labels = [int(i) for i in labels]
        if not labels:
            return []
        res0, resN = residues[0], residues[-1]
        out = []
        for a, b, v in runs(labels, residues):
            if v == -1:
                # skip terminal linkers
                if a == res0 or b == resN:
                    continue
                out.append((a, b))
        return out

    def domain_domain_points(labels, residues):
        """Positions (as singleton ranges) where label changes and both sides are domains (≠ -1)."""
        pts = []
        prev = labels[0]
        for i in range(1, len(labels)):
            cur = labels[i]
            if cur != prev and prev != -1 and cur != -1:
                # boundary is the first residue of the right domain
                pts.append((residues[i], residues[i]))  # store as (a,b) singleton
            prev = cur
        return pts

    def distance_between_ranges(r1, r2):
        """Min distance between two inclusive ranges (a1,b1) and (a2,b2); 0 if they overlap."""
        (a1, b1), (a2, b2) = r1, r2
        # overlap?
        if not (b1 < a2 or b2 < a1):
            return 0
        # non-overlap: minimal gap
        if b1 < a2:
            return a2 - b1
        else:
            return a1 - b2

    # ---- collect TRUE entities (columns) ----
    true_points  = domain_domain_points(true_labels, list_residue)        # [(x,x), ...]
    true_linkers = internal_linker_ranges(true_labels, list_residue)      # [(a,b), ...]
    true_entities = true_points + true_linkers                             # ranges uniformly

    # ---- collect PRED entities (rows) ----
    pred_points  = domain_domain_points(pred_labels, list_residue)
    pred_linkers = internal_linker_ranges(pred_labels, list_residue)
    pred_entities = pred_points + pred_linkers

    # If either side has no entities, return an empty matrix
    if len(pred_entities) == 0 or len(true_entities) == 0:
        return []

    # ---- build distance matrix ----
    mtx = []
    for pr in pred_entities:
        row = [distance_between_ranges(pr, tr) for tr in true_entities]
        mtx.append(row)

    return mtx


def CSD(domain_distance_mtx, threshold=20):
    scoring_mtx = [[threshold - i if i < threshold else 0 for i in row] for row in domain_distance_mtx]
    
    # Max values by column
    max_by_col = np.max(scoring_mtx, axis=0).tolist() 

    # Max values by row
    max_by_row = np.max(scoring_mtx, axis=1).tolist() 
    
    if len(max_by_row) >= len(max_by_col):
        csd = sum(max_by_row)/(threshold*len(max_by_row))
    else:
        csd = sum(max_by_col)/(threshold*len(max_by_col))

    return csd


def DBD(domain_distance_mtx, threshold=20):
    scoring_mtx = [[threshold - i if i < threshold else 0 for i in row] for row in domain_distance_mtx]
    
    # Max values by column
    sum_all = sum(np.sum(scoring_mtx, axis=0).tolist()) 
    max_by_col = np.max(scoring_mtx, axis=0).tolist()

    # Max values by row
    max_by_row = np.max(scoring_mtx, axis=1).tolist() 
    
    if len(max_by_row) >= len(max_by_col):
        dbd = sum_all/(threshold*len(max_by_row))
    else:
        dbd = sum_all/(threshold*len(max_by_col))

    return dbd


def IoU(domain_matrix, lists_label, conflict='first'):  #Order in lists_label: ground_truth, prediction
    """Length-weighted IoU with a STRICT best-overlap matching.

    Each true domain is matched to the predicted domain it overlaps most, taken over ALL predicted domains.
    When two true domains claim the same predicted domain only one can keep it and the other scores ZERO; it does
    NOT fall back to its second-best partner.

    Which claimant keeps it:
    conflict='first' (DEFAULT) : the earlier true domain (Lau et al. 2023, Supplementary Algorithm 1), so the metric
                                 depends on the order in which the true domains are enumerated.
    conflict='max'             : the claimant with the larger overlap, which removes that order dependence.
    """
    min_labels = [min(set(lists_label[1])), min(set(lists_label[0]))]
    domain_matrix_no_linker = np.array([row[(min_labels[1] == -1):] for row in domain_matrix[(min_labels[0] == -1):]])

    if domain_matrix_no_linker.size == 0:
        return 0

    true_labels = sorted(set(lists_label[0]) - {-1})
    pred_labels = sorted(set(lists_label[1]) - {-1})
    n_col = domain_matrix_no_linker.shape[1]

    pos_true = {c: [i for i, v in enumerate(lists_label[0]) if v == true_labels[c]] for c in range(n_col)}
    pos_pred = {r: [i for i, v in enumerate(lists_label[1]) if v == pred_labels[r]]
                for r in range(domain_matrix_no_linker.shape[0])}

    best_row, best_val = {}, {}
    for c in range(n_col):
        col = domain_matrix_no_linker[:, c]
        r = int(np.argmax(col)) if col.size else None
        best_row[c] = r if (r is not None and col[r] > 0) else None
        best_val[c] = float(col[r]) if best_row[c] is not None else 0.0

    keeper = {}
    for c in range(n_col):
        r = best_row[c]
        if r is None:
            continue
        if r not in keeper:
            keeper[r] = c
        elif conflict == 'max' and best_val[c] > best_val[keeper[r]]:
            keeper[r] = c

    num = 0.0; den = 0.0
    for c in range(n_col):
        w = len(pos_true[c]); den += w
        r = best_row[c]
        if r is None or keeper.get(r) != c:
            continue                                  # lost the contest -> contributes 0, no second choice
        union = len(set(pos_pred[r] + pos_true[c]))
        num += (best_val[c] / union) * w if union else 0.0

    return num / den if den else 0

def _label_key(label):
    """sort key of an author residue label: '-3' -> (-3, ''), '11A' -> (11, 'A')"""
    s = str(label); k = 1 if s.startswith('-') else 0
    while k < len(s) and s[k].isdigit():
        k += 1
    return int(s[:k]), s[k:]


def domain_ranges(domain_labels, chain_labels):
    """Segments of one domain, [(start, end)], sorted by residue number.

    chain_labels: every residue of the output, in chain order. Two residues of the domain that follow each other in the
    chain are one segment when the numbering goes up and no residue of the structure is numbered between them, so a gap
    made only of residues without coordinates does not cut the domain ('2-17+20-48' -> '2-48' when 18-19 are missing);
    a residue of the structure in between (in another domain, or unassigned) still does. Segments are listed in
    residue-number order ('2-13+102-335+381-463' even when 2-13 follows 335 in the file)."""
    pos = {str(l): i for i, l in enumerate(chain_labels)}
    keys = sorted(_label_key(l) for l in chain_labels)
    import bisect
    mem = sorted((str(l) for l in domain_labels if str(l) in pos), key=lambda l: pos[l])
    if not mem:
        return []
    segs = [[mem[0], mem[0]]]
    for l in mem[1:]:
        e = segs[-1][1]
        ke, kl = _label_key(e), _label_key(l)
        if pos[l] == pos[e] + 1 and kl > ke and bisect.bisect_right(keys, ke) == bisect.bisect_left(keys, kl):
            segs[-1][1] = l
        else:
            segs.append([l, l])
    return sorted(((a, b) for a, b in segs), key=lambda ab: _label_key(ab[0]))


def bridged_gaps(segments, chain_labels):
    """Residue numbers without coordinates that the segments of one domain span, as [(first, last)]. domain_ranges joins
    two residues of a domain across such a gap, so every output range (stdout, JSON / .pml PyMOL, .tsv) includes them."""
    have = {_label_key(l)[0] for l in chain_labels}
    gaps = []
    for a, b in segments:
        for n in range(_label_key(a)[0] + 1, _label_key(b)[0]):
            if n in have:
                continue
            if gaps and gaps[-1][1] == n - 1:
                gaps[-1][1] = n
            else:
                gaps.append([n, n])
    return [(str(a), str(b)) for a, b in gaps]


def list_to_range_with_insertions(labels):
    """
    Convert list of residue labels (with insertions) to ranges.
    
    Input: ["1", "2", "11", "11A", "11B", "12", "13"]
    Output: [("1", "2"), ("11", "11B"), ("12", "13")]
    
    Input: ["100", "100A", "100B", "101"]
    Output: [("100", "100B"), ("101", "101")]
    """
    if not labels:
        return []
    
    def extract_base_num(label):
        """Extract integer part from label like '11A' -> 11"""
        return int(''.join(c for c in str(label) if c.isdigit()))
    
    def is_consecutive(prev_label, curr_label):
        """Check if two labels are consecutive (including insertions)"""
        prev_base = extract_base_num(prev_label)
        curr_base = extract_base_num(curr_label)
        
        # Same base number (insertions) or next number
        return curr_base == prev_base or curr_base == prev_base + 1
    
    ranges = []
    start = labels[0]
    prev = labels[0]
    
    for i in range(1, len(labels)):
        curr = labels[i]
        
        if is_consecutive(prev, curr):
            prev = curr
        else:
            # Break in sequence
            ranges.append((start, prev))
            start = curr
            prev = curr
    
    # Add final range
    ranges.append((start, prev))
    return ranges


def format_range_string(range_tuples):
    """
    Format range tuples to display string.
    
    Input: [("1", "2"), ("11", "11B"), ("12", "13")]
    Output: "1-2, 11-11B, 12-13"
    """
    parts = []
    for start, end in range_tuples:
        if start == end:
            parts.append(f"{start}")
        else:
            parts.append(f"{start}-{end}")
    return ", ".join(parts)


def detect_chain_types(structure_file):
    """
    Detect whether each chain in a structure is protein or nucleic acid (only protein chains are parsed).

    Returns:
        dict {chain_id: "protein" or "rna"}
    """
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa

    def _is_nt(resname: str) -> bool:
        r = (resname or "").strip().upper()
        return r in {"A","C","G","U","I","T"} or r in {"DA","DC","DG","DT","DU"} or r.startswith("D")

    fname = (structure_file or "").lower()
    if fname.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif fname.endswith((".cif", ".mmcif")):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("detect_chain_types expects a .pdb or .cif file")

    structure_id = os.path.basename(structure_file).split(".")[0]
    structure = parser.get_structure(structure_id, structure_file)

    chain_types = {}
    # Use first model only (consistent with the rest of the pipeline)
    model = next(structure.get_models())
    for chain in model:
        aa_count = 0
        nt_count = 0
        for res in chain.get_residues():
            hetflag, _, _ = res.id
            if hetflag == "W":
                continue
            if is_aa(res, standard=True):
                aa_count += 1
            else:
                try:
                    if _is_nt(res.get_resname()):
                        nt_count += 1
                except Exception:
                    pass

        if aa_count == 0 and nt_count == 0:
            continue
        chain_types[chain.id] = "rna" if nt_count > aa_count else "protein"

    return chain_types


def get_secondary_structure_stride_auto(structure_file, chain_types=None, chain_filter=None):
    """
    STRIDE wrapper for protein chains.
    Handles mmCIF by converting to temp PDB with remapped chain IDs.
    """
    if chain_types is None:
        chain_types = detect_chain_types(structure_file)
    
    if chain_filter == None:
        protein_chains = [ch for ch, t in chain_types.items() if t == "protein"]
    
    else:
        protein_chains = [ch for ch, t in chain_types.items() if t == chain_filter]
    
    
    if not protein_chains:
        return {}
    
    # ========================================
    # Check if too many protein chains
    # ========================================
    if len(protein_chains) > 62:
        print(f"WARNING: {len(protein_chains)} protein chains detected, which exceeds PDB format limit.")
        print(f"STRIDE secondary structure will be skipped. Clustering will use fallback (no SSE constraints).")
        return {}

    tmp_pdb = None
    chain_mapping = None
    
    try:
        if structure_file.lower().endswith(".pdb"):
            stride_input = structure_file
            chain_mapping = {ch: ch for ch in protein_chains}
        elif structure_file.lower().endswith((".cif", ".mmcif")):
            tmp_pdb, reverse_map = _cif_to_temp_pdb(structure_file, chains_to_keep=protein_chains)
            stride_input = tmp_pdb
            chain_mapping = {v: k for k, v in reverse_map.items()}
        else:
            raise ValueError("Expected .pdb or .cif file")

        ss_pdb = get_secondary_structure_stride(stride_input)
        
        # Remap chain IDs back to original
        ss_result = {}
        for new_id, ss_data in ss_pdb.items():
            original_id = None
            for orig, new in chain_mapping.items():
                if new == new_id:
                    original_id = orig
                    break
            
            if original_id and original_id in protein_chains:
                ss_result[original_id] = ss_data
        
        return ss_result
        
    except Exception as e:
        print(f"Warning: STRIDE processing failed: {e}")
        return {}
        
    finally:
        if tmp_pdb and os.path.exists(tmp_pdb):
            try:
                os.remove(tmp_pdb)
            except Exception:
                pass


def get_secondary_structure_stride(pdb_file, chain_id=None):
    """
    Run STRIDE to get secondary structure annotation.
    STRIDE is installed by the user: BIHC_STRIDE (--stride), src/stride or the PATH (bihc_core.find_program).
    """

    stride_path = _load_core().find_program('stride')
    if stride_path is None:
        raise FileNotFoundError(_load_core().missing_program_message('stride'))

    # Create temp file for output
    with tempfile.NamedTemporaryFile(mode='w', suffix='_ss', delete=False) as tmp:
        tmp_path = tmp.name
    
    # Convert PDB path to absolute path (important!)
    pdb_file_abs = os.path.abspath(pdb_file)
    
    # Run STRIDE
    cmd = [stride_path, pdb_file_abs]
    if chain_id:
        cmd = [stride_path, '-r', chain_id, pdb_file_abs]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(tmp_path, 'w') as f:
            f.write(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"STRIDE execution failed: {e}")
        print(f"Command: {' '.join(cmd)}")
        print(f"STDERR: {e.stderr}")
        os.remove(tmp_path)
        return {}
    
    # Parse STRIDE output
    ss_dict = {}
    with open(tmp_path, 'r') as f:
        for line in f:
            if line.startswith('ASG'):
                res_name = line[5:8].strip()
                chain = line[9:10].strip()
                res_num_str = line[10:15].strip()
                ss_type = line[24:25].strip()
                
                # Map to simplified categories
                if ss_type in ['H', 'G', 'I']:  # Helices
                    ss_simple = 'H'
                elif ss_type in ['E', 'B', 'b']:  # Strands; STRIDE writes an isolated bridge as 'B' or 'b' (bihc_core._ss3)
                    ss_simple = 'E'
                else:                            # Coils/Turns
                    ss_simple = 'C'
                
                if chain not in ss_dict:
                    ss_dict[chain] = []
                ss_dict[chain].append((res_num_str, ss_simple))
    
    # Cleanup
    os.remove(tmp_path)
    
    return ss_dict


def _cif_to_temp_pdb(structure_file, chains_to_keep=None):
    """
    Convert mmCIF to temporary PDB, remapping long chain IDs to single characters.
    Only includes chains in chains_to_keep.
    Returns: (tmp_pdb_path, reverse_mapping_dict)
    """
    from Bio.PDB import MMCIFParser, PDBIO
    import string

    parser = MMCIFParser(QUIET=True)
    structure_id = os.path.basename(structure_file).split(".")[0]
    structure = parser.get_structure(structure_id, structure_file)
    model = structure[0]

    # only the chains in chains_to_keep
    if chains_to_keep is None:
        chains_to_process = [ch.id for ch in model]
    else:
        chains_to_process = [ch.id for ch in model if ch.id in set(chains_to_keep)]
    
    # Check if we have too many chains AFTER filtering
    if len(chains_to_process) > 62:
        raise ValueError(
            f"Too many chains to convert ({len(chains_to_process)} > 62). "
            f"chains_to_keep={chains_to_keep}"
        )

    # Create mapping: long chain ID → single char
    available_chars = list(string.ascii_uppercase + string.digits + string.ascii_lowercase)
    chain_mapping = {}  # old_id → new_id
    reverse_mapping = {}  # new_id → old_id
    
    for old_id in chains_to_process:
        new_id = available_chars.pop(0)
        chain_mapping[old_id] = new_id
        reverse_mapping[new_id] = old_id

    # ========================================
    # Create new structure with only selected chains
    # ========================================
    from Bio.PDB import Structure, Model
    
    new_structure = Structure.Structure(structure_id)
    new_model = Model.Model(0)
    new_structure.add(new_model)
    
    for chain in model:
        if chain.id in chain_mapping:
            # Create new chain with remapped ID
            new_chain = chain.copy()
            new_chain.id = chain_mapping[chain.id]
            new_model.add(new_chain)

    # Save to temp PDB
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp_path = tmp.name
    tmp.close()

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(tmp_path)

    return tmp_path, reverse_mapping


