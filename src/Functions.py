import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import itertools
import shutil
from matplotlib import cm
from io import StringIO
import re
import subprocess

from typing import List, Tuple

from scipy.sparse import coo_matrix, csr_matrix

from Bio.PDB.PDBIO import Select
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score
from sklearn.metrics import mutual_info_score, pairwise_distances, silhouette_score, silhouette_samples
from scipy.stats import entropy
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
import tempfile
from Bio.PDB.Polypeptide import is_aa

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from numpyencoder import NumpyEncoder

import os
import json


import tempfile

from scipy.spatial import distance_matrix
import networkx as nx
import igraph as ig


'''import markov_clustering as mc

from infomap import Infomap

from itertools import groupby
from pathlib import Path
from cdlib import algorithms'''

SS_BY_CHAIN = None  # expected: dict {chain_id: ss_info_aligned}

def set_ss_by_chain(ss_by_chain: dict):
    """
    Provide SSE externally for algorithm 'B2'.
    ss_by_chain format:
      { 'A': ss_info, 'B': ss_info, ... }

    Where ss_info can be:
      - ['H','E','C', ...] length N
      - [(res_label,'H'), (res_label,'C'), ...] length N
    """
    global SS_BY_CHAIN
    SS_BY_CHAIN = ss_by_chain
    
    
def _normalize_ss_info(ss_info):
    """
    Accept:
      - ['H','E','C',...]
      - [(res_label,'H'), ...]  (STRIDE-like)
    Return:
      ['H','E','C',...]
    """
    if not ss_info:
        return []
    first = ss_info[0]
    if isinstance(first, tuple) and len(first) >= 2:
        return [str(x[1]) for x in ss_info]
    return [str(x) for x in ss_info]

def _align_ss_to_length(ss_info, N, fill='C'):
    """
    Make sure ss_info length == N.
    If shorter -> pad with fill; if longer -> truncate.
    """
    ss = _normalize_ss_info(ss_info)
    if len(ss) == N:
        return ss_info  # keep original type if already aligned
    if len(ss) < N:
        # pad using normalized type (simple list of chars)
        ss = ss + [fill] * (N - len(ss))
        return ss
    return ss[:N]

    
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
    
def flatten(l):
    result = []
    for sublist in l:
        if isinstance(sublist, list):
            result += sublist
        else:
            result += [sublist]

    return result

def flatten_np(l):
    return np.asarray(np.concatenate(l))

def get_coordinate(x):
    xs = []
    ys = []
    zs = []
    for line in x:
        xs += [float(line[8])]
        ys += [float(line[9])]
        zs += [float(line[10])]
    
    return xs, ys, zs

def make_canonical_seq(res_nums, res_names):
    """
    Given a list of residue numbers and names, return a canonical sequence string with '-' for gaps.
    """
    if not res_nums or not res_names:
        return ''
    # Sort both by residue number
    pairs = sorted(zip(res_nums, res_names), key=lambda x: x[0])
    seq = []
    prev_num = pairs[0][0] - 1
    for num, name in pairs:
        gap = num - prev_num - 1
        if gap > 0:
            seq.append('-' * gap)
        seq.append(MODIFIED_BASES.get(name.upper(), name.upper()))
        prev_num = num
    return ''.join(seq)

MODIFIED_BASES = {
    # Canonical bases
    'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U', 'T': 'T',
    'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'T', 'DI': 'I',

    # Modified adenosines
    '1MA': 'A',  # 1-methyladenosine
    '2MA': 'A',  # 2-methyladenosine
    '6MA': 'A',  # N6-methyladenosine
    'M2A': 'A',  # 2-methyladenosine
    'I': 'A',    # Inosine (sometimes mapped to A)
    'A2M': 'A',  # 2,2-dimethyladenosine
    'M6A': 'A',  # N6-methyladenosine
    'MIA': 'A',  # N6-isopentenyladenosine
    'RIA': 'A',  # N6-(Ribosylcarboxyamino)adenosine
    'MA6': 'A',   # N6-methyladenosine

    # Modified cytidines
    '5MC': 'C',  # 5-methylcytidine
    'OMC': 'C',  # 2'-O-methylcytidine
    'CBR': 'C',  # 5-bromocytidine
    'CBV': 'C',  # 5-bromovinylcytidine
    'CCC': 'C',  # 5-carboxycytidine
    'M5C': 'C',  # 5-methylcytosine
    'RCY': 'C',  # 5-carboxymethylaminomethylcytidine
    'MCT': 'C',  # 5-methylcytidine
    '4AC': 'C',   # 4-acetylcytidine

    # Modified guanosines
    'M2G': 'G',  # N2-methylguanosine
    '7MG': 'G',  # 7-methylguanosine
    'OMG': 'G',  # 2'-O-methylguanosine
    '2MG': 'G',  # N2,7-dimethylguanosine
    'G7M': 'G',  # 7-methylguanosine
    'Y': 'G',    # Wybutosine (sometimes mapped to G)
    'YG': 'G',   # Wybutosine
    'GFM': 'G',  # 2′-O-(2-methylthio)guanosine

    # Modified uridines
    '5MU': 'U',  # 5-methyluridine
    'H2U': 'U',  # 5,6-dihydrouridine
    'PSU': 'U',  # Pseudouridine
    'OMU': 'U',  # 2'-O-methyluridine
    'M2U': 'U',  # 2-methyluridine
    'D': 'U',    # Dihydrouridine (alternate)
    'S4U': 'U',  # 4-thiouridine
    '4SU': 'U',  # 4-thiouridine
    'MSE': 'U',  # 2-selenouridine

    # Modified thymidines
    'T6A': 'T',  # N6-threonylcarbamoyladenosine (sometimes T, sometimes A)
    '5MT': 'T',  # 5-methylthymidine

    # Others (rare or synthetic bases)
    'X': 'N',    # Unknown base
    'N': 'N',    # Any base

    # For completeness (common in DNA structures)
    'ADE': 'A', 'GUA': 'G', 'CYT': 'C', 'THY': 'T', 'URA': 'U',
    
    'B8N': 'G',   # 7,8-dihydro-8-oxoguanosine
    '6MZ': 'A',   # N6,N6-dimethyladenosine
    'UY1': 'U',   # 5-methoxyuridine
    'UR3': 'U',   # 3-(3-amino-3-carboxypropyl)uridine
    'GTP': 'G',   # Guanosine triphosphate
    'JMH': 'A',   # 2'-O-methyladenosine
    'C4J': 'C',   # N4-acetylcytidine
    'XSX': 'U',
    'JMC': 'C',
    'B8H': 'G',
    
}

def process_pdb(list_format, atom_type='C3', models=True, get_seq=False):
    coor_atoms_C = []
    chains = []
    res_num = []
    result = []
    res = []
    l = [(0,6),(6,11),(12,16),(16,17),(17,20),(20,22),(22,26),
         (26,27),(30,37),(38,46),(46,54),(54,60),(60,66),(72,76),
          (76,78),(78,80)]
    model = ''
    num_model = 0
    for line in list_format:
        if 'MODEL' in line[:5]:
            num_model += 1
            if models == False and num_model > 1:
                break
            model = line.replace(' ','')

        model = '' # Adapt the web-server
        if ("ATOM" in line[:6].replace(" ","") and (len(line[17:20].replace(" ","")) == 1 or line[17:20].replace(" ","")[0] == "D")) and atom_type in line[12:16]:
            new_line = [line[v[0]:v[1]].replace(" ","") for v in l ] + [model]

            chain_name = new_line[5]
            if chain_name not in chains:
                chains += [chain_name]

            coor_atoms_C += [new_line]

    if not chains:
        return False
    
    for chain in chains:
        sub_coor_atoms_C = [new_line for new_line in coor_atoms_C if new_line[5] == chain.split('_')[0]
                            and new_line[16] == ''.join(chain.split('_')[1:])]
        result += [get_coordinate(sub_coor_atoms_C)]
        res_num += [[int(new_line[6]) for new_line in coor_atoms_C if new_line[5] == chain.split('_')[0]
                            and new_line[16] == ''.join(chain.split('_')[1:])]]

        res += [[new_line[4] for new_line in coor_atoms_C if new_line[5] == chain.split('_')[0] 
                 and new_line[16] == ''.join(chain.split('_')[1:])]]
    
    # Build canonical sequence with gap for each chain
    seqs = []
    for nums, names in zip(res_num, res):
        seqs.append(make_canonical_seq(nums, names))

    if get_seq:
        return result, chains, res_num, seqs
    else:
        return result, chains, res_num

def process_structure(input_data, atom_type="C3'", models=True, get_res=False, filename=None, chain_filter=None):
    """
    Unified parser for BOTH PDB and mmCIF that supports BOTH RNA and protein chains.

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
    elif fname.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
        if is_list:
            structure = parser.get_structure("model", StringIO("\n".join(input_data)))
        else:
            structure = parser.get_structure("model", input_data)
    else:
        raise ValueError("Unknown file type: expected .pdb or .cif")

    # Helpers
    def _is_rna_resname(resname: str) -> bool:
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
                if _is_rna_resname(res.get_resname()):
                    nt_count += 1
            except Exception:
                pass
        return "rna" if nt_count > aa_count else "protein"

    def _default_atom(chain_type: str) -> str:
        return "C3'" if chain_type == "rna" else "CA"

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
            chain_type = _detect_chain_type(residues)

            # Try requested atom first; if empty, fallback by chain type
            atoms_to_try = [atom_type]
            fb = _default_atom(chain_type)
            if fb != atom_type:
                atoms_to_try.append(fb)

            xs = ys = zs = None
            labels = None
            chosen = None

            for at in atoms_to_try:
                _xs, _ys, _zs = [], [], []
                _labels = []

                for residue in residues:
                    hetflag, seq_id, icode = residue.id
                    if hetflag == "W":
                        continue

                    # Keep polymer residues relevant to chain type
                    if chain_type == "protein":
                        if not is_aa(residue, standard=True):
                            continue
                    else:
                        try:
                            if not _is_rna_resname(residue.get_resname()):
                                continue
                        except Exception:
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

def generate_colors(num_colors):
    colormap = cm.get_cmap('hsv', num_colors)
    return [colormap(i) for i in range(num_colors)]

def pymol_process(pred, res_num, name=None, color=None, verbose=False, res_labels=None):
    """
    Generate PyMOL commands for coloring clusters.
    
    Args:
        pred: cluster labels
        res_num: contiguous residue numbers [1,2,3,...]
        name: prefix for selection names
        color: list of colors
        verbose: print commands
        res_labels: original PDB residue labels (optional)
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
            cmd.append(command_pymol_with_labels(label_list, clust_name, 'grey', verbose))
        else:
            clust_name = name + f'_cluster_{num+1}' if name is not None else f'cluster_{num+1}'
            cmd.append(command_pymol_with_labels(label_list, clust_name, color[num], verbose))

    return cmd

def command_pymol_with_labels(label_list, name, color, verbose=False):
    """
    Generate PyMOL command using original residue labels (handles insertions).
    
    Args:
        label_list: list of residue labels like ["1", "2", "11A", "11B", "12"]
        name: selection name
        color: color name
        verbose: print command
    """
    # Group consecutive ranges
    ranges = list_to_range_with_insertions(label_list)
    
    mess = f'select {name}, res '
    for p, (start, end) in enumerate(ranges):
        if start == end:
            mess += f'{start}'
        else:
            mess += f'{start}-{end}'
        
        if p != len(ranges) - 1:
            mess += '+'
    
    mess += f'; color {color}, {name}'
    
    if verbose:
        print(mess)
    
    return mess

def command_pymol(l, name, color, verbose = False):
    l2 = list_to_range(l)
    mess = f'select {name}, res '
    for p,r in enumerate(l2):
        if len(r) > 1:
            mess += f'{r[0]}-{r[-1]}'
            if p != len(l2) - 1:
                mess += '+'
        else:
            mess += f'{r[0]}' + '+'
    mess += f'; color {color}, {name}'
    if verbose:
        print(mess)
    
    return mess

def distance_2arrays(arr1, arr2):
    dist = 1
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            dist -= 1/len(arr1)
    
    return dist

def join_clusters(list_cluster):
    prev_result = list_cluster
    result = []
    cont = True
    while cont:
        cont = False
        for cluster1, cluster2 in itertools.combinations(prev_result, 2):
            if any(i in cluster1 for i in cluster2):
                if cluster1 in result:
                    result.remove(cluster1)
                if cluster2 in result:
                    result.remove(cluster2)
                
                result += [tuple(set(cluster1 + cluster2))]
                cont = True
        
        result = list(set(result))
        prev_result = result

    return prev_result

def cluster_algo(*args):
    data = args[0]
    print('-'*40)

    # ====== community/custom group (return CLUSTERS) ======
    if args[1] in ['B']:
        edge_w = True if args[-5].lower() == 'true' else False
        dist_th = args[-4]
        chain_id = args[-2]
        chain_type = args[-1]

        if dist_th == None:
            if chain_type == 'protein':
                dist_th = 7.5
            else:
                dist_th = 15
        
        min_len = 10 if chain_type == 'protein' else 30
        
        G = build_graph(data, distance=dist_th, weight=edge_w)
        nodes = list(G.nodes)
        N = len(nodes)

        print(f"Executing BiHC on chain {args[-2]}...")
        split_mod_thresh = float(args[2])
        merge_ratio_thresh = float(args[3])
        resolution = float(args[4])

        # SSE: provided externally via set_ss_by_chain(...)
        ss_info = None
        if SS_BY_CHAIN is not None and isinstance(SS_BY_CHAIN, dict):
            ss_info = SS_BY_CHAIN.get(chain_id, None)

        if ss_info is None:
            # fallback to original BiHC top-down if no SSE
            segments = top_down(
                G, len(data),
                split_thresh=split_mod_thresh,
                min_len=min_len,
                resolution=resolution,
                discont=True
            )
        else:
            # ensure SSE length matches N to avoid mismatch
            ss_info_aligned = _align_ss_to_length(ss_info, len(data), fill='C')

            segments = top_down_stride(
                G, len(data),
                ss_info=ss_info_aligned,
                split_thresh=split_mod_thresh,
                min_len=min_len,
                resolution=resolution,
                discont=True,
                boundary_only=False,
                coil_step=6
            )

        pred = bottom_up(G, segments, ratio_thresh=merge_ratio_thresh) 

        labels = clusters_to_labels(pred, N)

        return np.asarray(labels, dtype=int)

    else:
        print(args[1])
        sys.exit("Unrecognized algorithm!")

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
    if len(result) == 4:  # New format: has res_labels
        coords_list, chain_ids, res_num_list, res_labels_list = result
    else:  # Old format: no res_labels
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
    

def make_list_of_ranges(res_list, cluster_list):
    list_of_ranges = []; l = [res_list[0]]
    for p,v in enumerate(res_list):
        if p == 0:
            continue

        if cluster_list[p] == cluster_list[p-1]:
            if (cluster_list[p] != -1 and res_list[p] - res_list[p-1] == 1) or cluster_list[p] == -1:
                l += [v]
        else:
            list_of_ranges += [l]
            l = [v]

        if p == len(res_list) - 1:
            list_of_ranges += [l]

    return list_of_ranges

def find_missing_res(file):
    """
    Missing residues for BOTH protein and RNA, for BOTH PDB and mmCIF.
    
    Returns: dict {chain_id: list of missing residue labels with insertions}
             Example: {'A': ['10', '11', '11A'], 'B': ['50']}
    """
    from collections import defaultdict
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa

    missing = defaultdict(list)  # Changed from set to list to preserve order
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

    elif fname.endswith(".cif"):
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


def post_process_update(cluster_list, res_list = False, missing_res = False, segment_length = [30,10,100]):
    l = segment_length
    cluster_list2 = cluster_list.copy()
    if res_list == False:
        res_list = list(range(len(cluster_list2)))
    
    if missing_res == False:
        missing_res = [i for i in range(min(res_list), max(res_list)+1) if i not in res_list]
    s = 0
    while True:
        list_of_ranges = make_list_of_ranges(res_list, cluster_list2)
        if len(list_of_ranges) == 1:
            break

        label_segments = [cluster_list2[res_list.index(p[0])] for p in list_of_ranges] # Get the labels of the segments
        
        pos_to_change = []
        for pos, ranges in enumerate(list_of_ranges):
            label = label_segments[pos]
            # If the selected segment is an outlier segment
            if label == -1:
                # If the length of outlier segment is 1
                if len(ranges) == 1:
                    val = label_segments[pos-1]
                    pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                # If the selected outlier segment is the first or last segment
                # If the selected outlier segment is the first segment
                if pos == 0:
                    if label_segments[pos+1] != -1 and len(ranges) not in range(l[1],l[2]) and len(list_of_ranges[pos+1]) >= l[0]:
                        val = label_segments[pos+1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])  
                
                # If the selected outlier segment is the last segment
                elif pos == len(list_of_ranges) - 1:
                    if label_segments[pos-1] != -1 and len(ranges) not in range(l[1],l[2]) and len(list_of_ranges[pos-1]) >= l[0]:
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])

                else:
                    # If 2 segments on both sides have the same label, label the outliner that label
                    if label_segments[pos-1] == label_segments[pos+1] and label_segments[pos-1] != -1:
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                
                    # If 2 segments on both sides have different labels, label the outliner half-half if it's < 10
                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] != -1 and label_segments[pos+1] != -1 and len(ranges) < l[1]:
                        val1 = label_segments[pos-1]
                        val2 = label_segments[pos+1]
                        if len(ranges) == 1:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[:int(len(ranges)/2)]])
                            pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[int(len(ranges)/2):]])
                
            # If the selected segment is not an outlier segment
            else:
                # If the selected segment is the first or last segment
                # If the selected segment is the first segment
                if pos == 0:
                    if label_segments[pos+1] != -1 and len(ranges) < l[0] and len(list_of_ranges[pos+1]) > len(ranges): # > 30
                        val = label_segments[pos+1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    elif label_segments[pos+1] == -1 and len(ranges) < l[0]:
                        if len(list_of_ranges[pos+1]) > len(ranges):
                            val = label_segments[pos+1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            if pos + 2 < len(list_of_ranges):
                                val1 = label_segments[pos]
                                val2 = label_segments[pos+2]
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos+1][:int(len(list_of_ranges[pos+1])/2)]])
                                pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos+1][int(len(list_of_ranges[pos+1])/2):]])
                        
                # If the selected segment is the last segment
                elif pos == len(list_of_ranges) - 1:
                    if label_segments[pos-1] != -1 and len(ranges) < l[0] and len(list_of_ranges[pos-1]) > len(ranges): # > 30
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    elif label_segments[pos-1] == -1 and len(ranges) < l[0]:
                        if len(list_of_ranges[pos-1]) > len(ranges):
                            val = label_segments[pos-1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            if pos >= 2:
                                val1 = label_segments[pos-2]
                                val2 = label_segments[pos]
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos-1][:int(len(list_of_ranges[pos-1])/2)]])
                                pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos-1][int(len(list_of_ranges[pos-1])/2):]])
                else:
                    if label_segments[pos-1] == label_segments[pos+1] and label_segments[pos-1] == -1 and len(ranges) < l[0] and len(list_of_ranges[pos-1]) >= l[0] and len(list_of_ranges[pos+1]) >= l[0]:
                        val = -1
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    elif label_segments[pos-1] == label_segments[pos+1] and label_segments[pos-1] != -1 and len(ranges) < l[0] and len(list_of_ranges[pos-1]) + len(list_of_ranges[pos+1]) > len(ranges):
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])

                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] != -1 and label_segments[pos+1] != -1 and len(ranges) < l[0] and len(list_of_ranges[pos-1]) > l[0] and len(list_of_ranges[pos+1]) > l[0]:
                        val1 = label_segments[pos-1]
                        val2 = label_segments[pos+1]
                        if len(ranges) == 1:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[:int(len(ranges)/2)]])
                            pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[int(len(ranges)/2):]])

                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] == -1 and label_segments[pos+1] != -1 and len(ranges) < l[0] and len(list_of_ranges[pos+1]) > l[0]:
                        val = label_segments[pos+1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    
                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] != -1 and label_segments[pos+1] == -1 and len(ranges) < l[0] and len(list_of_ranges[pos-1]) > l[0]:
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    
                    elif len(ranges) < l[0]:
                        if len(list_of_ranges[pos-1]) > len(ranges):
                            if len(list_of_ranges[pos-1]) > len(list_of_ranges[pos+1]):
                                val = label_segments[pos-1]
                            else:
                                val = label_segments[pos+1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        elif len(list_of_ranges[pos+1]) > len(ranges):
                            val = label_segments[pos+1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])                    
        
                    
        for new_pos,new_val in pos_to_change:
            cluster_list2[new_pos] = new_val

        # Check if any changes were made
        if bool(pos_to_change) == False or s == 30:
            break

        s += 1

    return cluster_list2  

def SDC(truth,pred, outliner = False):
    if outliner == False:
        truth = [i for i in truth if i != -1]
        pred =  [i for i in pred if i != -1]
    
    count_truth = len(set(truth))
    count_pred = len(set(pred))
    
    sdc = (max((count_truth), count_pred) - abs(count_truth - count_pred))/max(count_truth, count_pred)
    
    return sdc

def inf(tup_of_tups, val):
    start_eles = [min(tup) for tup in tup_of_tups if min(tup) <= val]
    end_eles = [max(tup) for tup in tup_of_tups if max(tup) <= val]
    
    inf_start = max(start_eles) if bool(start_eles) != False else -1
    inf_end = max(end_eles) if bool(end_eles) != False else -1

    return inf_start, inf_end

def sup(tup_of_tups, val):
    start_eles = [min(tup) for tup in tup_of_tups if min(tup) >= val]
    end_eles = [max(tup) for tup in tup_of_tups if max(tup) >= val]
    
    sup_start = min(start_eles) if bool(start_eles) != False else val
    sup_end = min(end_eles) if bool(end_eles) != False else val

    return sup_start, sup_end

def find_tuple_index(tuples, element, order = False):
    for i, t in enumerate(tuples):
        if element in t:
            if order == False:
                return i
            else:
                if order == "first":
                    if element == t[0]:
                        return i
                else:
                    if element == t[-1]:
                        return i
    return -1  # Return -1 if the element is not found in any tuple

def contact_prob(d, d0 = 8, sig = 1.5):
    p = 1/(1+np.exp((d - d0)/sig))
    
    return p

def DISinter(D1, D2, alpha=0.43, d0=8, sig=1.5):
    D1 = np.array(D1)
    D2 = np.array(D2)

    dists = np.linalg.norm(D1[:, np.newaxis] - D2, axis=2)
    probs = contact_prob(dists, d0, sig)

    s = np.sum(probs)
    s *= 1 / ((len(D1) * len(D2)) ** alpha)

    return s

def DISintra(D, indices = None, beta = 0.95, d0 = 8, sig = 1.5):
    l = len(D)
    if indices == None:
        indices = range(len(D))
    
    s = 0
    for i1, i2 in itertools.combinations(range(len(indices)),2):
        if abs(indices[i1]-indices[i2]) <= 2:
            continue
        
        d1 = np.array(D[i1]); d2 = np.array(D[i2])
        
        d = np.linalg.norm(d2 - d1)
        p = contact_prob(d,d0,sig)
        
        s += p
    
    s *= 1/(l**beta)
    
    return s

def largest_smaller_than(lst, value):
    # Initialize variables to store the largest element found and its index
    largest = None
    largest_index = -1
    
    # Iterate through the list with index
    for index, elem in enumerate(lst):
        # Check if the element is smaller than the given value
        if elem <= value:
            # If largest is None or current element is larger than largest found so far
            if largest is None or elem > largest:
                largest = elem
                largest_index = index
    
    return largest, largest_index,value

def tup_pos_process(tup_of_tup):
    result = []
    for tup in tup_of_tup:
        s = []
        for i in tup:
            s += list(range(i[0], i[1]))
        
        s = tuple(s)
        result += [s]
    
    return result

def process_cluster_format(clust_lst, res_lst = None):
    if res_lst == None:
        res_lst = list(range(1,len(clust_lst)+1))

    clust_by_res = []
    set_clust = set(clust_lst)
    for clust in set_clust:
        sublst = []

        for pos, res in enumerate(res_lst):
            if clust_lst[pos] ==  clust:
                sublst += [res]

        clust_by_res += [sublst]

    return clust_by_res
    
def split_pdb_by_clusters(pdb_file, clusters, output_prefix, chain=None):
    """
    Splits a PDB (or CIF) file into multiple PDB files based on provided clusters of residues.
    If a chain is specified, only residues from that chain will be processed.

    Parameters:
        pdb_file (str): Path to the input PDB or CIF file.
        clusters (list[list[int]]): Each cluster is a list of residue indices (PDB numbering).
        output_prefix (str): Prefix for output files (not used here; function returns lines).
        chain (str | None): Chain ID filter. If None, all chains are processed.

    Returns:
        dict[int, list[str]]: Map cluster_index -> list of PDB-formatted ATOM/HETATM lines
    """

    # If input is CIF, convert to a temp PDB first so the fixed-column parsing works.
    temp_pdb_path = None
    source_path = pdb_file
    if pdb_file.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("model", pdb_file)

        # --- NEW: select only requested chain(s) ---
        wanted = None
        if chain:
            wanted = set([chain])  # chain is original (can be 'BG')

        class _Sel(Select):
            def accept_chain(self, ch):
                if wanted is None:
                    return 1
                return 1 if ch.id in wanted else 0

        # --- NEW: remap long chain IDs for PDB writing ---
        # If only one chain is selected and it has length>1, rename it to 'A'
        if wanted is not None:
            # mutate structure in-memory (temp)
            for model in structure:
                for ch in model:
                    if ch.id in wanted and len(ch.id) > 1:
                        ch.id = "A"

        io = PDBIO()
        io.set_structure(structure)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        tmp.close()
        io.save(tmp.name, _Sel())
        temp_pdb_path = tmp.name
        source_path = temp_pdb_path

    # Read the (now guaranteed) PDB file
    with open(source_path, "r") as fh:
        lines = fh.readlines()

    # Prepare containers
    cluster_lines = {i: [] for i in range(len(clusters))}
    cluster_sets = [set(c) for c in clusters]  # O(1) membership checks

    # Parse PDB fixed columns
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # PDB format: residue sequence number in cols 23–26 (0-based 22:26)
            resseq_str = line[22:26].strip()
            if not resseq_str or not resseq_str[0].isdigit():
                # Skip any malformed or non-standard residue numbering lines
                continue
            try:
                residue_seq = int(resseq_str)
            except ValueError:
                # Handle cases like insertion codes embedded strangely, e.g., "123A"
                # Strip trailing non-digits if present
                num = ""
                for ch in resseq_str:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if not num:
                    continue
                residue_seq = int(num)

            residue_chain = line[21:22].strip()

            if chain and residue_chain != chain:
                continue

            # Assign to the first matching cluster
            for idx, cset in enumerate(cluster_sets):
                if residue_seq in cset:
                    cluster_lines[idx].append(line)
                    break

    # Cleanup temp file if we made one
    if temp_pdb_path and os.path.exists(temp_pdb_path):
        try:
            os.remove(temp_pdb_path)
        except OSError:
            pass

    return cluster_lines, False


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

def _sigmoid_contact_weight(d, d0=8.0, sigma=1.5):
    d = np.asarray(d, dtype=float)
    return 1.0 / (1.0 + np.exp((d - d0) / sigma))

def build_graph(data, distance=8.0, weight=True, segments=None, d0=8.0, sigma=1.5):
    coords = np.asarray(data, dtype=float)
    N = len(coords)
    D = distance_matrix(coords, coords)

    if weight is True:
        def w_of(d): return float(_sigmoid_contact_weight(d, d0=d0, sigma=sigma))
    else:
        def w_of(d): return 1.0

    G = nx.Graph()
    for i in range(N):
        G.add_node(i, pos=(coords[i, 0], coords[i, 1]))

    # backbone edges always
    for i in range(N - 1):
        G.add_edge(i, i + 1, weight=w_of(D[i, i + 1]))

    # contact edges (non-consecutive)
    for i in range(N):
        for j in range(i + 2, N):
            if D[i, j] < distance:
                G.add_edge(i, j, weight=w_of(D[i, j]))

    return G

def find_outlier(G, label, threshold = 2):
    #print(label, len(label))
    outliers = []
    for cluster in label:
        if len(cluster) == 0:
            continue
        segments = list_to_range(cluster)
        for segment in segments:
            for node in segment:
                if G.degree(node) <= threshold:
                    outliers += [node]
                else:
                    break
            for node in segment[::-1]:
                if G.degree(node) <= threshold:
                    outliers += [node]
                else:
                    break
    
    if outliers == []:
        return outliers
            
    outliers = list(set(outliers))
    
    outliers = list_to_range(outliers)
    
    outliers = flatten([list(range_lst) for range_lst in outliers if len(range_lst) >= 10])
    
    return outliers

def merge_small_clusters(labels, min_size=30):    
    clustered = [(key, len(list(group))) for key, group in groupby(labels)]
    new_labels = labels
    if len(labels) < min_size:
        return [-1] * len(labels)
    
    s = 0
    while any([j[1] < min_size for j in clustered if j[0] != -1]):
        new_labels = []
        i = 0
        while i < len(clustered):
            label, size = clustered[i]
            if label != -1 and size < min_size:
                prev_cluster = clustered[i - 1] if i > 0 else None
                next_cluster = clustered[i + 1] if i < len(clustered) - 1 else None
                
                if prev_cluster and prev_cluster[0] != -1 and (not next_cluster or prev_cluster[1] >= next_cluster[1]):
                    new_labels.extend([prev_cluster[0]] * size)
                elif next_cluster and next_cluster[0] != -1:
                    new_labels.extend([next_cluster[0]] * size)
                else:
                    new_labels.extend([-1] * size)

            else:
                new_labels.extend([label] * size)
            i += 1

        new_clustered = [(key, len(list(group))) for key, group in groupby(new_labels)]
        clustered = new_clustered
        s += 1
        #avoid infinite loop
        if s == 300:
            break

    return new_labels

def merge_outlier_segments(sequence, min_length=10, max_length=100):
    if not sequence:
        return sequence
    
    # Step 1: Identify segments and their positions
    segments = []
    current_label = sequence[0]
    current_start = 0
    
    for i in range(1, len(sequence)):
        if sequence[i] != current_label:
            segments.append((current_label, current_start, i - current_start))
            current_label = sequence[i]
            current_start = i
    # Append the last segment
    segments.append((current_label, current_start, len(sequence) - current_start))
    
    # Step 2: Process segments and determine merges
    result = sequence.copy()  # Work on a copy of the sequence
    for i, (label, start, length) in enumerate(segments):
        if label == -1 and (length < min_length or length >= max_length):  # Outlier segment to merge
            if i == 0:  # At the beginning
                # Merge with the right cluster
                right_label = segments[i + 1][0]
                for j in range(start, start + length):
                    result[j] = right_label
            elif i == len(segments) - 1:  # At the end
                # Merge with the left cluster
                left_label = segments[i - 1][0]
                for j in range(start, start + length):
                    result[j] = left_label
            else:  # In the middle
                # Compare adjacent clusters and merge with the longer one
                left_length = segments[i - 1][2]
                right_length = segments[i + 1][2]
                merge_label = segments[i - 1][0] if left_length >= right_length else segments[i + 1][0]
                for j in range(start, start + length):
                    result[j] = merge_label
    
    return result

def nx_to_igraph(nx_graph):
    # Get all nodes and edges from nx_graph
    nodes = list(nx_graph.nodes())
    edges = list(nx_graph.edges())

    # Extract edge weights, defaulting to 1.0 if 'weight' attribute is missing
    weights = [nx_graph[u][v].get('weight', 1.0) for u, v in edges]

    # Create igraph graph with the number of nodes and add edges
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False, edge_attrs={'weight': weights})

    # Optionally, set vertex attributes (e.g., node names) to match nx_graph
    ig_graph.vs['name'] = nodes

    return ig_graph

def inter_community_edges2(G,segments):
    nodes = []

    for edge in G.edges:
        if ((edge[0] in segments[0])^(edge[1] in segments[0])):
            if edge[0] not in nodes:
                nodes += [edge[0]]
            if edge[1] not in nodes:
                nodes += [edge[1]]
    
    return len(nodes)

def segment_func(A, B):
    result = []
    for sublist in A:
        temp = []
        for num in sublist:
            temp.append(num)
            if num in B and num != sublist[0] and num != sublist[-1]:
                result.append(temp[:-1])
                temp = [num]
        if temp:
            result.append(temp)
    return result

def process_ref_txt(ref, res):
    """
    Processes a reference text file to extract residue labels and their corresponding indices.
    
    Parameters:
        ref (str): positions of the domains
        res (list): List of residue indices.

    Returns:
        list: A list of residue labels corresponding to the indices in 'res'.
    """
    ref_lst = ref.replace(' ','').split(';')
    ref_dict = {}

    result = []
    for label, domain_pos in enumerate(ref_lst):
        segment_pos_lst = domain_pos.split('+')
        ref_dict[label] = []

        for segment_pos in segment_pos_lst: 
            if '-' in segment_pos:
                start, end = map(int, segment_pos.split('-'))
                ref_dict[label] += list(range(start, end + 1))
            else:
                ref_dict[label] += [int(segment_pos)]
    
    for residue in res:
        s = 0
        for label in ref_dict.keys():
            if residue in ref_dict[label]:
                result.append(label)
                s = 1
                break
        
        if s == 0:
            result.append(-1)
    
    return result

# Function to compute matrix for Domain Overlap (NDO)
def merge_ele_matrix(mtx, list_range_true, list_range_pred):
    merged_mtx_row = []
    for row in mtx:
        new_row = row.copy()  # Create a copy of the row to avoid modifying the original row
        for label_pos in range(len(list_range_true)):
            s = label_pos
            e = s + len(list_range_true[label_pos])
            new_row = new_row[:s] + [sum(new_row[s:e])] + new_row[e:]
        merged_mtx_row.append(new_row)
    
    merged_mtx_row = np.array(merged_mtx_row).T.tolist()

# Function to compute matrix for Domain Overlap (NDO)
def domain_overlap_matrix(lists_label, list_residue = None): #Order in lists_label: ground_truth, prediction 
    if list_residue == None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}

    group_residue = {'pred': {}, 'true': {}}

    for key in group_label.keys():
        for label in set(group_label[key]):
            #group_residue[key][label] = list_to_range([lists_residue[key][i] for i in range(len(lists_residue[key])) if lists_label[key][i] == label])
            group_residue[key][label] = [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label]

    domain_matrix = []
    for label in sorted(set(group_label['pred'])): 
        row = []
        for label2 in sorted(set(group_label['true'])):
            row += [len([intersect for intersect in group_residue['pred'][label] if intersect in group_residue['true'][label2]])]

        domain_matrix += [row]


    min_labels = [min(set(group_label['pred'])), min(set(group_label['true']))]
    #return domain_matrix
    return domain_matrix, min_labels

# Function to calculate Normalized Domain Overlap (NDO)
def NDO(domain_matrix, len_rnas, min_labels=[0, 0]):
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
    score = Y / (2 * (len_rnas - sum_col[0] * (min_labels[1] == -1)))
    return score

# Function to compute domain distance
def domain_distance(segment1, segment2):
    return (abs(min(segment1) - min(segment2)) + abs(max(segment1) - max(segment2))) / 2

# Matrix for calculating Chain Segment Distance (CSD)
def domain_distance_matrix2(lists_label, list_residue=None): #Order in lists_label: ground_truth, prediction
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

def remove_row_col(matrix, n, m):
    # Remove rows with indices in list n
    matrix = [row for i, row in enumerate(matrix) if i not in n]
    
    # Remove columns with indices in list m from each remaining row
    matrix = [[elem for j, elem in enumerate(row) if j not in m] for row in matrix]
    
    return matrix

# Function to compute matrix for Domain Boundary Distance (DBD)
def domain_distance_matrix(lists_label, list_residue=None):
    """
    Build a distance matrix between predicted and true boundary entities.
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

# Function to compute Chain Segment Distance (CSD)
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

# Function to compute Domain Boundary Distance (DBD)
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

# Function to compute Structural Domain Count (SDC)
def SDC(truth, pred, outliner=False):
    if not outliner:
        truth, pred = [i for i in truth if i != -1], [i for i in pred if i != -1]
    return (max(len(set(truth)), len(set(pred))) - abs(len(set(truth)) - len(set(pred)))) / max(len(set(truth)), len(set(pred)))

# Function to compute Intersection over Union (IoU)
def IoU(domain_matrix, lists_label): #Order in lists_label: ground_truth, prediction
    min_labels = [min(set(lists_label[1])), min(set(lists_label[0]))]
    domain_matrix_no_linker = np.array([row[(min_labels[1] == -1):] for row in domain_matrix[(min_labels[0] == -1):]])
    
    if domain_matrix_no_linker.size == 0:
        return 0
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}
    
    used_row_index = []; max_val_lst = []; union_true_length_lst = []
    for col_index in range(len(domain_matrix_no_linker[0])):
        max_val = 0; max_row_index = None

        for row_index in range(len(domain_matrix_no_linker)):
            if row_index in used_row_index:
                continue
            
            if domain_matrix_no_linker[row_index][col_index] > max_val:                
                max_val = domain_matrix_no_linker[row_index][col_index]
                max_row_index = row_index

        max_val_lst.append(max_val)

        label_pos_true = [i for i in range(len(lists_label[0])) 
            if lists_label[0][i] == sorted(set(group_label['true']) - {-1})[col_index]]
        
        if max_row_index == None:
            union_true_length_lst += [(1,len(label_pos_true))]
            continue
        
        label_pos_pred = [i for i in range(len(lists_label[1])) 
                          if lists_label[1][i] == sorted(set(group_label['pred']) - {-1})[max_row_index]]
        
        union_true_length_lst += [(len(set(label_pos_pred + label_pos_true)), len(label_pos_true))]
        used_row_index.append(max_row_index)

    iou_list = [max_val_lst[i]/union_true_length_lst[i][0] for i in range(len(max_val_lst))]

    iou_score = sum([iou_list[i]*union_true_length_lst[i][1] for i in range(len(iou_list))]) / sum(union_true_length_lst[i][1] for i in range(len(union_true_length_lst)))

    return iou_score

# Function to compute CDO (Correct Domain Overlap) score
def CDO(domain_matrix, lists_label, threshold = 0.85): #Order in lists_label: ground_truth, prediction
    min_labels = [min(set(lists_label[1])), min(set(lists_label[0]))]
    domain_matrix_no_linker = np.array([row[(min_labels[1] == -1):] for row in domain_matrix[(min_labels[0] == -1):]])
    
    if domain_matrix_no_linker.size == 0:
        return 0
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}
    
    max_val_lst = []; true_length_lst = []
    for col_index in range(len(domain_matrix_no_linker[0])):
        max_val = 0; max_row_index = None

        for row_index in range(len(domain_matrix_no_linker)):
            if domain_matrix_no_linker[row_index][col_index] > max_val:                
                max_val = domain_matrix_no_linker[row_index][col_index]
                max_row_index = row_index

        max_val_lst.append(max_val)
          
        label_pos_true = [i for i in range(len(lists_label[0])) 
                    if lists_label[0][i] == sorted(set(group_label['true']) - {-1})[col_index]]
        
        true_length_lst += [len(label_pos_true)]
    
    score_list = [max_val_lst[i]/true_length_lst[i] for i in range(len(max_val_lst))]
    
    cdo_score = 1 if all(i > threshold for i in score_list) else 0

    return cdo_score

# Function to compute the Variation of Information (VI) score
def norm_VI(truth, pred):
    """Calculate variance of information between two clusterings."""

    # Length of sequence
    N = max(len(truth), len(pred))
    occurence_truth = np.array([list(truth).count(i) for i in sorted(set(truth))])
    occurence_pred = np.array([list(pred).count(i) for i in sorted(set(pred))])

    # Calculate entropies
    h1 = entropy(occurence_truth)
    h2 = entropy(occurence_pred)
    
    # Calculate mutual information
    mi = mutual_info_score(truth, pred)
    
    # Calculate variance of information
    vi = h1 + h2 - 2 * mi

    vi_norm = vi/ np.log(N) if N > 1 else 0  # Avoid division by zero
    
    return vi_norm

def clusters_to_labels(clusters, n, outliers=None):
    """clusters: list[tuple[int]] → labels (len=n), giữ nguyên index residue"""
    labels = [-1] * n
    out = set(outliers or [])
    for k, idxs in enumerate(clusters):
        for j in idxs:
            if j not in out:
                labels[j] = k
    return labels

def labels_to_clusters(labels):
    """labels (len=n) → list[tuple[int]]"""
    buckets = defaultdict(list)
    for j, k in enumerate(labels):
        if k != -1:
            buckets[k].append(j)
    return [tuple(buckets[k]) for k in sorted(buckets)]

def auto_k_by_silhouette(
    data,
    make_model_fn,
    kmin=2,
    kmax=10,
    metric="euclidean",
    return_k1_on_fail=True,
):
    """
    Try k in [kmin..kmax] (adjusted to >=2, <= N-1) and pick the k
    with the best Silhouette. If all trials fail or are degenerate,
    optionally return (1, -1.0, True) to indicate a single-domain fallback.

    Returns
    -------
    best_k : int
    best_score : float
    single_domain : bool   # True => treat as 1 cluster
    """
    N = len(data)
    lo = max(2, int(kmin))
    hi = min(int(kmax), max(2, N - 1))
    best_k, best_score = None, -1.0

    for k in range(lo, hi + 1):
        try:
            model = make_model_fn(k)
            labels = model.fit_predict(data)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(data, labels, metric=metric)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue

    if best_k is None and return_k1_on_fail:
        # Explicit single-domain detection
        return 1, -1.0, True

    return (best_k if best_k is not None else lo), best_score, False

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
    Detect whether each chain in a structure is protein or RNA/DNA.

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
    elif fname.endswith(".cif"):
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
    
    # ========================================
    # DEBUG: Print chain info
    # ========================================
    
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
        elif structure_file.lower().endswith(".cif"):
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
    STRIDE binary should be in the same directory as Functions.py (src/)
    """
    
    # Get the directory where Functions.py is located
    functions_dir = os.path.dirname(os.path.abspath(__file__))
    stride_path = os.path.join(functions_dir, 'stride')
    
    # Alternative: if stride is in src/bin/
    # stride_path = os.path.join(functions_dir, 'bin', 'stride')
    
    # Check if stride exists
    if not os.path.exists(stride_path):
        raise FileNotFoundError(
            f"STRIDE not found at {stride_path}\n"
            f"Please place stride executable in: {functions_dir}/"
        )
    
    # Make executable if not already
    if not os.access(stride_path, os.X_OK):
        try:
            os.chmod(stride_path, 0o755)
            print(f"Made {stride_path} executable")
        except Exception as e:
            raise PermissionError(f"Cannot make stride executable: {e}")
    
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
                elif ss_type in ['E', 'B']:      # Strands
                    ss_simple = 'E'
                else:                            # Coils/Turns
                    ss_simple = 'C'
                
                if chain not in ss_dict:
                    ss_dict[chain] = []
                ss_dict[chain].append((res_num_str, ss_simple))
    
    # Cleanup
    os.remove(tmp_path)
    
    return ss_dict

def curate_by_secondary_structure(pred_labels, res_labels, ss_annotations, min_ss_length=3):
    """
    Refine clustering based on secondary structure elements.
    
    Args:
        pred_labels: cluster labels from algorithm
        res_labels: original residue labels ["1", "2", "11A", ...]
        ss_annotations: list of (res_label, ss_type) from STRIDE
        min_ss_length: minimum length of SS element to preserve
    
    Returns:
        refined_labels: curated cluster labels
    """
    # Build mapping res_label -> ss_type
    ss_map = {res: ss for res, ss in ss_annotations}
    
    # Get SS type for each position
    ss_types = []
    for res_label in res_labels:
        ss_types.append(ss_map.get(res_label, 'C'))  # default to coil
    
    # Identify SS elements (continuous H or E regions)
    ss_elements = []
    current_ss = None
    start_idx = 0
    
    for i, ss in enumerate(ss_types):
        if ss != current_ss:
            if current_ss in ['H', 'E'] and i - start_idx >= min_ss_length:
                ss_elements.append({
                    'type': current_ss,
                    'start': start_idx,
                    'end': i - 1,
                    'indices': list(range(start_idx, i))
                })
            current_ss = ss
            start_idx = i
    
    # Check last element
    if current_ss in ['H', 'E'] and len(ss_types) - start_idx >= min_ss_length:
        ss_elements.append({
            'type': current_ss,
            'start': start_idx,
            'end': len(ss_types) - 1,
            'indices': list(range(start_idx, len(ss_types)))
        })
    
    # Rule: SS elements should not be split across domains
    refined_labels = pred_labels.copy()
    
    for elem in ss_elements:
        indices = elem['indices']
        elem_labels = [pred_labels[i] for i in indices]
        
        # If SS element spans multiple clusters, merge to majority cluster
        if len(set(elem_labels)) > 1:
            # Find majority cluster (excluding -1)
            label_counts = {}
            for lbl in elem_labels:
                if lbl != -1:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            
            if label_counts:
                majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
                # Assign all positions in this SS element to majority cluster
                for idx in indices:
                    refined_labels[idx] = majority_label
    
    return refined_labels

def build_ordinal_map(res_labels):
    """
    res_labels: list[str] e.g. ["10", "10A", "10B", "10C", "11", "12"]
    Return:
      pos_ids: list[int] same length, 1..N (or 0..N-1)
      label2pos: dict label -> pos
    """
    label2pos = {lab: i+1 for i, lab in enumerate(res_labels)}  # 1-based
    pos_ids = [label2pos[lab] for lab in res_labels]
    return pos_ids, label2pos


def list_to_range_by_pos(pos_list):
    """
    pos_list: list[int] positions in 1..N
    Return: list of ranges as (start_pos, end_pos) inclusive
    """
    if not pos_list:
        return []
    xs = sorted(set(pos_list))
    out = []
    s = xs[0]
    for i in range(1, len(xs)):
        if xs[i] != xs[i-1] + 1:
            out.append((s, xs[i-1]))
            s = xs[i]
    out.append((s, xs[-1]))
    return out

def ranges_pos_to_labels(ranges, res_labels):
    """
    ranges: [(s,e)] in 1..N
    res_labels: list[str] length N
    """
    out = []
    for s, e in ranges:
        if s == e:
            out.append(res_labels[s-1])
        else:
            out.append(f"{res_labels[s-1]}-{res_labels[e-1]}")
    return out

_BASE_RE = re.compile(r"^(-?\d+)")  

def shift_residue_labels(res_labels):
    """
    Map a sequence of residue labels (strings or ints) into shifted ints
    that preserve sequential distance including insertion codes.

    Example:
      ["10","10A","10B","11","12"] -> [10,11,12,13,14]

    Rule:
      - Start at integer(base of first label)
      - Walk through res_labels in order; each next label increments by 1
      - So distance == difference in positions
    """
    if not res_labels:
        return []

    def base_int(x):
        if isinstance(x, int):
            return x
        s = str(x).strip()
        m = _BASE_RE.match(s)
        return int(m.group(1)) if m else None

    first_base = base_int(res_labels[0])
    if first_base is None:
        # fallback: just 1..N
        return list(range(1, len(res_labels) + 1))

    return [first_base + i for i in range(len(res_labels))]


# =========================================================
#   STRIDE-constrained TOP-DOWN (UniDoc-like)
#   Requires these to exist in the same module:
#     - _FastQ
#     - _best_discont_local_q_on_graph
# =========================================================

_SSItem = Union[str, Tuple[Union[int, str], str]]  # 'H' or (res_label, 'H')


def _normalize_ss_info(ss_info: Sequence[_SSItem]) -> List[str]:
    """
    Accepts:
      - ['H','H','C',...]
      - [(res_label,'H'), (res_label,'C'), ...]  (STRIDE-like)
    Returns:
      ss_by_i: ['H','H','C',...]
    """
    if not ss_info:
        return []
    first = ss_info[0]
    if isinstance(first, tuple) and len(first) >= 2:
        return [str(x[1]) for x in ss_info]
    return [str(x) for x in ss_info]


def _build_stride_cut_points(
    ss_info: Sequence[_SSItem],
    *,
    res_labels: Optional[Sequence[Union[int, str]]] = None,
    allow_discont_gap: bool = True,
    boundary_only: bool = False,
    coil_set: Iterable[str] = ("C", "T", "S"),
    coil_step: int = 6,
) -> set:
    """
    Build allowed cut positions k in [1..N-1] where split is between k-1 | k.

    - Always allow SS-run boundaries (ss[i] != ss[i-1]).
    - Optionally allow additional cuts inside long coil/turn/loop runs (thinned by coil_step).
    - Optionally allow cuts at residue-number discontinuities (missing residues), if res_labels are numeric.

    Notes:
      - This is exactly the “solution space constraint” idea: STRIDE shapes candidate splits.
      - coil_step throttles candidates to keep top-down fast.
    """
    ss_by_i = _normalize_ss_info(ss_info)
    N = len(ss_by_i)
    cuts = set()
    if N <= 1:
        return cuts

    # 1) SS boundary cuts
    for i in range(1, N):
        if ss_by_i[i] != ss_by_i[i - 1]:
            cuts.add(i)

    # 2) Optional: allow cuts *within* long coils (thinned)
    if not boundary_only and coil_step is not None and coil_step > 0:
        last = None
        coil_set = set(coil_set)
        for i in range(1, N):
            if ss_by_i[i] in coil_set:
                if last is None or (i - last) >= coil_step:
                    cuts.add(i)
                    last = i
            else:
                last = None

    # 3) Optional: residue discontinuity cuts (missing residues)
    if allow_discont_gap and res_labels is not None and len(res_labels) == N:

        def _to_int(x):
            try:
                return int(x)
            except Exception:
                return None

        prev = _to_int(res_labels[0])
        for i in range(1, N):
            cur = _to_int(res_labels[i])
            if prev is not None and cur is not None and cur != prev + 1:
                cuts.add(i)
            prev = cur

    # Safety: keep inside valid range
    cuts = {k for k in cuts if 1 <= k <= N - 1}
    return cuts


def _best_cont_local_q_on_graph_candidates(
    H, frag: List[int], min_len: int, allowed_ks: set,
    *, resolution: float = 1.0, weight: str = "weight"
):
    """
    Best *continuous* bipartition for `frag`, but only evaluate k positions in `allowed_ks`.
    Uses the same fast modularity core as UniDoc (_FastQ).
    Returns (A, B, best_q) where A,B are lists of node ids.
    """
    L = len(frag)
    if L < 2 * min_len:
        return (None, None, -np.inf)

    # If subgraph has no edges, modularity is not informative
    if H.number_of_edges() == 0:
        return (None, None, -np.inf)

    # Convert global cut positions k (node index boundary) to local k (0..L-2),
    # where split is frag[:k+1] | frag[k+1:].
    lo = frag[0]
    hi = frag[-1]

    local_candidates = []
    for k_global in allowed_ks:
        # k_global is the first index of the RIGHT part (split between k_global-1 | k_global)
        if not (lo < k_global <= hi):
            continue
        # local index k is the last index of LEFT part in frag
        # Since frag is a contiguous list of node ids (0..N-1), local_k = (k_global - lo) - 1
        local_k = (k_global - lo) - 1
        if 0 <= local_k <= L - 2:
            left_len = local_k + 1
            right_len = L - left_len
            if left_len >= min_len and right_len >= min_len:
                local_candidates.append(local_k)

    if not local_candidates:
        return (None, None, -np.inf)

    fq = _FastQ(H, frag, resolution=resolution, weight=weight)
    best_k = None
    best_q = -np.inf

    for k in local_candidates:
        q = fq.Q_two_blocks((0, k), (-1, -2), (k + 1, L - 1))
        if q > best_q:
            best_q = q
            best_k = k

    if best_k is None:
        return (None, None, -np.inf)

    A = frag[:best_k + 1]
    B = frag[best_k + 1:]
    return (A, B, float(best_q))


def _scan_best_local_q_on_graph_stride(
    H, frag: List[int],
    *, min_len: int, resolution: float, weight: str,
    discont: bool, min_gap: int, require_bridge_edge: bool, bridge_path_len,
    allowed_ks: set
):
    """
    Like _scan_best_local_q_on_graph, but continuous split is restricted by allowed_ks.
    Discontinuous split (bridge) stays identical to UniDoc logic.
    """
    A1, B1, q1 = _best_cont_local_q_on_graph_candidates(
        H, frag, min_len, allowed_ks, resolution=resolution, weight=weight
    )

    A2 = B2 = None
    q2 = -np.inf
    if discont:
        # Reuse UniDoc's bridge-based discontinuous split search unchanged
        A2, B2, q2 = _best_discont_local_q_on_graph(
            H, frag, min_len, min_gap, resolution=resolution, weight=weight,
            require_bridge_edge=require_bridge_edge, bridge_path_len=bridge_path_len
        )

    if q1 >= q2:
        return A1, B1, q1
    return A2, B2, q2


def top_down_stride(
    G, N: int,
    *,
    ss_info: Sequence[_SSItem],
    res_labels: Optional[Sequence[Union[int, str]]] = None,
    # same knobs as UniDoc top-down
    split_thresh: float = 0.2,
    min_len: int = 30,
    resolution: float = 1.0,
    discont: bool = True,
    min_gap: int = 30,
    require_bridge_edge: bool = True,
    bridge_path_len=None,
    weight: str = "weight",
    # stride-solution-space knobs
    boundary_only: bool = False,
    coil_set: Iterable[str] = ("C", "T", "S"),
    coil_step: int = 6,
    allow_discont_gap: bool = True,
):
    """
    UniDoc-like top-down but constrained by STRIDE SSE.
    Output format matches top_down: [tuple(node_ids), ...].

    ss_info must be aligned to node order 0..N-1 (same order as graph nodes for that chain).
    """
    # Build allowed global cut positions k in [1..N-1]
    allowed_ks = _build_stride_cut_points(
        ss_info,
        res_labels=res_labels,
        allow_discont_gap=allow_discont_gap,
        boundary_only=boundary_only,
        coil_set=coil_set,
        coil_step=coil_step,
    )
    if not allowed_ks:
        allowed_ks = set(range(1, N))  # fallback: no constraint

    pieces = [list(range(N))]
    changed = True

    while changed:
        changed = False
        new_pieces = []

        for frag in pieces:
            H0 = G.subgraph(frag).copy()

            A, B, q_obs = _scan_best_local_q_on_graph_stride(
                H0, frag,
                min_len=min_len, resolution=resolution, weight=weight,
                discont=discont, min_gap=min_gap,
                require_bridge_edge=require_bridge_edge, bridge_path_len=bridge_path_len,
                allowed_ks=allowed_ks
            )

            if A is None or B is None or q_obs < split_thresh:
                new_pieces.append(frag)
            else:
                new_pieces.extend([A, B])
                changed = True

        pieces = new_pieces

    return [tuple(p) for p in pieces]


def mark_outliers_by_silhouette_after_clustering(
    data,
    labels,
    *,
    ss_info=None,
    coil_set=("C", "T", "S"),
    tau=0.1,
    max_expand=30,
    min_cluster_size=5,
    cleanup_min_run=3,
    metric="euclidean",
):
    """
    Post-process labels (output of cluster_algo) to mark outlier residues as -1
    using silhouette_samples, focusing near domain boundaries and chain ends.

    NEW: if ss_info is provided, expansion will NOT invade SSE positions.
         Positions whose ss_code NOT in coil_set are "protected" (cannot become -1).

    Args:
        data: (N,3)
        labels: (N,) int
        ss_info: None OR
          - ['H','E','C', ...] length N
          - [(res_label,'H'), ...] length N (STRIDE-like from get_secondary_structure_stride_auto)
        coil_set: which SSE codes are considered "coil-like" (allowed for outlier marking).
        tau: silhouette threshold; s(i) < tau => candidate outlier.
        max_expand: max residues to expand from each boundary side.
        min_cluster_size: skip silhouette decisions if cluster too small.
        cleanup_min_run: fill short -1 runs back if surrounded by same label.
        metric: distance metric for pairwise_distances.

    Returns:
        new_labels: (N,) int, may contain -1
        sil: (N,) float, NaN where undefined
        protected: (N,) bool, True if SSE-protected (cannot be set to -1)
    """
    X = np.asarray(data, dtype=float)
    y = np.asarray(labels, dtype=int).copy()
    N = len(y)
    if N == 0:
        return y, np.array([]), np.array([], dtype=bool)

    # ---- normalize ss_info -> list of single-char codes length N
    def _normalize_ss(ss):
        if ss is None:
            return None
        if len(ss) == 0:
            return ["C"] * N
        first = ss[0]
        if isinstance(first, tuple) and len(first) >= 2:
            ss2 = [str(t[1]) for t in ss]
        else:
            ss2 = [str(t) for t in ss]
        if len(ss2) < N:
            ss2 = ss2 + ["C"] * (N - len(ss2))
        elif len(ss2) > N:
            ss2 = ss2[:N]
        return ss2

    ss = _normalize_ss(ss_info)
    if ss is None:
        protected = np.zeros(N, dtype=bool)
    else:
        coil_set = set(coil_set)
        protected = np.array([(c not in coil_set) for c in ss], dtype=bool)

    # ---- silhouette requires >=2 non-outlier clusters
    uniq = sorted([k for k in set(y.tolist()) if k != -1])
    if len(uniq) < 2:
        sil = np.full(N, np.nan, dtype=float)
        return y, sil, protected

    # ---- ignore tiny clusters (silhouette noisy)
    counts = {k: int(np.sum(y == k)) for k in uniq}
    valid_clusters = {k for k, c in counts.items() if c >= min_cluster_size}
    if len(valid_clusters) < 2:
        sil = np.full(N, np.nan, dtype=float)
        return y, sil, protected

    valid_mask = np.isin(y, list(valid_clusters))
    idx_valid = np.where(valid_mask)[0]
    sub_y = y[idx_valid]

    if len(set(sub_y.tolist())) < 2:
        sil = np.full(N, np.nan, dtype=float)
        return y, sil, protected

    # ---- compute silhouette samples on valid subset
    D = pairwise_distances(X[idx_valid], metric=metric)
    sub_sil = silhouette_samples(D, sub_y, metric="precomputed")

    sil = np.full(N, np.nan, dtype=float)
    sil[idx_valid] = sub_sil

    def _bad(i: int) -> bool:
        if y[i] == -1:
            return False
        if protected[i]:
            return False  # never mark SSE positions as outliers
        si = sil[i]
        if not np.isfinite(si):
            return False
        return si < tau

    # ---- boundaries between different non-outlier labels
    boundaries = []
    for i in range(N - 1):
        a, b = y[i], y[i + 1]
        if a != b and a != -1 and b != -1:
            boundaries.append(i)

    # ---- expand from left end (stop if hit protected or not-bad)
    if y[0] != -1 and not protected[0]:
        dom = y[0]
        steps, i = 0, 0
        while i < N and y[i] == dom and steps < max_expand:
            if protected[i] or (not _bad(i)):
                break
            y[i] = -1
            i += 1
            steps += 1

    # ---- expand from right end
    if y[N - 1] != -1 and not protected[N - 1]:
        dom = y[N - 1]
        steps, i = 0, N - 1
        while i >= 0 and y[i] == dom and steps < max_expand:
            if protected[i] or (not _bad(i)):
                break
            y[i] = -1
            i -= 1
            steps += 1

    # ---- expand from each boundary both sides
    for b in boundaries:
        # left side
        left_dom = y[b]
        i, steps = b, 0
        while i >= 0 and y[i] == left_dom and steps < max_expand:
            if protected[i] or (not _bad(i)):
                break
            y[i] = -1
            i -= 1
            steps += 1

        # right side
        right_dom = y[b + 1]
        i, steps = b + 1, 0
        while i < N and y[i] == right_dom and steps < max_expand:
            if protected[i] or (not _bad(i)):
                break
            y[i] = -1
            i += 1
            steps += 1

    # ---- cleanup: fill very short -1 runs if both sides are same label
    if cleanup_min_run and cleanup_min_run > 0:
        i = 0
        while i < N:
            if y[i] != -1:
                i += 1
                continue
            j = i
            while j < N and y[j] == -1:
                j += 1
            run_len = j - i

            left_label = y[i - 1] if i - 1 >= 0 else None
            right_label = y[j] if j < N else None

            if (
                run_len < cleanup_min_run
                and left_label is not None
                and right_label is not None
                and left_label == right_label
                and left_label != -1
            ):
                y[i:j] = left_label
            i = j

    return y, sil, protected

def get_secondary_structure_rna_auto(structure_file, chain_types=None, dssr_path=None, fallback='C'):
    """
    RNA pairing-aware "secondary structure" wrapper (DSSR-based).

    - Runs ONLY for RNA chains (based on detect_chain_types).
    - Supports .pdb and .cif (NO cif->pdb conversion; DSSR reads mmCIF directly).
    - Returns dict {chain_id: [(res_label, ss_simple), ...]} where:
        'H' = paired (stem-like; proxy "structured")
        'C' = unpaired (loop/bulge/junction; coil-like)
    - If DSSR missing / fails: returns all fallback (default 'C') per residue (safe).
    - DSSR byproducts are ALWAYS deleted (runs in TemporaryDirectory with cwd=td).
    """

    # --- chain typing ---
    if chain_types is None:
        chain_types = detect_chain_types(structure_file)

    rna_chains = [ch for ch, t in chain_types.items() if t == "rna"]
    if not rna_chains:
        return {}

    # --- input sanity ---
    lower = structure_file.lower()
    if not (lower.endswith(".pdb") or lower.endswith(".cif")):
        raise ValueError("get_secondary_structure_rna_auto expects a .pdb or .cif file")

    dssr_input = structure_file  # run DSSR directly on PDB/mmCIF (avoid PDB chain-id limits)

    # --- helpers ---
    def _is_nt(resname: str) -> bool:
        r = (resname or "").strip().upper()
        return (
            r in {"A", "C", "G", "U", "I", "T"} or
            r in {"DA", "DC", "DG", "DT", "DU"} or
            r.startswith("D")
        )

    def _load_res_labels(pdb_or_cif_path):
        """Return ordered residue labels per RNA chain: {chain_id: ['1','2','11A',...]}"""
        if pdb_or_cif_path.lower().endswith(".pdb"):
            parser = PDBParser(QUIET=True)
        else:
            parser = MMCIFParser(QUIET=True)

        sid = os.path.basename(pdb_or_cif_path).split(".")[0]
        st = parser.get_structure(sid, pdb_or_cif_path)
        model = next(st.get_models())

        out = {}
        wanted = set(rna_chains)
        for chain in model:
            if chain.id not in wanted:
                continue
            labels = []
            for res in chain.get_residues():
                hetflag, resseq, icode = res.id
                if hetflag == "W":
                    continue
                if is_aa(res, standard=True):
                    continue
                if not _is_nt(res.get_resname()):
                    continue
                ic = (icode.strip() if isinstance(icode, str) else "")
                labels.append(f"{int(resseq)}{ic}")
            if labels:
                out[chain.id] = labels
        return out

    def _resolve_dssr_path(p):
        if p and os.path.exists(p):
            return p
        for cand in ("x3dna-dssr", "dssr"):
            found = shutil.which(cand)
            if found:
                return found
        return None

    chain_to_labels = _load_res_labels(dssr_input)

    # if DSSR missing -> fallback
    dssr_bin = _resolve_dssr_path(dssr_path)
    if dssr_bin is None:
        print('fallback missing')
        return {ch: [(lab, fallback) for lab in labs] for ch, labs in chain_to_labels.items()}

    # --- run DSSR in temp dir, ALWAYS clean byproducts ---
    data = None
    base_dir = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=".dssr_tmp_", dir=base_dir) as td:
        json_path = os.path.join(td, "dssr.json")

        # Preferred: DSSR writes JSON file "dssr.json" in cwd
        cmd = [dssr_bin, f"-i={os.path.abspath(dssr_input)}", "--json", "dssr.json"]
        try:
            subprocess.run(cmd, cwd=td, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            # Fallback: some builds print JSON to stdout when "--json" has no filename
            cmd2 = [dssr_bin, f"-i={os.path.abspath(dssr_input)}", "--json"]
            r = subprocess.run(cmd2, cwd=td, capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        f.write(r.stdout)
                except Exception:
                    data = None
            else:
                data = None

        if data is None:
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)
                except Exception:
                    data = None

    # DSSR failed -> fallback
    if not isinstance(data, dict):
        print('fallback failed')
        return {ch: [(lab, fallback) for lab in labs] for ch, labs in chain_to_labels.items()}


    # --- parse paired residues from DSSR JSON (defensive across versions) ---
    paired = {ch: set() for ch in rna_chains}

    _re_num_ic = re.compile(r"(-?\d+)([A-Za-z]?)")

    def _extract_chain_and_label(x):
        """
        Best-effort parse DSSR nt identifier -> (chain_id, '12'/'12A').
        Supports chain IDs longer than 1 char (mmCIF).
        """
        s = str(x)

        # chain is often before '.' or ':' (and can be multi-char, e.g. 'C1')
        ch = None
        if "." in s:
            ch = s.split(".", 1)[0].strip()
        elif ":" in s:
            ch = s.split(":", 1)[0].strip()

        # residue label: take the last token that looks like digits+optional insertion
        m = None
        for tok in re.findall(r"-?\d+[A-Za-z]?", s):
            m = tok
        if m is None:
            return ch, None

        mm = _re_num_ic.match(m)
        if not mm:
            return ch, None
        label = f"{int(mm.group(1))}{mm.group(2)}"
        return ch, label

    # gather candidate pair lists from possible keys
    pair_lists = []
    for key in ("pairs", "pair", "basePairs", "base_pairs", "bps", "bp_list"):
        v = data.get(key, None)
        if isinstance(v, list):
            pair_lists.append(v)

    stems = data.get("stems", None)
    if isinstance(stems, list):
        for st in stems:
            if isinstance(st, dict) and isinstance(st.get("pairs"), list):
                pair_lists.append(st["pairs"])

    def _iter_nt_pairs(lst):
        if not isinstance(lst, list):
            return
        for it in lst:
            if not isinstance(it, dict):
                continue

            # common direct keys
            for akey, bkey in (("nt1", "nt2"), ("res1", "res2"), ("i", "j"), ("ntA", "ntB")):
                if akey in it and bkey in it:
                    yield it[akey], it[bkey]
                    break

            # nested dict style: {"nt1": {"nt_id": ...}, "nt2": {...}}
            if "nt1" in it and "nt2" in it and isinstance(it["nt1"], dict) and isinstance(it["nt2"], dict):
                a = it["nt1"].get("nt_id") or it["nt1"].get("id") or it["nt1"].get("name")
                b = it["nt2"].get("nt_id") or it["nt2"].get("id") or it["nt2"].get("name")
                if a and b:
                    yield a, b

    for plist in pair_lists:
        for a, b in _iter_nt_pairs(plist):
            ch1, lab1 = _extract_chain_and_label(a)
            ch2, lab2 = _extract_chain_and_label(b)
            if ch1 in paired and lab1 is not None:
                paired[ch1].add(lab1)
            if ch2 in paired and lab2 is not None:
                paired[ch2].add(lab2)

    # --- build output aligned to residue order ---
    out = {}
    for ch in rna_chains:
        labels = chain_to_labels.get(ch, [])
        if not labels:
            continue
        pset = paired.get(ch, set())
        out[ch] = [(lab, ("H" if lab in pset else fallback)) for lab in labels]

    return out

def _graph_is_weighted(H, weight='weight'):
    wlist = []
    is_w = False
    for _, _, d in H.edges(data=True):
        w = float(d.get(weight, 1.0))
        wlist.append(w)
        if abs(w - 1.0) > 1e-12:
            is_w = True
    return is_w, wlist

def _assign_shuffled_weights(H, wlist, weight='weight'):
    if not wlist:
        return
    import random
    edges = list(H.edges())
    if not edges:
        return
    random.shuffle(wlist)
    for (e, w) in zip(edges, wlist):
        u, v = e
        H[u][v][weight] = float(w)

def _double_edge_swap_safe(H, nswap, max_tries):
    try:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    except nx.NetworkXError:
        pass
    return H

# ===============================
#   Fast bipartition modularity on H = G[frag]
# ===============================

class _FastQ:
    """Precompute dense W, prefix sums, and degree prefix for H = G[frag].
    Node order is exactly `frag` to make contiguous ranges match the sequence.
    """
    def __init__(self, H: nx.Graph, frag: List[int], resolution: float = 1.0, weight: str = 'weight'):
        self.gamma = float(resolution)
        self.frag = list(map(int, frag))
        self.pos = {v: i for i, v in enumerate(self.frag)}
        n = len(self.frag)
        self.n = n
        W = np.zeros((n, n), dtype=np.float64)
        for u, v, d in H.edges(data=True):
            i = self.pos.get(int(u)); j = self.pos.get(int(v))
            if i is None or j is None:  # should not happen on H
                continue
            w = float(d.get(weight, 1.0))
            if i == j:
                continue
            W[i, j] += w
            W[j, i] += w
        self.W = W
        # total weight m (each undirected edge counted once)
        self.m = W.sum() / 2.0
        # degree vector and its prefix (row sums)
        deg = W.sum(axis=1)
        self.deg = deg
        self.deg_prefix = np.concatenate([[0.0], np.cumsum(deg)])
        # 2D inclusive prefix of W for O(1) rectangle sums
        self.S = W.cumsum(axis=0).cumsum(axis=1)

    # inclusive rectangle sum for rows [r1..r2], cols [c1..c2]
    def rect(self, r1: int, r2: int, c1: int, c2: int) -> float:
        if r1 > r2 or c1 > c2:
            return 0.0
        S = self.S
        a = S[r2, c2]
        b = S[r1-1, c2] if r1 > 0 else 0.0
        c = S[r2, c1-1] if c1 > 0 else 0.0
        d = S[r1-1, c1-1] if (r1 > 0 and c1 > 0) else 0.0
        return float(a - b - c + d)

    # intra weight for contiguous block [a..b], edges counted once
    def L_block(self, a: int, b: int) -> float:
        if a > b:
            return 0.0
        box = self.rect(a, b, a, b)   # counts both directions
        return 0.5 * box               # undirected edges once

    # between weight for two disjoint contiguous blocks [a..b] (rows) × [c..d] (cols)
    # counted once by taking one direction only
    def L_between(self, a: int, b: int, c: int, d: int) -> float:
        if a > b or c > d:
            return 0.0
        return self.rect(a, b, c, d)

    def d_block(self, a: int, b: int) -> float:
        return float(self.deg_prefix[b+1] - self.deg_prefix[a])

    def Q_two_blocks(self, A1: Tuple[int, int], A2: Tuple[int, int], B: Tuple[int, int]) -> float:
        """Modularity for partition {A1 ∪ A2, B} where A1 and A2 are contiguous (A2 may be empty),
        and B contiguous. All inclusive index pairs in [0..n-1].
        """
        if self.m <= 0:
            return 0.0
        a1, a2 = A1; c1, c2 = A2; b1, b2 = B
        # L_A: within left + within right + between(left,right)
        L_A = self.L_block(a1, a2)
        if c1 <= c2:
            L_A += self.L_block(c1, c2)
            # add cross between (a1..a2) and (c1..c2) once
            L_A += self.L_between(a1, a2, c1, c2)
        # L_B: within middle
        L_B = self.L_block(b1, b2)
        # degrees
        dA = self.d_block(a1, a2)
        if c1 <= c2:
            dA += self.d_block(c1, c2)
        dB = self.d_block(b1, b2)
        m = self.m
        g = self.gamma
        Q = (L_A/m - g * (dA/(2*m))**2) + (L_B/m - g * (dB/(2*m))**2)
        return float(Q)

# ===============================
#   Contact ratio (unchanged)
# ===============================

def _contact_ratio(G, A, B, mode = 'min'):
    A = set(A); B = set(B)
    if not A or not B:
        return 0.0
    At = sum(1 for a in A if any(nb in B for nb in G.neighbors(a)))
    Bt = sum(1 for b in B if any(nb in A for nb in G.neighbors(b)))
    if mode == 'max':
        return max(At/len(A), Bt/len(B))
    elif mode == 'avg':
        return 0.5 * (At/len(A) + Bt/len(B))
    else:  # min
        return min(At/len(A), Bt/len(B))

# ===============================
#   Best splits (now using fast Q)
# ===============================

def _best_cont_local_q_on_graph(H, frag, min_len, resolution=1.0, weight='weight'):
    L = len(frag)
    if L < 2 * min_len:
        return (None, None, -np.inf)

    fq = _FastQ(H, frag, resolution=resolution, weight=weight)
    best_k = None
    best_q = -np.inf

    k_start = min_len - 1
    k_end   = L - min_len - 1

    for k in range(k_start, k_end + 1):
        # A = [0..k], B = [k+1..L-1]
        q = fq.Q_two_blocks((0, k), (-1, -2), (k+1, L-1))  # second block empty sentinel
        if q > best_q:
            best_q = q
            best_k = k

    if best_k is None:
        return (None, None, -np.inf)

    A = frag[:best_k+1]
    B = frag[best_k+1:]
    return (A, B, float(best_q))


def _best_discont_local_q_on_graph(
    H, frag, min_len, min_gap, resolution=1.0, weight='weight',
    require_bridge_edge=True, bridge_path_len=None
):
    L = len(frag)
    if L < 2 * min_len:
        return (None, None, -np.inf)

    pos = {node: i for i, node in enumerate(frag)}
    fq = _FastQ(H, frag, resolution=resolution, weight=weight)

    best_pair, best_q = None, -np.inf

    t_min = min_len - 1
    t_max = L - min_len - 1
    for t in range(t_min, t_max + 1):
        u = frag[t]
        for v in H.neighbors(u):
            s = pos.get(v, None)
            if s is None:
                continue
            s = s + 1
            if s <= t + max(min_gap, 1 + min_len) or s > L - min_len:
                continue
            if not require_bridge_edge and bridge_path_len is not None:
                try:
                    if nx.shortest_path_length(H, u, frag[s-1]) > bridge_path_len:
                        continue
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            # A = left ∪ right, B = middle
            left_a, left_b   = (0, t)
            mid_a,  mid_b    = (t+1, s-1)
            right_a, right_b = (s, L-1)
            if (mid_b - mid_a + 1) < min_len or ((left_b - left_a + 1) + (right_b - right_a + 1)) < min_len:
                continue
            q = fq.Q_two_blocks((left_a, left_b), (right_a, right_b), (mid_a, mid_b))
            if q > best_q:
                best_q = q
                best_pair = (frag[:t+1] + frag[s:], frag[t+1:s])

    return (best_pair[0], best_pair[1], float(best_q)) if best_pair else (None, None, -np.inf)


def _scan_best_local_q_on_graph(
    H, frag, *, min_len=30, resolution=1.0, weight='weight',
    discont=True, min_gap=30, require_bridge_edge=True, bridge_path_len=None
):
    A1, B1, q1 = _best_cont_local_q_on_graph(H, frag, min_len, resolution, weight)
    A2 = B2 = None; q2 = -np.inf
    if discont:
        A2, B2, q2 = _best_discont_local_q_on_graph(
            H, frag, min_len, min_gap, resolution, weight,
            require_bridge_edge=require_bridge_edge, bridge_path_len=bridge_path_len
        )
    if q1 >= q2:
        return A1, B1, q1
    else:
        return A2, B2, q2

# ===============================
#   Permutation p‑value (unchanged logic, just reuses fast scorer inside)
# ===============================

def _best_split_pvalue_on_graph(
    H0, frag, *, min_len=30, resolution=1.0, weight=True,
    discont=True, min_gap=30, require_bridge_edge=True, bridge_path_len=None,
    R=10, swap_mult=5, alpha=None
):
    A_obs, B_obs, q_obs = _scan_best_local_q_on_graph(
        H0, frag, min_len=min_len, resolution=resolution, weight=weight,
        discont=discont, min_gap=min_gap,
        require_bridge_edge=require_bridge_edge, bridge_path_len=bridge_path_len
    )
    if A_obs is None or B_obs is None:
        return (None, None, q_obs, 1.0)

    m = H0.number_of_edges()
    if m == 0:
        return (A_obs, B_obs, q_obs, 1.0)

    is_w, wlist0 = _graph_is_weighted(H0, weight=weight)
    ge = 0
    nswap    = max(1, int(swap_mult * m))
    maxtries = max(100, 10 * nswap)

    for r in range(R):
        Hr = H0.copy()
        _double_edge_swap_safe(Hr, nswap=nswap, max_tries=maxtries)
        if is_w:
            _assign_shuffled_weights(Hr, wlist0[:], weight=weight)
        _, _, q_r = _scan_best_local_q_on_graph(Hr, frag, min_len=min_len, resolution=resolution,
                                                weight=weight, discont=discont, min_gap=min_gap,
                                                require_bridge_edge=require_bridge_edge,
                                                bridge_path_len=bridge_path_len)
        if q_r >= q_obs - 1e-12:
            ge += 1
        lb = (ge + 1) / (R + 1)
        if lb > alpha:
            return (A_obs, B_obs, q_obs, lb)

    pval = (ge + 1) / (R + 1)
    return (A_obs, B_obs, q_obs, pval)

# ===============================
#   Public API (unchanged)
# ===============================

def top_down(
    G, N, *,
    split_thresh=0.2,
    min_len=30,
    resolution=1.0,
    discont=True,
    min_gap=30,
    require_bridge_edge=True,
    bridge_path_len=None,
    use_pval=False,
    alpha=0.1,
    R=10,
    swap_mult=5,
    weight=True
):
    pieces = [list(range(N))]
    changed = True

    while changed:
        changed = False
        new_pieces = []

        for frag in pieces:
            H0 = G.subgraph(frag).copy()
            A, B, q_obs = _scan_best_local_q_on_graph(
                H0, frag, min_len=min_len, resolution=resolution, weight=weight,
                discont=discont, min_gap=min_gap,
                require_bridge_edge=require_bridge_edge, bridge_path_len=bridge_path_len
            )
            if A is None or B is None or q_obs < split_thresh:
                new_pieces.append(frag)
                continue
            if use_pval:
                _, _, _, pval = _best_split_pvalue_on_graph(
                    H0, frag, min_len=min_len, resolution=resolution, weight=weight,
                    discont=discont, min_gap=min_gap,
                    require_bridge_edge=require_bridge_edge, bridge_path_len=bridge_path_len,
                    R=R, swap_mult=swap_mult, alpha=alpha
                )
                accept = (pval <= alpha)
            else:
                accept = True
            if accept:
                new_pieces.extend([A, B])
                changed = True
            else:
                new_pieces.append(frag)

        pieces = new_pieces

    return [tuple(p) for p in pieces]


def bottom_up(G, pieces, ratio_thresh=0.1, force_merge_num=None, mode = 'max'):
    def best_pair_by_contact_ratio(sets):
        best_pair, best_val = None, -1.0
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                val = _contact_ratio(G, sets[i], sets[j], mode= mode)
                if val > best_val:
                    best_val, best_pair = val, (i, j)
        return best_pair, best_val

    pieces = [set(p) for p in pieces]
    target = None if force_merge_num is None else max(1, int(force_merge_num))

    while True:
        if target is not None and len(pieces) <= target:
            break
        pair, val = best_pair_by_contact_ratio(pieces)
        if pair is None or val < ratio_thresh:
            break
        i, j = pair
        merged = pieces[i] | pieces[j]
        keep = [k for k in range(len(pieces)) if k not in (i, j)]
        pieces = [pieces[k] for k in keep] + [merged]

    if target is not None and len(pieces) > target:
        while len(pieces) > target:
            pair, val = best_pair_by_contact_ratio(pieces)
            if pair is None:
                break
            i, j = pair
            merged = pieces[i] | pieces[j]
            keep = [k for k in range(len(pieces)) if k not in (i, j)]
            pieces = [pieces[k] for k in keep] + [merged]

    return [tuple(sorted(p)) for p in pieces]

def bottom_up_size_first(G, pieces, ratio_thresh=0.1, force_merge_num=None, mode = 'min'):
    pieces = [set(p) for p in pieces]
    target = None if force_merge_num is None else max(1, int(force_merge_num))

    def contact_ratio(a, b, mode):
        return _contact_ratio(G, a, b, mode = mode)

    changed = True
    while changed:
        changed = False
        if target is not None and len(pieces) <= target:
            break

        # sort by size
        order = sorted(range(len(pieces)),
                       key=lambda i: (len(pieces[i]), min(pieces[i]) if pieces[i] else -1))

        used = set()
        for i in order:
            if i in used or i >= len(pieces):
                continue
            
            # Find the first candidate with contact_ratio >= ratio_thresh, prioritise small size
            candidates = []
            for j in range(len(pieces)):
                if j == i or j in used:
                    continue
                r = contact_ratio(pieces[i], pieces[j], mode)
                if r >= ratio_thresh:
                    candidates.append((len(pieces[j]), min(pieces[j]) if pieces[j] else -1, j))
            if not candidates:
                continue

            candidates.sort()
            j = candidates[0][2]

            # Merge then restart
            merged = pieces[i] | pieces[j]
            keep_idx = sorted(set(range(len(pieces))) - {i, j})
            pieces = [pieces[k] for k in keep_idx] + [merged]
            changed = True
            break  # restart sweep

    # ---- PASS 2: force merging small-first ----
    if target is not None and len(pieces) > target:
        while len(pieces) > target:
            order = sorted(range(len(pieces)),
                           key=lambda i: (len(pieces[i]), min(pieces[i]) if pieces[i] else -1))
            i = order[0]

            best_j, best_r = None, -1.0
            for j in range(len(pieces)):
                if j == i:
                    continue
                r = contact_ratio(pieces[i], pieces[j], mode)
                if r > best_r or (abs(r - best_r) < 1e-12 and (len(pieces[j]), min(pieces[j]) if pieces[j] else -1, j) <
                                  (len(pieces[best_j]), min(pieces[best_j]) if best_j is not None and pieces[best_j] else -1, best_j if best_j is not None else 10**9)):
                    best_r, best_j = r, j
            if best_j is None:
                break
            merged = pieces[i] | pieces[best_j]
            keep_idx = sorted(set(range(len(pieces))) - {i, best_j})
            pieces = [pieces[k] for k in keep_idx] + [merged]

    return [tuple(sorted(p)) for p in pieces]

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

    # ========================================
    # FIX: Only process chains_to_keep
    # ========================================
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