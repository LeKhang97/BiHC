import os
import sys
import json

import numpy as np
from pathlib import Path

from src.argument import *
from src import graph_io


EXTS = ('.pdb', '.cif', '.ent', '.mmcif')


def collect_inputs(path):
    """One structure file, or every structure file inside a directory (sorted, non-recursive)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in EXTS)
        return [str(f) for f in files]
    return [str(p)]


def with_hetero(in_path, chain_id, pred, subdata, res_num, res_labels):
    """Put the chain's HETATM polymer residues (MSE and other modified residues, left out of the graph) back into the
    domains for the output (src.Functions.fill_hetero), so no domain is cut by one- or two-residue gaps. Returns
    (pred, data, res_num, res_labels) over every polymer residue; unchanged when there is none or the structure file
    is not available."""
    chain = chain_id.split('_')[0] if 'MODEL' in chain_id else chain_id.replace('_', '')
    if not in_path or not os.path.exists(in_path):
        return pred, subdata, res_num, res_labels
    try:
        order = hetero_polymer_order(in_path, chain, res_labels)
    except Exception:
        order = None
    if not order:
        return pred, subdata, res_num, res_labels
    lab, labels, xyz = fill_hetero(pred, res_labels, order)
    return np.asarray(lab, int), np.asarray(xyz, float), list(range(1, len(labels) + 1)), labels


DOMAIN_ROWS = {}     # structure name -> one row per domain for the .tsv output (report_chain fills it, export writes it)
TSV_HEADER = ('structure', 'chain', 'domain', 'n_residues', 'res_positions', 'missing_res')


def report_chain(result, filename, chain_id, pred, subdata, res_num, current_res_labels, missing, y):
    """Print one chain's domains, add its entry to result and its rows to DOMAIN_ROWS."""

    msg = 'Output information:'
    decorate_message(msg)

    pred_ext, res_num_ext, res_labels_ext = extend_missing_res(
        pred, res_num, res_labels=current_res_labels
    )
    use_pred = pred_ext
    use_res_num = res_num_ext
    use_res_labels = res_labels_ext

    rows = DOMAIN_ROWS.setdefault(filename, [])
    for h, k in enumerate(set(j for j in use_pred if j != -1)):
        positions = [use_res_labels[j] for j in range(len(use_pred)) if use_pred[j] == k]
        range_pos = domain_ranges(positions, use_res_labels)
        mess_pos = format_range_string(range_pos)
        gaps = format_range_string(bridged_gaps(range_pos, use_res_labels))

        print(f'Number of residues of cluster {h + 1}: {len(positions)}')
        print(f'Cluster {h + 1} positions:\n{mess_pos}\n')
        if gaps:
            word = 'residues' if ('-' in gaps.lstrip('-') or ',' in gaps) else 'residue'
            print(f'  WARNING: cluster {h + 1} spans {word} {gaps} without coordinates in the structure, '
                  f'bridged in its ranges\n')
        rows.append((filename, chain_id.replace('_', ''), h + 1, len(positions), mess_pos.replace(' ', ''),
                     gaps.replace(' ', '') or '-'))

    print('\n')

    result[filename][chain_id.replace('_', '')] = {
        'data': subdata,
        'cluster': pred,
        'res': res_num,
        'res_labels': current_res_labels,
        'missing_res': missing,
    }


def export(result, filename, in_path, y, batch, graph_chains=(), min_length=0):
    """Write the JSON, .tsv and graph outputs (-o), and the structure file + PyMOL script per chain (--output-pdb)."""
    if y[0] is not None:
        target_dir = Path(y[0])
        if y[1]:
            decorate_message('Exporting output:')
            print(f"Writing to the path {target_dir}", end='\n\n')

        target_dir.mkdir(parents=True, exist_ok=True)

        # base name of every output file: -j (one input) or the structure name (batch)
        stem = os.path.basename(filename if batch or y[3] is None else y[3])
        for ext in ('.json', '.pdb', '.mmcif', '.cif', '.ent'):
            stem = stem.replace(ext, '')

        if y[3] is not None:
            outfile1 = target_dir / f"{stem}.json"

            with open(outfile1, 'w') as outfile:
                json.dump(result, outfile, indent=2, cls=CustomNumpyEncoder)
            # one row per domain; segments in author numbering joined by ',', residues without coordinates inside a
            # domain bridged as in every other output (missing_res lists them, '-' when there are none)
            outfile3 = outfile1.with_name(outfile1.stem + '_domains.tsv')
            with open(outfile3, 'w') as outfile:
                outfile.write('\t'.join(TSV_HEADER) + '\n')
                for row in DOMAIN_ROWS.get(filename, []):
                    outfile.write('\t'.join(str(v) for v in row) + '\n')
            if y[1]:
                print(f"Wrote {outfile1} and {outfile3}")

        if len(y) > 11 and y[11] and graph_chains:
            graph_file = target_dir / f"{stem}_graph.npz"
            keep = [c for c in graph_chains if c['graph'] is not None]
            for c in graph_chains:
                if c['graph'] is None:
                    print(f"  WARNING: chain {c['chain_id']} has no graph and is left out of {graph_file}")
            if keep:
                graph_io.save(graph_file, os.path.abspath(in_path), filename, min_length, keep)
                if y[1]:
                    print(f"Wrote {graph_file} (re-run with -i on it to change -t without STRIDE / Voronota)")

        if y[4] is not None:                                            # --output-pdb
            # one structure file per chain (NAME_<chain> when several chains were processed), in the input's format
            # and residue order, and next to it a PyMOL script that loads that file and colours the chain's domains
            chains = list(result[filename].keys())
            ext = '.cif' if in_path.lower().endswith(('.cif', '.mmcif')) else '.pdb'
            for chain in chains:
                base = stem if len(chains) == 1 else f"{stem}_{chain}"
                out_struct = (target_dir / f"{base}{ext}").resolve()
                labels = [str(x) for x in result[filename][chain]['res_labels']]
                write_chain_structure(in_path, chain, labels, str(out_struct))
                cmds = pymol_process(list(result[filename][chain]['cluster']), None, name=base, verbose=y[1],
                                     res_labels=labels, chain=chain, obj=base)
                with open(target_dir / f"{base}_pymolcmd.pml", 'w') as outfile:
                    outfile.write(f"load {out_struct}, {base}; " + '; '.join(cmds) + ';\n')
                if y[1]:
                    print(f"Wrote {out_struct} and {target_dir / f'{base}_pymolcmd.pml'}")

        if y[1]:
            print("Writing completed!")


def process_graph(graph_path, x, y, batch=False):
    """Saved-graph input (-i NAME_graph.npz): re-run the partition selection and clean-up from a saved graph file. Returns the result dict (or None)."""
    if y[1]:
        print(f"\nProcessing {graph_path}", end='\n\n')
    try:
        meta, chains = graph_io.load(graph_path)
    except (ValueError, OSError, KeyError) as e:
        print(f"  ERROR: {e}"); return None
    set_options(t=(y[9] if len(y) > 9 else None),
                   post=(y[10] if len(y) > 10 else True))
    filename, in_path = meta['filename'], meta['source']
    if x[-1] < meta['min_length']:
        print(f"  NOTE: the graph was saved with -m / --min-length = {meta['min_length']}; shorter chains are not in it")
    chains = [c for c in chains if len(c['res']) >= x[-1]]
    if not chains:
        print(f"  no chain long enough to process (minimum length -m / --min-length = {x[-1]} residues)"); return None
    if y[5] == 'longest':
        print('Processing longest valid chain!', end='\n\n')
        n = max(len(c['res']) for c in chains)
        chains = [c for c in chains if len(c['res']) == n]
    elif y[5]:
        print(f'Processing chain {y[5]}!', end='\n\n')
        chains = [c for c in chains if c['chain_id'] in (y[5] + '_', y[5])]
        if not chains:
            print(f"  chain {y[5]} not found in the graph file (or shorter than -m / --min-length = {x[-1]})"); return None
    else:
        print('Processing each chain independently!', end='\n\n')
    result = {filename: {}}
    DOMAIN_ROWS[filename] = []
    for c in chains:
        pred = cluster_from_graph(c['graph'], c['chain_id'])
        pe, de, re_, le = with_hetero(in_path, c['chain_id'], pred, c['data'], c['res'], c['res_labels'])
        report_chain(result, filename, c['chain_id'], pe, de, re_, le, c['missing_res'], y)
    export(result, filename, in_path, y, batch)
    return result


def collect_graphs(path):
    """One graph file, or every *_graph.npz inside a directory (sorted, non-recursive)."""
    p = Path(path)
    if p.is_dir():
        return [str(f) for f in sorted(p.iterdir()) if f.is_file() and f.name.endswith('_graph.npz')]
    return [str(p)]


def process_one(in_path, x, y, batch=False):
    """Run the parser on a single structure file. Returns its result dict (or None on a fatal problem)."""
    if y[1]:
        print(f"\nProcessing {in_path}", end='\n\n')

    with open(in_path, 'r') as infile:
        missing_res = find_missing_res(in_path)
        file = infile.read().split('\n')
        C = process_structure(file, atom_type=y[2], filename=in_path, get_res=True)

    if C == False:
        print("  ERROR: file is unreadable or the chosen atom type is absent"); return None

    # ===== Secondary structure (protein only, via STRIDE) =====
    # Run STRIDE on the whole structure once: this yields the correct per-chain SS
    # for every protein chain, so each chain is later clustered independently.
    set_options(t=(y[9] if len(y) > 9 else None),
                   post=(y[10] if len(y) > 10 else True))
    set_structure_path(in_path)

    chain_types = detect_chain_types(in_path)
    skipped_na = sorted(c for c, ty in chain_types.items() if ty != 'protein')
    if skipped_na:
        print(f"  nucleic-acid chain(s) skipped (BiHC is a protein method): {', '.join(skipped_na)}")

    ss_data = get_secondary_structure_stride_auto(in_path, chain_types=chain_types)

    if '\\' in in_path:
        filename = ''.join(in_path.split('\\')[-1]).replace('.pdb', '').replace('.mmcif', '').replace('.cif', '').replace('.ent', '')
    else:
        filename = ''.join(in_path.split('/')[-1]).replace('.pdb', '').replace('.mmcif', '').replace('.cif', '').replace('.ent', '')


    data, res_num_array, removed_chain_index, res_labels_array = check_C(C, x[-1])

    all_chains = C[1]
    remaining_chains = [all_chains[i] for i in range(len(all_chains))
                        if i not in removed_chain_index]

    if data == False:
        print("  ERROR: file is unreadable"); return None

    num_chains = len(remaining_chains)
    if y[1]:
        print("Number of chains: ", num_chains, end='\n\n')

    if len(data) == 0:
        print(f"  no chain long enough to process (minimum length -m / --min-length = {x[-1]} residues)"); return None

    result = {filename: {}}
    DOMAIN_ROWS[filename] = []
    graph_chains = []

    if y[5] == 'longest':
        print('Processing longest valid chain!', end='\n\n')
        chains = list(dict.fromkeys(i for i in remaining_chains     # file order (a set's order changes between runs)
                                    if len(data[remaining_chains.index(i)]) == max(len(data[j]) for j in range(len(data)))))
    elif y[5]:
        print(f'Processing chain {y[5]}!', end='\n\n')
        chains = list(dict.fromkeys(i for i in remaining_chains if i in [y[5] + '_', y[5]]))
        if not chains:
            print(f"  chain {y[5]} not found (or shorter than the minimum length -m / --min-length = {x[-1]})"); return None
    else:
        print('Processing each chain independently!', end='\n\n')
        chains = list(dict.fromkeys(remaining_chains))

    # Map chain_id -> index in data/res_num_array/res_labels_array
    chain_to_data_idx = {chain_id: idx for idx, chain_id in enumerate(remaining_chains)}

    for chain_id in chains:
        data_idx = chain_to_data_idx[chain_id]

        subdata = data[data_idx]
        res_num = res_num_array[data_idx]
        current_res_labels = res_labels_array[data_idx]

        if y[1]:
            print(f"\nProcessing chain: {chain_id}")
            print(f"  Residue range: {current_res_labels[0]} - {current_res_labels[-1]}")
            print(f"  Number of residues: {len(current_res_labels)}")

        if 'MODEL' in chain_id:
            chain = chain_id.split('_')[0]
        else:
            chain = chain_id.replace('_', '')

        name = filename + f'_chain_{chain}'

        if 'ss_by_chain' not in locals():
            ss_by_chain = {}

        # Align protein SSE to current residue order (best effort)
        if chain_types.get(chain) == 'protein' and chain in ss_data:
            ss_map = {str(r): s for (r, s) in ss_data[chain]}
            ss_aligned = [ss_map.get(str(r), 'C') for r in current_res_labels]
            if len(ss_aligned) != len(subdata):
                ss_aligned = ss_aligned[:len(subdata)] + ['C'] * max(0, len(subdata) - len(ss_aligned))
            ss_by_chain[chain_id] = ss_aligned

        if 'res_labels_by_chain' not in locals():
            res_labels_by_chain = {}
        res_labels_by_chain[chain_id] = list(current_res_labels)

        set_ss_by_chain(ss_by_chain)
        set_res_labels_by_chain(res_labels_by_chain)

        pred = cluster_algo(subdata, *x[1:], chain_id, chain_types.get(chain))
        graph_chains.append({'chain_id': chain_id, 'graph': last_graph(), 'data': subdata, 'res': res_num,
                             'res_labels': list(current_res_labels),
                             'missing_res': missing_res[chain_id] if chain_id in missing_res else None})
        pe, de, re_, le = with_hetero(in_path, chain_id, pred, subdata, res_num, current_res_labels)
        report_chain(result, filename, chain_id, pe, de, re_, le,
                                 missing_res[chain_id] if chain_id in missing_res else None, y)

    export(result, filename, in_path, y, batch, graph_chains, x[-1])
    return result


if __name__ == "__main__":
    x, y = process_args()

    if not Path(x[0]).exists():
        sys.exit("Input path does not exist!")

    graph_mode = len(y) > 12 and y[12]
    targets = collect_graphs(x[0]) if graph_mode else collect_inputs(x[0])
    if not targets:
        sys.exit(f"No graph file (*_graph.npz) found in {x[0]}" if graph_mode else
                 f"No structure file ({', '.join(EXTS)}) found in {x[0]}")
    run = process_graph if graph_mode else process_one

    batch = len(targets) > 1 or Path(x[0]).is_dir()
    if batch:
        decorate_message(f"Batch: {len(targets)} {'graph' if graph_mode else 'structure'} file(s) in {x[0]}")

    failed = []
    for n, path in enumerate(targets, 1):
        if batch:
            print(f"\n[{n}/{len(targets)}] {path}")
        try:
            if run(path, x, y, batch=batch) is None and batch:
                failed.append(path)
        except Exception as e:
            if not batch:
                raise
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed.append(path)

    if batch:
        print(f"\nProcessed {len(targets) - len(failed)}/{len(targets)} files"
              + (f"; failed: {', '.join(os.path.basename(f) for f in failed)}" if failed else ""))
