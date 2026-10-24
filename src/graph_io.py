"""
Saved unit graphs (<name>_graph.npz): everything BiHC computes before the partition selection, for every processed
chain of one structure file, so that a new threshold t (or --no-post) is applied without re-running STRIDE / Voronota:

    python BiHC.py -i 1a4i.pdb -o out/                      # also writes out/1a4i_graph.npz
    python BiHC.py -i out/1a4i_graph.npz -t 0.40 -o out_t040/

Per chain: the unit graph (contact-area matrices W / Wh, unit sizes, SSE units, residue -> unit map, secondary structure),
the bottom-up merge trajectory, which contact-area backend was used, and what the JSON output reports (CA coordinates,
residue numbers / labels, missing residues). Each chain also holds every candidate partition (bottom-up stop points
and top-down partitions) with its F values -- the t-independent part of the selection (bihc_hm.scored_candidates) -- so
a new t only re-scores the margins and re-runs the clean-up; when a file holds the graph only, the candidates are
recomputed from W / Wh. Stored as a NumPy .npz without pickled objects (np.load(..., allow_pickle=False)), so
opening a file received from someone else cannot execute code.
"""
import json

import numpy as np

FORMAT = 'bihc-graph'
VERSION = 2


def pack(g, prefix=''):
    """(meta, arrays) of one chain's graph (bihc_core.LAST_GRAPH); array names start with prefix"""
    mc = {'N': int(g['N']), 'trivial': bool(g.get('trivial'))}
    arrays = {}
    if not mc['trivial']:
        mc.update(exact=bool(g['exact']), F_default=float(g['F_default']), beta=float(g['beta']), graph_kind=g['graph_kind'])
        arrays[prefix + 'W'] = np.asarray(g['W'], float)
        arrays[prefix + 'Wh'] = np.asarray(g['Wh'], float)
        arrays[prefix + 'usz'] = np.asarray(g['usz'], float)
        arrays[prefix + 'units'] = np.asarray(g['units'], int).reshape(-1, 2)
        arrays[prefix + 'memb'] = np.asarray(g['memb'], int)
        arrays[prefix + 'merges'] = np.asarray(g['merges'], float).reshape(-1, 3)
        arrays[prefix + 'ss'] = np.asarray(g['ss'], dtype=str)
        if g.get('scored') is not None:                           # candidates + F (t-independent)
            sc, U = g['scored'], len(g['units'])
            mc['scored'] = True
            arrays[prefix + 'cand'] = np.asarray([ul for ul, _, _ in sc], int).reshape(-1, U)
            arrays[prefix + 'candF'] = np.asarray([x for _, Fp, _ in sc for x in np.asarray(Fp, float)], float)
            arrays[prefix + 'candF_n'] = np.asarray([len(Fp) for _, Fp, _ in sc], int)
            arrays[prefix + 'cuts'] = np.asarray([x for _, _, cu in sc for x in cu], float)
            arrays[prefix + 'cuts_n'] = np.asarray([len(cu) for _, _, cu in sc], int)
    return mc, arrays


def unpack(mc, z, prefix=''):
    """the graph dict of one chain from its meta and the opened .npz"""
    if mc['trivial']:
        return {'trivial': True, 'N': mc['N']}
    g = {'trivial': False, 'N': mc['N'], 'W': z[prefix + 'W'], 'Wh': z[prefix + 'Wh'], 'usz': z[prefix + 'usz'],
         'units': [(int(a), int(b)) for a, b in z[prefix + 'units']], 'memb': z[prefix + 'memb'],
         'merges': [(int(a), int(b), float(r)) for a, b, r in z[prefix + 'merges']],
         'ss': [str(x) for x in z[prefix + 'ss']], 'exact': mc['exact'], 'F_default': mc['F_default'],
         'beta': mc['beta'], 'graph_kind': mc['graph_kind'], 'scored': None}
    if mc.get('scored'):
        cand, F, nF, cu, nc = (z[prefix + x] for x in ('cand', 'candF', 'candF_n', 'cuts', 'cuts_n'))
        oF, oc = np.concatenate([[0], np.cumsum(nF)]), np.concatenate([[0], np.cumsum(nc)])
        g['scored'] = [(cand[j], F[oF[j]:oF[j + 1]], [float(x) for x in cu[oc[j]:oc[j + 1]]]) for j in range(len(cand))]
    return g


def save(path, source, filename, min_length, chains):
    """chains: list of dicts with chain_id, graph (bihc_core.LAST_GRAPH), data, res, res_labels, missing_res"""
    arrays, meta_chains = {}, []
    for i, c in enumerate(chains):
        mc, a = pack(c['graph'], f'c{i}_')
        mc = {'chain_id': c['chain_id'], **mc, 'missing_res': c['missing_res']}
        arrays.update(a)
        arrays[f'c{i}_data'] = np.asarray(c['data'], float)
        arrays[f'c{i}_res'] = np.asarray(c['res'], int)
        arrays[f'c{i}_res_labels'] = np.asarray([str(x) for x in c['res_labels']], dtype=str)
        meta_chains.append(mc)
    meta = {'format': FORMAT, 'version': VERSION, 'source': source, 'filename': filename,
            'min_length': int(min_length), 'chains': meta_chains}
    arrays['meta'] = np.asarray(json.dumps(meta, default=_plain))
    with open(path, 'wb') as fh:            # a file handle: np.savez would otherwise append '.npz' to the name
        np.savez_compressed(fh, **arrays)


def load(path):
    """-> (meta, [chain dicts with chain_id, graph, data, res, res_labels, missing_res])"""
    with np.load(path, allow_pickle=False) as z:
        if 'meta' not in z.files:
            raise ValueError(f'{path} is not a BiHC graph file')
        meta = json.loads(str(z['meta']))
        if meta.get('format') != FORMAT:
            raise ValueError(f'{path} is not a BiHC graph file')
        if meta.get('version', 0) > VERSION:
            raise ValueError(f"{path} was written by a newer BiHC (graph format {meta['version']}); please update BiHC")
        out = []
        for i, mc in enumerate(meta['chains']):
            out.append({'chain_id': mc['chain_id'], 'graph': unpack(mc, z, f'c{i}_'), 'data': z[f'c{i}_data'].tolist(),
                        'res': z[f'c{i}_res'].tolist(), 'res_labels': [str(x) for x in z[f'c{i}_res_labels']],
                        'missing_res': mc['missing_res']})
    return meta, out


def _plain(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f'not serialisable: {type(x).__name__}')
