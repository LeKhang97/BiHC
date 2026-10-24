"""Unit tests of the HM pipeline (src/bihc_hm.py, src/bihc_post.py, src/bihc_core.py).
Run from the repository root:  python3 -m pytest tests"""
import os, shutil

import numpy as np
import pytest

from src import bihc_hm, bihc_post, bihc_core

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(HERE, 'examples')
# STRIDE and Voronota are installed by the user (README, "Installation"); tests that run them are skipped without them
HAVE_PROGRAMS = bihc_core.find_program('stride') is not None and bihc_core.backend() == 'voronota'
needs_programs = pytest.mark.skipif(not HAVE_PROGRAMS, reason='STRIDE / Voronota not installed (README, "Installation")')


def block_graph(blocks, unit_len=20, within=30.0, between=0.5, self_area=40.0):
    """units in sequence order; blocks = list of unit counts per domain; returns W, Wh, usz, units, memb"""
    U = sum(blocks); lab = np.repeat(np.arange(len(blocks)), blocks)
    W = np.where(lab[:, None] == lab[None, :], within, between).astype(float)
    np.fill_diagonal(W, self_area)
    Wh = 0.6 * W
    usz = np.full(U, float(unit_len))
    units = [(i * unit_len, (i + 1) * unit_len - 1) for i in range(U)]
    memb = np.repeat(np.arange(U), unit_len)
    return W, Wh, usz, units, memb, lab


def select(W, Wh, usz, t=bihc_hm.T_HM):
    return bihc_hm.select(W, Wh, usz, bihc_core.agglomerate(W, Wh, usz), t)


def n_dom(ul):
    return len(np.unique(ul))


def test_two_blocks_give_two_domains():
    W, Wh, usz, *_ , lab = block_graph([5, 5])
    ul = select(W, Wh, usz)
    assert n_dom(ul) == 2
    assert all(len(np.unique(ul[lab == d])) == 1 for d in (0, 1))


def test_one_block_gives_one_domain():
    W, Wh, usz, *_ = block_graph([10])
    assert n_dom(select(W, Wh, usz)) == 1


def test_three_blocks_and_insertion():
    # discontinuous domain: units of block 0 on both sides of block 1 (insertion) -> needs the double cut
    W, Wh, usz, units, memb, _ = block_graph([3, 4, 3])
    lab = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    W = np.where(lab[:, None] == lab[None, :], 30.0, 0.5); np.fill_diagonal(W, 40.0); Wh = 0.6 * W
    ul = select(W, Wh, usz)
    assert n_dom(ul) == 2
    assert len(np.unique(ul[lab == 0])) == 1 and ul[0] != ul[4]


def test_short_chain_is_one_domain():
    W, Wh, usz, *_ = block_graph([2, 1], unit_len=25)          # 75 residues < 80
    assert float(usz.sum()) < bihc_hm.MIN_SPLIT
    assert n_dom(select(W, Wh, usz)) == 1


def test_candidates_span_both_directions():
    W, Wh, usz, *_ = block_graph([4, 4, 4])
    merges = bihc_core.agglomerate(W, Wh, usz)
    c = bihc_hm.candidates(W, Wh, usz, merges)
    nbu = len(merges) + 1
    assert n_dom(c[0]) == W.shape[0] and n_dom(c[nbu - 1]) == 1        # bottom-up: all units ... one domain
    assert n_dom(c[nbu]) == 1                                            # top-down starts at one domain
    ks = [n_dom(u) for u in c[nbu:]]
    assert ks == sorted(ks) and ks[-1] >= 3                              # and splits to the end of the tree


def test_best_cut_respects_min_fragment():
    W, Wh, usz, *_ = block_graph([2, 2], unit_len=20)                    # 80 residues: only the 40 | 40 cut is admissible
    f, a, b = bihc_hm.best_cut(W, Wh, usz, list(range(4)))
    assert sorted(map(len, (a, b))) == [2, 2]
    W, Wh, usz, *_ = block_graph([3], unit_len=20)                       # 60 residues: no admissible cut
    assert bihc_hm.best_cut(W, Wh, usz, [0, 1, 2]) is None


def test_higher_t_never_merges_a_clear_split():
    W, Wh, usz, *_ = block_graph([5, 5])
    assert all(n_dom(select(W, Wh, usz, t)) == 2 for t in (0.3, 0.375, 0.45))


def test_postprocess_absorbs_short_segment():
    W, Wh, usz, units, memb, _ = block_graph([3, 1, 3])                  # a 20-residue middle domain
    ul = np.array([0, 0, 0, 1, 2, 2, 2])
    ss = (['H'] * 15 + ['C'] * 5) * len(units)                          # one helix per unit
    lab = bihc_post.postprocess_hm(ul, units, memb, W, Wh, usz, ss, bihc_core.ratio)
    assert len(np.unique(lab)) == 2


@needs_programs
def test_cli_end_to_end_1a4i(tmp_path):
    import subprocess, sys
    pdb = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(pdb):
        pytest.skip('example structure not shipped')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'BiHC.py'), '-i', pdb, '-c', 'A'], capture_output=True, text=True,
                       cwd=HERE, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    assert 'harmonic-mean' in r.stdout and 'Cluster 1 positions' in r.stdout


def test_postprocess_min2ss_removes_lone_helix_domain():
    W, Wh, usz, units, memb, _ = block_graph([3, 3, 3])
    ul = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ss = (['H'] * 15 + ['C'] * 5) * 3 + ['H'] * 60 + (['H'] * 15 + ['C'] * 5) * 3   # middle domain = one 60-residue helix
    lab = bihc_post.postprocess_hm(ul, units, memb, W, Wh, usz, ss, bihc_core.ratio)
    assert len(np.unique(lab)) == 2


def _unk_variant(tmp_path):
    """chain A of 1a4i with residues 20-40 renamed UNK (backbone + CB kept) and a free SAM 'ligand' carrying N, CA, C"""
    pdb = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(pdb):
        pytest.skip('example structure not shipped')
    out, last = [], None
    for ln in open(pdb):
        if ln.startswith('ENDMDL'):
            break
        if ln.startswith(('ATOM', 'HETATM')) and ln[21] == 'A' and ln[17:20].strip() != 'HOH':
            if ln.startswith('ATOM') and 20 <= int(ln[22:26]) <= 40:
                if ln[12:16].strip() not in ('N', 'CA', 'C', 'O', 'CB'):
                    continue
                ln = ln[:17] + 'UNK' + ln[20:]
            out.append(ln.rstrip('\n')); last = ln
    for i, (name, x) in enumerate((('N', 0.0), ('CA', 1.45), ('C', 2.9))):      # far from the protein, bonded to nothing
        out.append(f"HETATM{9000 + i:5d}  {name:<3s} SAM A 901    {80.0 + x:8.3f}{80.0:8.3f}{80.0:8.3f}  1.00 20.00           {name[0]:>2s}")
    f = tmp_path / 'unk.pdb'; f.write_text('\n'.join(out) + '\nEND\n')
    return str(f)


def test_unk_residues_are_part_of_the_chain_ligands_are_not(tmp_path):
    from src import Functions
    f = _unk_variant(tmp_path)
    atoms = bihc_core._parse_atoms(f)[0]['A']
    assert all(str(n) in atoms for n in range(20, 41)), 'UNK residues must be kept by the contact-area parser'
    assert '901' not in atoms, 'a ligand that is not peptide-bonded must not become a chain residue'
    res = Functions.process_structure(f, atom_type='CA', models=False, get_res=True, filename=f)
    labels = res[3][0]
    assert all(str(n) in labels for n in range(20, 41)) and '901' not in labels


def test_unk_chain_uses_voronota_not_the_ca_fallback(tmp_path):
    from src import Functions
    if bihc_core.backend() != 'voronota':
        pytest.skip('Voronota binary not available')
    f = _unk_variant(tmp_path)
    xs, ys, zs = Functions.process_structure(f, atom_type='CA', models=False, get_res=True, filename=f)[0][0]
    labels = Functions.process_structure(f, atom_type='CA', models=False, get_res=True, filename=f)[3][0]
    bihc_core.set_structure_path(f); bihc_core._STRUCT_CACHE.clear()
    got = bihc_core._chain_atoms('A', labels, ca_ref=np.c_[xs, ys, zs])
    assert got is not None, 'coordinates must resolve (no CA-contact fallback)'
    g = got[2]; i = labels.index('25')
    assert abs(g[i] - (bihc_core.KD.get('UNK', -0.4) + 4.5) / 9.0) < 1e-9, 'UNK gets the default hydropathy'


def _cli(*args):
    import subprocess, sys
    pdb = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(pdb):
        pytest.skip('example structure not shipped')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'BiHC.py'), '-i', pdb, '-c', 'A', *args], capture_output=True,
                       text=True, cwd=HERE, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    return r.stdout


@needs_programs
def test_cli_threshold_and_min_length_flags():
    assert 'threshold t: 0.365' in _cli()
    hi = _cli('--threshold', '0.55')
    assert 'threshold t: 0.55' in hi and 'Cluster 2 positions' in hi          # higher t splits 1a4i A
    assert _cli('-t', '0.55').split('Output information')[1] == hi.split('Output information')[1]
    assert 'no chain long enough' in _cli('-m', '400') and 'no chain long enough' in _cli('--min-length', '400')


def _run(*args, cwd=HERE):
    import subprocess, sys
    return subprocess.run([sys.executable, os.path.join(HERE, 'BiHC.py'), *args], capture_output=True, text=True,
                          cwd=cwd, timeout=600)


@needs_programs
def test_graph_file_reproduces_the_full_run(tmp_path):
    import json
    pdb = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(pdb):
        pytest.skip('example structure not shipped')
    full = tmp_path / 'full'
    assert _run('-i', pdb, '-o', str(full)).returncode == 0
    graph = full / '1a4i_graph.npz'
    assert graph.exists()                                              # written by default with the outputs
    for t, extra in (('0.375', []), ('0.55', []), ('0.55', ['--no-post'])):
        a, b = tmp_path / f'a{t}{extra}', tmp_path / f'b{t}{extra}'
        assert _run('-i', pdb, '-t', t, '--no-graph', '-o', str(a), *extra).returncode == 0
        r = _run('-i', str(graph), '-t', t, '-o', str(b), *extra)          # a saved graph is an -i input too
        assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
        assert json.load(open(a / '1a4i.json')) == json.load(open(b / '1a4i.json'))
        assert (a / '1a4i_domains.tsv').read_text() == (b / '1a4i_domains.tsv').read_text()
        assert not (a / '1a4i_graph.npz').exists()                     # --no-graph


@needs_programs
def test_graph_mode_refuses_a_structure_input(tmp_path):
    pdb = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(pdb):
        pytest.skip('example structure not shipped')
    assert _run('-i', pdb, '-c', 'A', '-o', str(tmp_path), '--output-pdb').returncode == 0
    graph = str(tmp_path / '1a4i_graph.npz')
    for args in (('-i', graph, '--output-pdb'), ()):
        r = _run(*args, '-o', str(tmp_path / 'x'))
        assert r.returncode != 0 and 'error' in r.stderr
    assert (tmp_path / '1a4i.pdb').exists()                # the -o directory holds graphs AND structures
    r = _run('-i', str(tmp_path), '-o', str(tmp_path / 'y'))     # -> its graphs are read, the structures ignored
    assert r.returncode == 0 and 'saved graph' in r.stdout and 'Batch: 1 graph' in r.stdout, r.stdout[-800:] + r.stderr[-500:]


def _hetatm_variant(tmp_path, resno=50):
    """chain A of 1a4i with residue `resno` written as a HETATM modified residue (MSE)"""
    pdb = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(pdb):
        pytest.skip('example structure not shipped')
    out = []
    for ln in open(pdb):
        if ln.startswith('ENDMDL'):
            break
        if ln.startswith(('ATOM', 'HETATM')) and ln[21] == 'A' and ln[17:20].strip() != 'HOH':
            if ln.startswith('ATOM') and int(ln[22:26]) == resno:
                ln = 'HETATM' + ln[6:17] + 'MSE' + ln[20:]
            out.append(ln.rstrip('\n'))
    f = tmp_path / 'mse.pdb'; f.write_text('\n'.join(out) + '\nEND\n')
    return str(f)


def test_hetatm_residue_out_of_the_graph_back_in_the_output(tmp_path):
    from src import Functions
    f = _hetatm_variant(tmp_path)
    labels = Functions.process_structure(f, atom_type='CA', models=False, get_res=True, filename=f)[3][0]
    assert '50' not in labels and '49' in labels and '51' in labels, 'a HETATM residue is not a graph residue'
    order = Functions.hetero_polymer_order(f, 'A', labels)
    assert order is not None and [l for l, h, _ in order if h] == ['50']
    pred = [0 if int(l) < 100 else 1 for l in labels]
    lab, ext, _ = Functions.fill_hetero(pred, labels, order)
    assert ext[ext.index('50')] == '50' and lab[ext.index('50')] == 0, 'inside one domain it takes that domain'
    lab, ext, _ = Functions.fill_hetero([0 if int(l) < 50 else 1 for l in labels], labels, order)
    assert lab[ext.index('50')] == 1, 'between two domains a single residue goes to the larger one'


@needs_programs
def test_cli_output_has_no_hole_at_a_hetatm_residue(tmp_path):
    import subprocess, sys, json
    f = _hetatm_variant(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'BiHC.py'), '-i', f, '-c', 'A', '-o', str(tmp_path), '--no-graph'],
                       capture_output=True, text=True, cwd=HERE, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(tmp_path / 'mse.json'))['mse']['A']
    assert '50' in d['res_labels']
    i = d['res_labels'].index('50')
    assert d['cluster'][i] in (d['cluster'][i - 1], d['cluster'][i + 1])


@needs_programs
def test_structure_output_uses_author_numbering(tmp_path):
    import subprocess, sys
    src = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(src):
        pytest.skip('example structure not shipped')
    # renumber chain A by +1000 so that author numbers and internal positions differ
    out = []
    for ln in open(src):
        if ln.startswith(('ATOM', 'HETATM')) and ln[21] == 'A':
            ln = ln[:22] + f'{int(ln[22:26]) + 1000:4d}' + ln[26:]
        out.append(ln)
    f = tmp_path / 'ren.pdb'; f.write_text(''.join(out))
    od = tmp_path / 'out'
    r = subprocess.run([sys.executable, os.path.join(HERE, 'BiHC.py'), '-i', str(f), '-c', 'A', '-o', str(od), '--output-pdb', '--no-graph'],
                       capture_output=True, text=True, cwd=HERE, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    got = {int(l[22:26]) for l in open(od / 'ren.pdb') if l.startswith('ATOM')}
    want = {int(l[22:26]) for l in out if l.startswith('ATOM') and l[21] == 'A' and l[12:16].strip() == 'CA'}
    assert want <= got, 'every residue of the chain must be written, matched on its author number'


def test_domain_ranges_join_missing_residues_only():
    from src.Functions import domain_ranges, command_pymol_with_labels
    chain = ['1', '2', '5', '6', '7', '8']                     # 3-4 have no coordinates
    assert domain_ranges(['1', '2', '5', '6'], chain) == [('1', '6')], 'a gap of missing residues does not cut a domain'
    assert domain_ranges(['1', '2', '6', '7'], chain) == [('1', '2'), ('6', '7')], 'a residue of the structure in between does'
    perm = [str(x) for x in list(range(102, 120)) + list(range(2, 14)) + list(range(381, 390))]   # non-monotonic chain
    assert domain_ranges(perm, perm) == [('2', '13'), ('102', '119'), ('381', '389')], 'segments in residue-number order'
    assert command_pymol_with_labels(['-2', '-1', '0'], 'd', 'blue', chain_labels=['-2', '-1', '0']).startswith(r'select d, res \-2-0')


def _fake_voronota(d):
    d.mkdir(exist_ok=True)
    p = d / 'voronota'; p.write_text('#!/bin/sh\necho "Voronota version 1.29.4602"\n'); p.chmod(0o755)
    return str(p)


def test_programs_are_found_on_the_path_or_where_given(tmp_path, monkeypatch):
    monkeypatch.setattr(bihc_core, '_HERE', str(tmp_path / 'src'))           # no copy next to the code
    monkeypatch.delenv('BIHC_VORONOTA', raising=False)
    on_path = _fake_voronota(tmp_path / 'bin')
    monkeypatch.setenv('PATH', str(tmp_path / 'bin'))
    assert bihc_core.find_program('voronota') == on_path
    assert bihc_core.voronota_version(on_path) == '1.29.4602'
    given = _fake_voronota(tmp_path / 'other')
    monkeypatch.setenv('BIHC_VORONOTA', str(tmp_path / 'other'))           # a directory holding it
    assert bihc_core.find_program('voronota') == given
    monkeypatch.setenv('BIHC_VORONOTA', str(tmp_path / 'missing'))
    assert bihc_core.find_program('voronota') is None, 'a wrong explicit path is never replaced by the copy on the PATH'
    assert '--voronota' in bihc_core.missing_program_message('voronota')


def test_cli_refuses_to_run_without_the_programs(tmp_path):
    import subprocess, sys
    pdb = os.path.join(EXAMPLES, '1a45.pdb')
    env = dict(os.environ, BIHC_STRIDE=str(tmp_path / 'no_stride'), BIHC_VORONOTA=str(tmp_path / 'no_voronota'))
    run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, 'BiHC.py'), '-i', pdb, *a], capture_output=True,
                                    text=True, cwd=HERE, env=env, timeout=600)
    r = run()
    assert r.returncode != 0 and 'STRIDE' in r.stderr and 'Voronota 1.29' in r.stderr and 'README' in r.stderr
    r = run('--allow-lattice')                    # the lattice replaces Voronota on request only; STRIDE has no stand-in
    assert r.returncode != 0 and 'STRIDE' in r.stderr and 'Voronota 1.29' not in r.stderr


@needs_programs
def test_tsv_stdout_and_pymol_agree_and_missing_residues_are_bridged(tmp_path):
    import re
    pdb = os.path.join(EXAMPLES, '1a4i.pdb')
    if not os.path.exists(pdb):
        pytest.skip('example structure not shipped')
    r = _run('-i', pdb, '-o', str(tmp_path), '--no-graph', '--output-pdb')
    assert r.returncode == 0, r.stderr[-500:]
    rows = [l.split('\t') for l in (tmp_path / '1a4i_domains.tsv').read_text().splitlines()]
    assert rows[0] == ['structure', 'chain', 'domain', 'n_residues', 'res_positions', 'missing_res']
    a = [x for x in rows[1:] if x[1] == 'A']
    assert [x[4] for x in a] == ['2-296'] and a[0][5] == '241-250', '241-250 have no coordinates in chain A'
    assert 'cluster 1 spans residues 241-250 without coordinates' in r.stdout
    printed = re.findall(r'Cluster \d+ positions:\n(.+)', r.stdout)
    assert [x[4] for x in rows[1:]] == [p.replace(' ', '') for p in printed], 'TSV = standard output'
    for chain in ('A', 'B'):                              # two chains: one structure file + .pml per chain
        struct = tmp_path / f'1a4i_{chain}.pdb'
        assert {l[21] for l in open(struct) if l.startswith(('ATOM', 'HETATM'))} == {chain}
        pml = (tmp_path / f'1a4i_{chain}_pymolcmd.pml').read_text()
        assert pml.startswith(f'load {struct}, 1a4i_{chain}; '), 'the .pml loads its chain file by its absolute path'
        for x in (x for x in rows[1:] if x[1] == chain):
            sel = f"select 1a4i_{chain}_cluster_{x[2]}, 1a4i_{chain} and chain {chain} and resi {x[4].replace(',', '+')};"
            assert sel in pml, 'TSV = .pml'
    assert not (tmp_path / '1a4i.pdb').exists() and not (tmp_path / '1a4i_pymolcmd.pml').exists()
    js = __import__('json').load(open(tmp_path / '1a4i.json'))['1a4i']
    assert all('PyMOL' not in c for c in js.values()), 'the JSON holds no PyMOL commands (the .pml does)'


@needs_programs
def test_default_outputs_are_json_tsv_and_graph(tmp_path):
    pdb = os.path.join(EXAMPLES, '1a45.pdb')
    assert _run('-i', pdb, '-o', str(tmp_path)).returncode == 0
    assert sorted(os.listdir(tmp_path)) == ['1a45.json', '1a45_domains.tsv', '1a45_graph.npz']
    assert _run('-i', pdb, '-o', str(tmp_path / 'all'), '--output-pdb').returncode == 0          # + the .pml
    assert sorted(os.listdir(tmp_path / 'all')) == ['1a45.json', '1a45.pdb', '1a45_domains.tsv', '1a45_graph.npz',
                                                    '1a45_pymolcmd.pml']
