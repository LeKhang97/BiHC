"""
BiHC -- graph, self-containment score, bottom-up merge and the public driver predict().

    STRIDE SSE super-nodes
      -> edge weights = VORONOI CONTACT AREA x mean nonpolarity
      -> greedy bottom-up agglomeration on the self-containment ratio
             F(A,B) = (ic_h / mu^beta) / ave ,   mu = nA*nB/(nA+nB) ,  beta = 0.86
             ave    = (intra_A + intra_B + ic) / (nA + nB)
      -> candidate partitions: every stop point of that trajectory + every top-down best-first partition down to the
         finest admissible split (lowest-F single or double cut on the SSE order, parts >= 40 res)  (bihc_hm.py)
      -> choose the admissible candidate (every cut interface F < t, every domain's weakest cut > t) with the
         highest harmonic mean of the margins at threshold t; chains < 80 residues are one domain     (bihc_hm.py)
      -> post-processing, every merge into the sequence-adjacent domain with the highest F:          (bihc_post.py)
           MINSEGMENT  sequence segments < 40 residues
           min2ss      domains with < 2 secondary-structure components (STRIDE H/G/E runs)
         (post=False returns the chosen partition without any post-processing)
      fallback graphs (lattice areas, CA contacts): the trajectory is cut at their own F instead of HM

Deterministic: no model is fitted; t is the single operating parameter.

Contacts are Voronoi facet areas rather than distance cut-offs: two residues are neighbours only when their Voronoi
cells share a face, so a pair screened by a third residue contributes nothing, a long side-chain contact is not
missed, and the weight is a physical area rather than a count.

Public API:
    set_structure_path(path)                             # once per input file
    predict(ca, ss, res_labels, chain_id, F=..., post=True)     # -> one domain label per residue
"""
import itertools
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.spatial import distance_matrix

try:
    from scipy.spatial import Voronoi, cKDTree
    _HAVE_VORONOI = True
except Exception:                                        # pragma: no cover
    _HAVE_VORONOI = False

# ---------------------------------------------------------------- constants
BETA = 0.86         # exponent of the reduced size mu^beta in F
F_CUT_VORONOTA = 0.40   # F at which partition_at() cuts the merge trajectory on Voronota areas
F_CUT_LATTICE = 0.40    # the same cut on lattice areas (a different ratio scale)
F_CUT_FALLBACK = 0.30   # cut on the CA-contact-count graph (fallback path only; a different ratio scale)
MINDOM = 40         # a domain must reach this many residues
LMIN = 0            # chains shorter than this are one domain without analysis (0: off; with MINDOM = 40 every
                    # chain shorter than 80 residues has no admissible split anyway)
CONTACT_D = 8.0     # CA-CA cutoff, used only by the fallback graph when Voronoi is unavailable
VOR_GRID = 2.5      # solvent pseudo-atom grid spacing (A)
VOR_SHELL = (3.0, 6.0)   # keep grid points this far from the nearest protein atom
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'SE': 1.90}   # Bondi radii
VDW_DEFAULT = 1.70
R_PROBE = 1.4       # water probe; solvent centres sit on the solvent-accessible surface r_vdW + R_PROBE
SAS_POINTS = 64     # probe centres generated per atom (Fibonacci sphere), before burial filtering
# Contact-area backend. 'voronota' is the constrained additively weighted Voronoi tessellation of van-der-Waals balls with a
# 1.4 A probe: exact areas, no lattice and no spacing parameter. 'lattice' is the pure-Python approximation of the same
# areas, used only on request (BiHC.py --allow-lattice) or when the tessellation fails for a chain.
_HERE = os.path.dirname(os.path.abspath(__file__))

# External programs. STRIDE (free for academic use) and Voronota 1.29 (MIT) are not shipped with BiHC; the user
# installs them. Each is looked up in this order: the BIHC_STRIDE / BIHC_VORONOTA environment variable (BiHC.py
# --stride / --voronota set the same variables), an executable called stride / voronota in this directory, the PATH.
PROGRAM_ENV = {'stride': 'BIHC_STRIDE', 'voronota': 'BIHC_VORONOTA'}
PROGRAM_OPTION = {'stride': '--stride', 'voronota': '--voronota'}
PROGRAM_LABEL = {'stride': 'STRIDE (secondary structure)', 'voronota': 'Voronota 1.29 (contact areas)'}


def find_program(name):
    """Path of the 'stride' / 'voronota' executable, or None. A path set in the environment variable is never swapped
    for another copy: if it is not an executable file (or a directory holding one), the answer is None."""
    given = os.environ.get(PROGRAM_ENV[name])
    if given:
        given = os.path.expanduser(given)
        if os.path.isdir(given):
            given = os.path.join(given, name)
        return given if os.path.isfile(given) and os.access(given, os.X_OK) else None
    local = os.path.join(_HERE, name)
    if os.path.isfile(local):
        if not os.access(local, os.X_OK):
            try:
                os.chmod(local, 0o755)
            except OSError:
                pass
        if os.access(local, os.X_OK):
            return local
    return shutil.which(name)


def missing_program_message(name):
    """What to tell the user when find_program(name) is None."""
    given = os.environ.get(PROGRAM_ENV[name])
    head = (f'{PROGRAM_LABEL[name]}: the path given with {PROGRAM_OPTION[name]} / {PROGRAM_ENV[name]} is not an '
            f'executable file: {given}' if given else
            f'{PROGRAM_LABEL[name]}: no "{name}" executable found on the PATH or in {_HERE}')
    return (f'{head}\n  BiHC does not include it; install it as described under "Installation" in README.md, then '
            f'put it on the PATH or pass {PROGRAM_OPTION[name]} /path/to/{name}.')


def voronota_version(path=None):
    """'1.29.4602' etc. as printed by the executable, or None."""
    path = path or VORONOTA_BIN
    try:
        out = subprocess.run([path], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    m = re.search(r'Voronota version (\S+)', out.stdout + out.stderr)
    return m.group(1) if m else None


VORONOTA_BIN = find_program('voronota')      # None when not installed; BiHC.py --voronota resets it
VORONOTA_RADII = os.path.join(_HERE, 'voronota_radii')
SOLVENT_MODE = None        # None -> decide from availability; force with 'voronota' / 'lattice'


def backend():
    """'voronota' when the binary is usable, otherwise 'lattice'."""
    if SOLVENT_MODE:
        return SOLVENT_MODE
    return 'voronota' if (VORONOTA_BIN and os.access(VORONOTA_BIN, os.X_OK)
                          and os.path.exists(VORONOTA_RADII)) else 'lattice'


# Voronota writes a residue as 'c<H>r<31>R<GLY>', and 'c<H>r<31>i<A>R<GLY>' when it carries an insertion
# code. The insertion code is part of the residue identity: 31 and 31A are different residues.
_VORO_RE = re.compile(r'^c<([^>]*)>r<(-?\d+)>(?:i<([^>]*)>)?R<[^>]*>\s+'
                      r'c<([^>]*)>r<(-?\d+)>(?:i<([^>]*)>)?R<[^>]*>\s+([\d.eE+-]+)')


def _reskey_str(label, icode=None):
    """'31' / '31A' -- the same key `_reskey` builds from a parsed residue, but from a label or a
    number+code pair. Callers may pass plain integers (no insertion codes), which map to themselves."""
    if icode is not None:
        return f'{int(str(label).strip())}{str(icode).strip().upper()}'
    t = str(label).strip().upper()
    i = 1 if t[:1] == '-' else 0
    while i < len(t) and t[i].isdigit():
        i += 1
    num = t[:i]
    return f'{int(num)}{t[i:].strip()}' if num not in ('', '-') else t


def voronota_areas(path, chain, res_keys):
    """Exact residue-residue contact areas from the constrained additively weighted Voronoi tessellation (Voronota 1.x).

    No solvent lattice is involved: cells are those of the van-der-Waals balls, clipped by a 1.4 A water
    probe, so there is no spacing parameter to converge and the construction is rotation- and
    translation-invariant by nature. Atom radii enter properly, which the plain Voronoi fallback cannot do.
    """
    if not path or not os.path.exists(path):
        return None
    src = path
    tmpdir = None
    if not path.lower().endswith('.pdb'):
        from Bio.PDB import MMCIFParser, PDBIO
        tmpdir = tempfile.mkdtemp()
        src = os.path.join(tmpdir, 'in.pdb')
        io = PDBIO(); io.set_structure(MMCIFParser(QUIET=True).get_structure('s', path)); io.save(src)
    try:
        q = subprocess.run(
            f'"{VORONOTA_BIN}" get-balls-from-atoms-file --annotated --radii-file "{VORONOTA_RADII}" '
            f'< "{src}" | grep "^c<{chain}>" | '
            f'"{VORONOTA_BIN}" calculate-contacts --annotated --probe 1.4 | '
            f'"{VORONOTA_BIN}" query-contacts --inter-residue',
            shell=True, capture_output=True, text=True, timeout=900)
    except Exception:
        return None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    if q.returncode != 0 or not q.stdout.strip():
        return None
    # keyed by number+insertion code: keying by the number alone makes 31 and 31A the same row, which
    # silently empties one of them and turns it into an isolated unit
    pos = {}
    for i, r in enumerate(res_keys):
        pos.setdefault(_reskey_str(r), i)
    RA = np.zeros((len(res_keys), len(res_keys)))
    for line in q.stdout.splitlines():
        m = _VORO_RE.match(line)
        if not m:
            continue
        c1, r1, i1, c2, r2, i2, area = m.groups()
        if c1 != chain or c2 != chain:
            continue
        i, j = pos.get(_reskey_str(r1, i1 or '')), pos.get(_reskey_str(r2, i2 or ''))
        if i is None or j is None or i == j:
            continue
        RA[i, j] += float(area); RA[j, i] += float(area)
    return RA if RA.sum() > 0 else None
MIN_ATOMS_PER_RES = 3.0  # below this the structure is CA-only / coarse-grained and the tessellation is
                         # meaningless (its cells become huge), so the CA-contact graph is used instead

# Kyte-Doolittle hydropathy, rescaled to [0, 1] as (KD + 4.5) / 9 where it is used.
KD = {'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4,
      'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
      'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5}
AA3 = set(KD)
MODMAP = {'MSE': 'MET', 'SEC': 'CYS', 'PYL': 'LYS', 'CSO': 'CYS', 'PTR': 'TYR', 'SEP': 'SER',
          'TPO': 'THR', 'HYP': 'PRO', 'MLY': 'LYS', 'KCX': 'LYS', 'LLP': 'LYS', 'M3L': 'LYS'}

_STRUCT_PATH = None
_STRUCT_CACHE = {}


def set_structure_path(path):
    """Tell the module which file the coordinates came from (needed for the all-atom tessellation)."""
    global _STRUCT_PATH, _STRUCT_CACHE
    _STRUCT_PATH = path
    _STRUCT_CACHE = {}


# ---------------------------------------------------------------- structure reading
def _reskey(resseq, icode):
    ic = (icode or ' ').strip()
    return f'{int(resseq)}{ic}'


def is_polymer_residue(residues, i, max_peptide_bond=2.0):
    """True for an amino-acid residue of the polymer chain whatever its name: it has backbone N, CA and C and is
    peptide-bonded (C-N <= max_peptide_bond A) to the residue before or after it in the chain. This keeps UNK (unknown
    amino acid, usually modelled as backbone + CB) and modified residues missing from MODMAP, and leaves out ligands
    that merely carry an amino-acid moiety (e.g. SAM / SAH cofactors), which are not bonded into the chain."""
    r = residues[i]
    if r.id[0] == 'W' or not all(a in r for a in ('N', 'CA', 'C')):
        return False
    bonded = lambda c_atom, n_atom: np.linalg.norm(c_atom.get_coord() - n_atom.get_coord()) <= max_peptide_bond
    if i > 0 and 'C' in residues[i - 1] and bonded(residues[i - 1]['C'], r['N']):
        return True
    if i + 1 < len(residues) and 'N' in residues[i + 1] and bonded(r['C'], residues[i + 1]['N']):
        return True
    return False


def _parse_atoms(path):
    """[{chain_id: {res_key: (resname, CA xyz, heavy atom xyz)}}] -- one dict per model.

    Every model is kept, because an NMR ensemble is a set of different conformations and the caller may
    be working from any one of them; `_chain_atoms` then picks the model whose CA coordinates match the
    ones it was given, so the tessellation is always built on the same conformation as the input.
    """
    from Bio.PDB import PDBParser, MMCIFParser
    parser = (MMCIFParser(QUIET=True) if path.lower().endswith(('.cif', '.mmcif'))
              else PDBParser(QUIET=True))
    st = parser.get_structure('s', path)
    models = []
    for model in st:
        out = {}
        for ch in model:
            d = {}
            residues = list(ch)
            for ri, res in enumerate(residues):
                rn = res.get_resname().strip().upper()
                if rn not in AA3 and rn not in MODMAP and not is_polymer_residue(residues, ri):
                    continue
                xyz = [a.get_coord() for a in res
                       if a.element not in ('H', 'D') and not a.get_name().startswith('H')]
                if not xyz:
                    continue
                el = [(a.element or a.get_name()[:1]).strip().upper() for a in res
                      if a.element not in ('H', 'D') and not a.get_name().startswith('H')]
                ca = res['CA'].get_coord() if 'CA' in res else None
                d[_reskey(res.id[1], res.id[2])] = (MODMAP.get(rn, rn), ca,
                                                    np.asarray(xyz, float), el)
            if d:
                out[ch.id] = d
        if out:
            models.append(out)
    return models


def _pick_chain(models, cid):
    """the per-model chain dicts for `cid`, in model order (empty when the chain is absent)."""
    out = []
    for m in models:
        d = m.get(cid) or m.get(cid.strip())
        if d is None and len(m) == 1:
            d = next(iter(m.values()))
        if d:
            out.append(d)
    return out


def _chain_atoms(chain_id, res_labels, ca_ref=None):
    """(xyz Mx3, res_of M, g N) aligned to res_labels, or None when the chain cannot be resolved.

    With several models present, the one whose CA coordinates sit closest to `ca_ref` is used, so the
    tessellation describes the same conformation as the coordinates the caller passed in.
    """
    if _STRUCT_PATH is None or not os.path.exists(_STRUCT_PATH):
        return None
    if 'atoms' not in _STRUCT_CACHE:
        try:
            _STRUCT_CACHE['atoms'] = _parse_atoms(_STRUCT_PATH)
        except Exception:
            _STRUCT_CACHE['atoms'] = []
    cid = str(chain_id).replace('_', '').split('MODEL')[0]
    cands = _pick_chain(_STRUCT_CACHE['atoms'], cid)
    if not cands:
        return None
    d = cands[0]
    if len(cands) > 1 and ca_ref is not None:
        def misfit(dd):
            err = []
            for i, lab in enumerate(res_labels):
                e = dd.get(str(lab).strip())
                if e is not None and e[1] is not None and i < len(ca_ref):
                    err.append(float(np.sum((np.asarray(e[1], float) - ca_ref[i]) ** 2)))
            return float(np.mean(err)) if err else float('inf')
        d = min(cands, key=misfit)
    # secondary index by residue NUMBER alone, merging insertion-coded residues, so a residue
    # label written without its insertion code still resolves.
    by_seq = {}
    for k, (name, _ca, pts, _el) in d.items():
        try:
            n = int(''.join(c for c in k if c.isdigit() or c == '-'))
        except ValueError:
            continue
        if n in by_seq:
            by_seq[n] = (by_seq[n][0], by_seq[n][1], np.vstack([by_seq[n][2], pts]),
                         by_seq[n][3] + _el)
        else:
            by_seq[n] = (name, _ca, pts, _el)

    xyz = []; rof = []; els = []; g = np.full(len(res_labels), (0.0 + 4.5) / 9.0)
    hit = 0
    for i, lab in enumerate(res_labels):
        key = str(lab).strip()
        e = d.get(key)
        if e is None:
            try:
                e = by_seq.get(int(''.join(c for c in key if c.isdigit() or c == '-')))
            except ValueError:
                e = None
        if e is None:
            continue
        hit += 1
        name, _ca, pts, el = e
        g[i] = (KD.get(name, -0.4) + 4.5) / 9.0
        for p, e_ in zip(pts, el + ['C'] * max(0, len(pts) - len(el))):
            xyz.append(p); rof.append(i); els.append(e_)
    if hit < 0.8 * len(res_labels) or not xyz:
        return None
    return np.asarray(xyz, float), np.asarray(rof, int), g, np.asarray(els, object)


# ---------------------------------------------------------------- Voronoi contact areas
def _own_frame(xyz):
    """Rotate coordinates into the structure's own principal-axis frame.

    The solvent pseudo-atoms are laid on an axis-aligned cubic lattice, so building them in the file's
    frame would make the tessellation depend on the orientation of the input. In a frame derived from the
    coordinates themselves, rotating the input rotates the lattice with it, and facet areas are invariant under
    a global rotation. Axis signs are fixed by the third moment so the frame is deterministic rather than
    merely up to reflection; with near-degenerate principal moments the frame is only approximately determined.
    """
    c = xyz.mean(0)
    X = xyz - c
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return xyz
    R = vt.T                                        # columns = principal axes
    Y = X @ R
    for k in range(3):                              # deterministic sign: skew of each axis >= 0
        if np.sum(Y[:, k] ** 3) < 0:
            R[:, k] = -R[:, k]
    if np.linalg.det(R) < 0:                        # keep a right-handed frame
        R[:, 2] = -R[:, 2]
    return X @ R


def _fib_sphere(k):
    """k roughly uniform directions on the unit sphere (deterministic Fibonacci spiral)."""
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def solvent_sas(xyz, els, k=None):
    """Probe centres on the solvent-accessible surface: r_vdW + 1.4 A around each atom, buried ones dropped.

    Probe centres only have to cover a 2-D surface, unlike a volumetric lattice filling a 3-D shell, so fewer points
    are needed for the same sampling density. The radii are Bondi's and the probe is the standard 1.4 A water radius.
    """
    k = SAS_POINTS if k is None else k
    r = np.array([VDW.get(e, VDW_DEFAULT) for e in els], float) + R_PROBE
    U = _fib_sphere(k)
    P = (xyz[:, None, :] + r[:, None, None] * U[None, :, :]).reshape(-1, 3)
    # A probe centre is buried iff it lies inside some atom's own probe sphere. The nearest atoms are the
    # only candidates: any other atom is at least as far, so it can only contain the point if its radius is
    # larger, and the radii span just 0.28 A -- eight nearest neighbours is ample and keeps this vectorised.
    tree = cKDTree(xyz)
    kq = min(8, len(xyz))
    d, idx = tree.query(P, k=kq)
    if kq == 1:
        d = d[:, None]; idx = idx[:, None]
    buried = (d < r[idx] - 1e-6).any(axis=1)
    return P[~buried]


def contact_areas(xyz, rof, N, solvent=None):
    """residue x residue Voronoi contact-facet area (A^2); solvent pseudo-atoms bound surface cells.

    `solvent` overrides the built-in volumetric lattice with an explicit set of solvent points (see
    `solvent_sas`), which is how the lattice-free construction is driven.

    Facet areas are computed in batches grouped by vertex count: a Voronoi facet is a convex polygon whose
    plane normal is exactly the vector between the two generating atoms, so the polygon can be projected
    into that plane, its vertices sorted by angle and its area taken by the shoelace formula -- all of it
    vectorised across every facet with the same number of vertices, instead of one Python iteration per
    facet (there are tens of thousands of them per chain).
    """
    if not _HAVE_VORONOI:
        return None
    M = len(xyz)
    if solvent is None:
        xyz = _own_frame(xyz)      # solvent grid built in the protein's own frame -> rotation-invariant
        lo = xyz.min(0) - VOR_SHELL[1] - VOR_GRID
        hi = xyz.max(0) + VOR_SHELL[1] + VOR_GRID
        ax = [np.arange(lo[k], hi[k] + 1e-9, VOR_GRID) for k in range(3)]
        G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
        d, _ = cKDTree(xyz).query(G)
        shell = G[(d > VOR_SHELL[0]) & (d < VOR_SHELL[1])]
    else:
        shell = np.asarray(solvent, float)     # already expressed in the same frame as xyz
    if len(shell) < 10:
        return None
    pts = np.vstack([xyz, shell])
    try:
        vor = Voronoi(pts)
    except Exception:
        return None

    rp = np.asarray(vor.ridge_points, dtype=np.int64)
    rv = vor.ridge_vertices
    lens = np.fromiter((len(r) for r in rv), dtype=np.int64, count=len(rv))
    flat = np.fromiter(itertools.chain.from_iterable(rv), dtype=np.int64, count=int(lens.sum()))
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])

    # keep only protein-protein facets between two DIFFERENT residues, with at least a triangle of
    # finite vertices; a facet touching the unbounded cell (-1) is discarded.
    # rof is indexed by PROTEIN atom, while ridge_points indexes protein atoms and solvent points
    # alike, so the protein-protein mask has to be applied before rof is used at all.
    pp = (rp[:, 0] < M) & (rp[:, 1] < M) & (lens >= 3)
    ri = np.full(len(rp), -1, np.int64); rj = np.full(len(rp), -1, np.int64)
    ri[pp] = rof[rp[pp, 0]]; rj[pp] = rof[rp[pp, 1]]
    keep = pp & (ri != rj)
    if not keep.any():
        return np.zeros((N, N))

    RA = np.zeros((N, N))
    V = vor.vertices
    for k in np.unique(lens[keep]):
        sel = np.where(keep & (lens == k))[0]
        cols = starts[sel][:, None] + np.arange(k)
        verts = flat[cols]                                   # (n, k)
        ok = (verts >= 0).all(1)
        if not ok.any():
            continue
        sel = sel[ok]; verts = verts[ok]
        P = V[verts]                                         # (n, k, 3)
        nrm = pts[rp[sel, 1]] - pts[rp[sel, 0]]
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        e1 = np.where(np.abs(nrm[:, :1]) > 0.9,
                      np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        e1 = e1 - nrm * np.sum(e1 * nrm, axis=1, keepdims=True)
        e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
        e2 = np.cross(nrm, e1)
        Q = P - P.mean(1)[:, None, :]
        x = np.einsum('nkj,nj->nk', Q, e1)
        y = np.einsum('nkj,nj->nk', Q, e2)
        order = np.argsort(np.arctan2(y, x), axis=1)
        x = np.take_along_axis(x, order, axis=1)
        y = np.take_along_axis(y, order, axis=1)
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))
        np.add.at(RA, (ri[sel], rj[sel]), area)
        np.add.at(RA, (rj[sel], ri[sel]), area)
    return RA


# ---------------------------------------------------------------- SSE super-node graph
def _ss3(code):
    if code in ('H', 'G', 'I'):
        return 'H'
    if code in ('E', 'B', 'b'):
        return 'E'
    return 'C'


def make_units(ss, res_ints):
    """maximal runs of one SS class AND contiguous residue numbering."""
    units = []; N = len(ss); i = 0
    while i < N:
        j = i
        while (j + 1 < N and ss[j + 1] == ss[i]
               and res_ints[j + 1] is not None and res_ints[j] is not None
               and res_ints[j + 1] - res_ints[j] == 1):
            j += 1
        units.append((i, j)); i = j + 1
    return units


def build_W(units, A):
    U = len(units)
    memb = np.empty(A.shape[0], dtype=int)
    for u, (a, b) in enumerate(units):
        memb[a:b + 1] = u
    W = np.zeros((U, U))
    for u, (a, b) in enumerate(units):
        for v, (c, d) in enumerate(units):
            if v < u:
                continue
            blk = A[a:b + 1, c:d + 1].sum()
            if u == v:
                W[u, u] = blk / 2.0
            else:
                W[u, v] = W[v, u] = blk
    return W, memb


# ---------------------------------------------------------------- self-containment ratio
def _inter(W, m1, m2):
    return float(sum(W[a, b] for a in m1 for b in m2))


def _ibb(W, mem):
    s = sum(W[a, b] for i, a in enumerate(mem) for b in mem[i + 1:])
    return s if s > 0 else sum(W[a, a] for a in mem)


def ratio(W, Wh, A, B, nA, nB, beta=BETA):
    """self-containment of two unit-groups: (hydrophobic interface / mu^beta) / mean internal density."""
    ic_geo = _inter(W, A, B)
    if ic_geo <= 0:
        return None
    ic = _inter(Wh, A, B)
    denom = (nA * nB / (nA + nB)) ** beta
    ave = (_ibb(W, A) + _ibb(W, B) + ic_geo) / (nA + nB)
    return (ic / denom) / ave if ave > 0 else 1e9


def agglomerate(W, Wh, usz, beta=BETA):
    """full trajectory: repeatedly merge the most self-contained pair -> [(a, b, ratio)].

    Cluster-level quantities are kept incrementally instead of being recomputed from the member lists at
    every step. Merging A into B is exact bookkeeping -- interface areas add (inter(A|B, C) = inter(A,C) +
    inter(B,C)), and the internal cross-area of the union is cross_A + cross_B + inter(A,B) -- so only the
    row and column of the surviving cluster change and one merge costs O(U) instead of a full rescan.
    Pairs are scanned in the order of the surviving clusters and a tie goes to the first pair found.
    """
    U = W.shape[0]
    G = np.array(W, dtype=float); np.fill_diagonal(G, 0.0)      # inter-cluster geometric area
    H = np.array(Wh, dtype=float); np.fill_diagonal(H, 0.0)     # inter-cluster hydrophobic area
    cross = np.zeros(U)                                         # internal cross-unit area of each cluster
    diag = np.array([W[i, i] for i in range(U)], dtype=float)   # each cluster's own within-unit area
    size = np.array(usz, dtype=float)
    active = list(range(U))
    merges = []
    while len(active) > 1:
        bp, br = None, -1.0
        for i in range(len(active)):
            a = active[i]
            for j in range(i + 1, len(active)):
                b = active[j]
                ic_geo = G[a, b]
                if ic_geo <= 0:
                    continue
                na, nb = size[a], size[b]
                ave = ((cross[a] if cross[a] > 0 else diag[a]) +
                       (cross[b] if cross[b] > 0 else diag[b]) + ic_geo) / (na + nb)
                if ave <= 0:
                    r = 1e9
                else:
                    r = (H[a, b] / (na * nb / (na + nb)) ** beta) / ave
                if r > br:
                    br, bp = r, (a, b)
        if bp is None:
            break
        a, b = bp
        cross[a] += cross[b] + G[a, b]
        diag[a] += diag[b]
        size[a] += size[b]
        G[a, :] += G[b, :]; G[:, a] += G[:, b]; G[a, a] = 0.0
        H[a, :] += H[b, :]; H[:, a] += H[:, b]; H[a, a] = 0.0
        active.remove(b)
        merges.append((a, b, br))
    return merges


def partition_at(W, Wh, usz, merges, F=None, beta=BETA):
    """replay the trajectory down to F, then force every domain to reach MINDOM.

    F=None is resolved here, per backend (F_CUT_VORONOTA / F_CUT_LATTICE), at call time.

    CAVEAT: backend() reports which tessellation is AVAILABLE, not which one built W. If you assembled W
    yourself, pass F explicitly -- the two thresholds are on different ratio scales and are not
    interchangeable.
    """
    if F is None:
        F = F_CUT_VORONOTA if backend() == 'voronota' else F_CUT_LATTICE
    U = W.shape[0]
    members = {i: [i] for i in range(U)}; size = {i: float(usz[i]) for i in range(U)}
    for (a, b, r) in merges:
        if r < F:
            break
        members[a] += members[b]; size[a] += size[b]; del members[b]; del size[b]
    while len(members) > 1 and any(size[d] < MINDOM for d in members):
        ds = list(members); bp, br = None, -1
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = ds[i], ds[j]
                if size[a] >= MINDOM and size[b] >= MINDOM:
                    continue
                r = ratio(W, Wh, members[a], members[b], size[a], size[b], beta)
                if r is not None and r > br:
                    br, bp = r, (a, b)
        if bp is None:
            break
        a, b = bp
        members[a] += members[b]; size[a] += size[b]; del members[b]; del size[b]
    lab = np.empty(U, int)
    for c, d in enumerate(members):
        for u in members[d]:
            lab[u] = c
    return lab


# ---------------------------------------------------------------- segment cleanup
def _segments(lab):
    out = []; s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append([s, i - 1, int(lab[s])]); s = i
    return out


# ---------------------------------------------------------------- driver
def predict(ca, ss, res_labels, chain_id, F=None, post=True, verbose=False):
    """Domain label per residue.

    On exact Voronota areas the partition is chosen among the bottom-up stop points and the top-down best-first
    partitions by the two-sided harmonic-mean objective at t = bihc_hm.T_HM (F, if given, overrides t), then
    MINSEGMENT -> min2ss (bihc_post.postprocess_hm). The fallback graphs (lattice areas, CA contacts) cut the bottom-up
    trajectory at their own F instead (F_CUT_LATTICE / F_CUT_FALLBACK), with the same clean-up.
    post=False: the raw partition, with no post-processing at all.
    """
    global LAST_GRAPH
    LAST_GRAPH = None
    F_given = F
    ca = np.asarray(ca, float)
    N = len(ca)
    if N < LMIN:
        if verbose:
            print(f'  chain shorter than {LMIN} residues -> single domain')
        LAST_GRAPH = {'trivial': True, 'N': N}
        return np.zeros(N, int)

    res_ints = []
    for lab in res_labels:
        try:
            res_ints.append(int(''.join(c for c in str(lab) if c.isdigit() or c == '-')))
        except ValueError:
            res_ints.append(None)
    units = make_units([_ss3(s) for s in _align(ss, N)], res_ints)
    if len(units) < 2:
        if verbose:
            print('  fewer than two SSE units -> single domain')
        LAST_GRAPH = {'trivial': True, 'N': N}
        return np.zeros(N, int)

    got = _chain_atoms(chain_id, res_labels, ca_ref=ca)
    RA = None; g = None; coarse = False; mode = backend(); F_default = None
    if got is not None:
        xyz, rof, g, els = got
        per_res = len(xyz) / max(1, len(set(rof.tolist())))
        coarse = per_res < MIN_ATOMS_PER_RES
        if coarse:
            print(f'  WARNING: only {per_res:.1f} heavy atoms per residue -- this looks like a CA-only or '
                  f'coarse-grained structure, so the tessellation is not meaningful')
        else:
            if mode == 'voronota':
                # res_labels, not res_ints: the int coercion below drops insertion codes
                RA = voronota_areas(_STRUCT_PATH, str(chain_id).replace('_', '').split('MODEL')[0],
                                    res_labels)
                if RA is None:
                    print('  WARNING: the exact tessellation failed; falling back to the lattice '
                          'approximation of the contact areas')
                    mode = 'lattice'
            if RA is None:
                RA = contact_areas(xyz, rof, N)
    if RA is None:
        # a different contact graph: always say so, not only under -v
        why = ('no usable coordinates for this chain '
               '(the chain id or the residue numbering did not match the structure)' if got is None else
               'the structure is not all-atom' if coarse else 'the tessellation could not be built')
        print(f'  WARNING: {why}; falling back to the 8 A CA-contact graph instead of Voronota contact areas.')
        RA = (distance_matrix(ca, ca) < CONTACT_D).astype(float)
        np.fill_diagonal(RA, 0.0)
        F_default = F_CUT_FALLBACK
        if F is None:             # caller kept the default, which is on the scale of AREAS
            F = F_CUT_FALLBACK
            print(f'  using F={F} instead, the threshold of the contact-count graph')
    if g is None:
        print('  WARNING: no residue types available; the hydrophobic weighting is disabled')
        g = np.full(N, 0.5)

    if F_default is None:         # each backend has its own threshold
        F_default = F_CUT_VORONOTA if mode == 'voronota' else F_CUT_LATTICE
    if F is None:
        F = F_default
    if verbose:
        print(f'  contact areas: {mode}' + ('' if mode == 'voronota' and not coarse and got is not None else f'   F={F}'))

    H = 0.5 * (g[:, None] + g[None, :])
    W, memb = build_W(units, RA)
    Wh, _ = build_W(units, RA * H)
    usz = np.array([e - s + 1 for (s, e) in units], float)

    merges = [(int(a), int(b), float(r)) for a, b, r in agglomerate(W, Wh, usz)]
    exact = mode == 'voronota' and not coarse and got is not None
    LAST_GRAPH = {'trivial': False, 'N': N, 'W': W, 'Wh': Wh, 'usz': usz, 'units': [(int(a), int(b)) for a, b in units],
                  'memb': np.asarray(memb, int), 'merges': merges, 'ss': [str(x) for x in _align(ss, N)],
                  'exact': bool(exact), 'F_default': float(F_default), 'beta': float(BETA),
                  'graph_kind': ('voronota' if exact else
                                 'lattice-area' if mode == 'lattice' and not coarse and got is not None else 'CA-contact')}
    return predict_from_graph(LAST_GRAPH, t=F_given, post=post, verbose=verbose)


LAST_GRAPH = None   # the unit graph of the last predict() call (BiHC.py saves it; predict_from_graph() re-uses it)


def predict_from_graph(graph, t=None, post=True, verbose=False):
    """Domain label per residue from a saved unit graph (predict() ends here too, so both give identical labels).

    Everything up to the selection -- STRIDE, Voronota areas, SSE units, the bottom-up merge trajectory, the top-down
    tree, every candidate partition and its F values (graph['scored'], computed on the first call and then kept) -- does
    not depend on t; a new t only re-scores the margins and re-runs the clean-up. t: HM threshold (None -> bihc_hm.T_HM);
    on a fallback graph, the F at which the merge trajectory is cut (None -> that graph's default F).
    """
    try:
        from src import bihc_post, bihc_hm
    except ImportError:
        import bihc_post, bihc_hm
    if graph.get('trivial'):
        return np.zeros(int(graph['N']), int)
    W, Wh, usz = graph['W'], graph['Wh'], graph['usz']
    units, memb, merges = graph['units'], graph['memb'], graph['merges']
    if abs(float(graph.get('beta', BETA)) - BETA) > 1e-12:      # saved by a BiHC with another beta: redo the trajectory
        merges = [(int(a), int(b), float(r)) for a, b, r in agglomerate(W, Wh, usz)]
        graph = dict(graph, merges=merges, beta=BETA, scored=None)
    if graph['exact']:
        t = bihc_hm.T_HM if t is None else t
        if verbose:
            print(f'  partition selection: two-sided harmonic-mean objective, t={t}')
        if graph.get('scored') is None:                          # candidates and their F do not depend on t:
            graph['scored'] = bihc_hm.scored_candidates(W, Wh, usz, merges)   # computed once, kept in the graph
        ul = bihc_hm.select_scored(graph['scored'], W.shape[0], t)
    else:
        F = graph['F_default'] if t is None else t
        print(f"  NOTE: the HM selection works on exact Voronota areas; the {graph['graph_kind']} fallback graph "
              f"is cut at its own threshold F={F} instead")
        ul = bihc_post.raw_partition(merges, W.shape[0], F)
    if not post:
        return np.unique(np.asarray(ul)[memb], return_inverse=True)[1]
    lab = bihc_post.postprocess_hm(ul, units, memb, W, Wh, usz, graph['ss'], ratio)
    return np.asarray(lab, int)


def _align(ss, N, fill='C'):
    ss = list(ss) if ss is not None else []
    if len(ss) >= N:
        return ss[:N]
    return ss + [fill] * (N - len(ss))
