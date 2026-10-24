"""
BiHC post-processing: the partition chosen by the harmonic-mean objective (bihc_hm.py) is cleaned by two rules, in this
order. Every rule merges the offending piece into a SEQUENCE-ADJACENT domain (one that owns the segment immediately before
or after one of its segments), choosing the adjacent domain with the highest self-containment ratio F against it; only
when no adjacent domain exists (or none is in contact) is the nearest domain along the sequence used.

  1. MINSEGMENT  every sequence segment shorter than MINSEG (= 40) residues, shortest first, recomputed after each
                 merge, until every segment has >= MINSEG residues or one domain is left. A domain made of a single
                 short segment disappears this way.
  2. min2ss      a domain with fewer than 2 secondary-structure components (maximal runs of STRIDE H, G or E) is
                 not a domain (a lone helix or strand, however long); smallest first.
"""
import numpy as np

MINSEG = 40
SS_COMPONENT_CODES = set('HGE')


# ---------------------------------------------------------------- helpers
def raw_partition(merges, n_units, F):
    """unit labels after replaying the merge trajectory down to F (no size rule)"""
    members = {i: [i] for i in range(n_units)}
    for a, b, r in merges:
        if r < F:
            break
        members[a] += members.pop(b)
    lab = np.empty(n_units, int)
    for c, d in enumerate(members):
        lab[members[d]] = c
    return lab


def _segments(lab):
    out = []; s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append((s, i - 1, int(lab[s]))); s = i
    return out


class _Graph:
    """F between arbitrary residue groups, via the SSE-unit graph (groups are unions of whole units here)"""
    def __init__(self, W, Wh, usz, units, ratio):
        self.W, self.Wh, self.usz, self.ratio = W, Wh, usz, ratio
        self.unit_of = np.empty(int(units[-1][1]) + 1, int)
        for u, (a, b) in enumerate(units):
            self.unit_of[a:b + 1] = u

    def units(self, idx):
        return sorted(set(self.unit_of[np.asarray(idx)].tolist()))

    def F(self, A_idx, B_idx):
        A, B = self.units(A_idx), self.units(B_idx)
        if not A or not B:
            return None
        return self.ratio(self.W, self.Wh, A, B, float(self.usz[A].sum()), float(self.usz[B].sum()))


def _merge_target(lab, piece_idx, candidates, graph):
    best, tgt = None, None
    for t in candidates:
        f = graph.F(piece_idx, np.where(lab == t)[0])
        if f is not None and (best is None or f > best):
            best, tgt = f, t
    if tgt is None:                                  # no contact with any candidate: nearest along the sequence
        dist = {t: np.min(np.abs(np.where(lab == t)[0][:, None] - np.asarray(piece_idx)[None, :])) for t in candidates}
        tgt = min(dist, key=dist.get)
    return tgt


def _adjacent_domains(lab, d):
    segs = _segments(lab); nb = []
    for i, s in enumerate(segs):
        if s[2] == d:
            for j in (i - 1, i + 1):
                if 0 <= j < len(segs) and segs[j][2] != d:
                    nb.append(segs[j][2])
    return list(dict.fromkeys(nb))


def n_ss_components(idx, ss):
    n = 0; prev = None; pi = None
    for i in np.sort(idx):
        c = ss[i] if ss[i] in SS_COMPONENT_CODES else None
        if c is not None and (c != prev or pi is None or i != pi + 1):
            n += 1
        prev, pi = c, i
    return n


# ---------------------------------------------------------------- the two rules
def minsegment(lab, graph, minseg=MINSEG):
    lab = np.unique(np.asarray(lab, int), return_inverse=True)[1]
    for _ in range(10 * len(lab)):
        if len(np.unique(lab)) < 2:
            break
        segs = _segments(lab)
        short = [(s[1] - s[0] + 1, i) for i, s in enumerate(segs) if s[1] - s[0] + 1 < minseg]
        if not short:
            break
        _, i = min(short)
        a, b, d = segs[i]; idx = np.arange(a, b + 1)
        cand = list(dict.fromkeys(segs[j][2] for j in (i - 1, i + 1) if 0 <= j < len(segs) and segs[j][2] != d))
        cand = cand or [t for t in np.unique(lab) if t != d]
        lab[idx] = _merge_target(lab, idx, cand, graph)
    return np.unique(lab, return_inverse=True)[1]


def dissolve(lab, graph, bad):
    """repeatedly merge the worst domain flagged by bad(residue_idx) -> score or None into an adjacent domain"""
    lab = np.unique(np.asarray(lab, int), return_inverse=True)[1]
    for _ in range(len(lab)):
        doms = np.unique(lab)
        if len(doms) < 2:
            break
        flagged = [(s, d) for d in doms for s in [bad(np.where(lab == d)[0])] if s is not None]
        if not flagged:
            break
        _, d = max(flagged)
        idx = np.where(lab == d)[0]
        cand = _adjacent_domains(lab, d) or [t for t in doms if t != d]
        lab[idx] = _merge_target(lab, idx, cand, graph)
    return np.unique(lab, return_inverse=True)[1]


def postprocess_hm(unit_labels, units, memb, W, Wh, usz, ss, ratio):
    """unit partition -> MINSEGMENT -> min2ss. Residue labels."""
    graph = _Graph(W, Wh, usz, units, ratio)
    lab = np.asarray(unit_labels, int)[memb]
    lab = minsegment(lab, graph)
    return dissolve(lab, graph, lambda idx: (-len(idx) if n_ss_components(idx, ss) < 2 else None))
