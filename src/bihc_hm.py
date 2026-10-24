"""
BiHC partition selection by the two-sided harmonic-mean objective (HM).

The SSE-unit graph (W, Wh, usz) and the bottom-up merge trajectory are those of bihc_core. Instead of stopping the
trajectory at the first merge with F < t, whole partitions are compared:

  candidates   (a) every stop point of the bottom-up trajectory (all units apart ... one domain)
               (b) top-down best-first partitions from K = 1 until no part has an admissible cut (the finest split the
                   size rule allows, the top-down counterpart of the first bottom-up stop point): the part whose best
                   cut has the lowest F is split next;
                   cuts are single (S[:i] | S[i:]) or double (S[i:j] | S[:i] + S[j:]) on the unit order, every part
                   >= 40 residues (a double cut keeps S[:i], S[i:j], S[j:] each >= 40), so domains inserted into
                   other domains are reachable
  margins      for every pair of contacting domains     t - F_ij          (the interface must be weak)
               for every domain of >= 80 residues        Fcut_d - t        (its weakest single/double cut must be
                                                                            stronger than t)
               F is bihc_core.ratio evaluated on the two groups; Fcut_d is the lowest F over the admissible cuts of d
  objective    a partition is admissible only if every margin is > 0; its score is the harmonic mean of its
               margins (0 when it has none); the highest score wins, ties -> fewer domains; if no candidate is
               admissible, the candidate with the fewest domains
  threshold    t = T_HM; a higher t splits more
Chains with fewer than 80 residues are one domain (no two parts of >= 40 residues exist).
Post-processing (bihc_post.postprocess_hm): MINSEGMENT 40 -> min2ss.

Reading: the harmonic mean is dominated by the smallest margin, so the selected partition is the one whose least
certain decision (the strongest interface it cuts, or the weakest cut it leaves inside a domain) lies furthest from t.
"""
import numpy as np

T_HM = 0.365        # margin threshold t (higher splits more)
MINFRAG = 40        # every part of a cut
MIN_SPLIT = 2 * MINFRAG


def _beta():
    try:
        from src import bihc_core
    except ImportError:                                  # pragma: no cover
        import bihc_core
    return bihc_core.BETA


class _Block:
    """2D prefix sums of W and Wh in the unit order S, so every interval block sum is O(1)"""
    def __init__(self, W, Wh, usz, S):
        ix = np.ix_(S, S)
        self.PW = np.zeros((len(S) + 1, len(S) + 1)); self.PW[1:, 1:] = W[ix].cumsum(0).cumsum(1)
        self.PH = np.zeros((len(S) + 1, len(S) + 1)); self.PH[1:, 1:] = Wh[ix].cumsum(0).cumsum(1)
        self.D = np.concatenate([[0], np.cumsum(np.diag(W[ix]))]); self.N = np.concatenate([[0], np.cumsum(usz[S])])

    def blk(self, P_, r0, r1, c0, c1):
        return P_[r1, c1] - P_[r0, c1] - P_[r1, c0] + P_[r0, c0]

    def cross(self, a, b):                               # area between distinct units inside [a, b)
        return (self.blk(self.PW, a, b, a, b) - (self.D[b] - self.D[a])) / 2

    def diag(self, a, b):                                # within-unit area inside [a, b)
        return self.D[b] - self.D[a]


def _F_parts(ig, ih, ibbA, ibbB, nA, nB, beta):
    if ig <= 0:
        return None
    ave = (ibbA + ibbB + ig) / (nA + nB)
    return 1e9 if ave <= 0 else float((ih / (nA * nB / (nA + nB)) ** beta) / ave)


def best_cut(W, Wh, usz, S, beta=None):
    """lowest-F admissible single or double cut of the ordered unit list S -> (F, partA, partB) or None"""
    beta = _beta() if beta is None else beta
    n = len(S)
    if n < 2:
        return None
    b = _Block(W, Wh, usz, S); N = b.N; best = None
    ibb = lambda cr, dg: cr if cr > 0 else dg
    for i in range(1, n):
        if N[i] >= MINFRAG and N[n] - N[i] >= MINFRAG:
            f = _F_parts(b.blk(b.PW, 0, i, i, n), b.blk(b.PH, 0, i, i, n), ibb(b.cross(0, i), b.diag(0, i)),
                         ibb(b.cross(i, n), b.diag(i, n)), N[i], N[n] - N[i], beta)
            if f is not None and (best is None or f < best[0]):
                best = (f, S[:i], S[i:])
    for i in range(1, n):
        if N[i] < MINFRAG:
            continue
        for j in range(i + 1, n):
            if N[j] - N[i] < MINFRAG:
                continue
            if N[n] - N[j] < MINFRAG:
                break
            ig = b.blk(b.PW, i, j, 0, i) + b.blk(b.PW, i, j, j, n)
            ih = b.blk(b.PH, i, j, 0, i) + b.blk(b.PH, i, j, j, n)
            cm = b.cross(i, j); co = b.cross(0, i) + b.cross(j, n) + b.blk(b.PW, 0, i, j, n)
            f = _F_parts(ig, ih, ibb(cm, b.diag(i, j)), ibb(co, b.diag(0, i) + b.diag(j, n)),
                         N[j] - N[i], N[i] + N[n] - N[j], beta)
            if f is not None and (best is None or f < best[0]):
                best = (f, S[i:j], S[:i] + S[j:])
    return best


def build_tree(W, Wh, usz):
    """recursive lowest-F bisection of the whole chain: nested {'units', 'F' (best cut F or None), 'kids'}"""
    def rec(S):
        c = best_cut(W, Wh, usz, S)
        if c is None:
            return {'units': S, 'F': None, 'kids': []}
        return {'units': S, 'F': float(min(c[0], 1e9)), 'kids': [rec(c[1]), rec(c[2])]}
    return rec(list(range(W.shape[0])))


def candidates(W, Wh, usz, merges):
    """bottom-up stop points (finest first), then top-down best-first partitions from one domain until no part has an
    admissible cut; unit labels."""
    U = W.shape[0]; out = []
    members = {i: [i] for i in range(U)}
    for j in range(len(merges) + 1):
        ul = np.empty(U, int)
        for c, d in enumerate(members):
            ul[members[d]] = c
        out.append(ul)
        if j < len(merges):
            a, b = int(merges[j][0]), int(merges[j][1]); members[a] += members.pop(b)
    leaves = [build_tree(W, Wh, usz)]
    while True:
        ul = np.empty(U, int)
        for c, nd in enumerate(leaves):
            ul[nd['units']] = c
        out.append(ul)
        split = [i for i, nd in enumerate(leaves) if nd['F'] is not None and nd['kids']]
        if not split:
            break
        i = min(split, key=lambda i: leaves[i]['F']); nd = leaves.pop(i); leaves += nd['kids']
    return out


def pair_F(W, Wh, usz, ul, beta=None):
    """F (bihc_core.ratio) for every pair of contacting groups of the unit labelling ul"""
    beta = _beta() if beta is None else beta
    g = ul.max() + 1
    G = np.zeros((len(ul), g)); G[np.arange(len(ul)), ul] = 1
    IG = G.T @ W @ G; IH = G.T @ Wh @ G; n = G.T @ usz
    du = G.T @ np.diag(W)
    ibb = (np.diag(IG) - du) / 2
    ibb = np.where(ibb > 0, ibb, du)
    iu = np.triu_indices(g, 1); ok = IG[iu] > 0
    a, b = iu[0][ok], iu[1][ok]
    mu = n[a] * n[b] / (n[a] + n[b])
    ave = (ibb[a] + ibb[b] + IG[a, b]) / (n[a] + n[b])
    F = np.where(ave > 0, (IH[a, b] / mu ** beta) / np.where(ave > 0, ave, 1), 1e9)
    return np.minimum(F, 1e3)


def scored_candidates(W, Wh, usz, merges):
    """the t-independent part of the selection: every candidate (bottom-up stop points, then top-down partitions) with
    the F of each pair of contacting domains (pair_F order) and the weakest-cut F of each domain that has a cut (label
    order). [] for a chain shorter than MIN_SPLIT. Changing t only needs select_scored() on this list."""
    if float(np.sum(usz)) < MIN_SPLIT:
        return []
    memo, out = {}, []
    for ul in candidates(W, Wh, usz, merges):
        Fp = pair_F(W, Wh, usz, ul); cuts = []
        for c in np.unique(ul):
            gr = tuple(np.where(ul == c)[0].tolist())
            if gr not in memo:
                bc = best_cut(W, Wh, usz, list(gr)) if usz[list(gr)].sum() >= MIN_SPLIT else None
                memo[gr] = None if bc is None else float(min(bc[0], 1e3))
            if memo[gr] is not None:
                cuts.append(memo[gr])
        out.append((ul, Fp, cuts))
    return out


def select_scored(scored, n_units, t=T_HM):
    """unit labels of the admissible candidate with the highest harmonic mean of margins (t - F for every pair of
    contacting domains, weakest-cut F - t for every domain with a cut); none admissible -> the candidate with the fewest
    domains; ties -> fewer domains"""
    if not scored:
        return np.zeros(n_units, int)
    best, bv = None, None
    for ul, Fp, cuts in scored:
        terms = list(t - np.asarray(Fp, float)) + [c - t for c in cuts]
        if terms and not all(x > 0 for x in terms):
            continue
        v = float(len(terms) / np.sum(1.0 / np.asarray(terms))) if terms else 0.0
        if bv is None or v > bv or (v == bv and ul.max() < best.max()):
            best, bv = ul, v
    if best is None:
        best = min((s_[0] for s_ in scored), key=lambda u: u.max())
    return np.unique(best, return_inverse=True)[1]


def select(W, Wh, usz, merges, t=T_HM):
    """unit labels of the admissible candidate with the highest harmonic mean of margins"""
    return select_scored(scored_candidates(W, Wh, usz, merges), W.shape[0], t)
