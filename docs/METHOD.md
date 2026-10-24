# BiHC: method and implementation notes

## How BiHC works

1. **SSE super-nodes** — STRIDE assigns a secondary-structure class to every residue; maximal runs of
   one class with contiguous numbering become the nodes of the graph. Working at element level rather
   than residue level keeps cuts out of the middle of a helix or strand.
2. **Contact areas from a constrained Voronoi tessellation of atomic balls** (additively weighted) — the chain's van-der-Waals balls are
   tessellated and the cells are clipped by a 1.4 Å water probe (via Voronota 1.29). Two residues
   are in contact only when their cells share a face, and the weight of that contact is the **face area in
   Å²**, not a count: a pair screened by a third residue contributes nothing, a long side-chain contact
   beyond 8 Å is not missed, atomic radii enter properly, and there is no distance cutoff, no lattice and
   no spacing parameter to choose. Each contact is additionally weighted by the mean Kyte–Doolittle
   nonpolarity of the two residues, because a domain core is held together by buried hydrophobic contacts.
3. **Bottom-up agglomeration on a self-containment ratio** — repeatedly merge the pair of groups that
   is most self-contained under

   ```
   F(A,B) = (ic_h / mu^beta) / ave        mu  = nA*nB / (nA+nB)        beta = 0.86
   ave    = (intra_A + intra_B + ic) / (nA + nB)
   ```

   `ic_h` is the nonpolarity-weighted interface area between the two groups, `mu` the reduced mass of
   their sizes, and `ave` a per-residue internal-support term. `intra_A` is the area between DISTINCT
   units of A when A has any; a group with no such area (e.g. a single SSE unit) contributes its
   within-unit area instead. So `ave` is not the internal contact density the merged group reports
   afterwards: when a single unit joins a cluster, its within-unit area enters the score of that merge and
   is then dropped from the union's bookkeeping. For two single-residue SSE units both terms vanish, `ave`
   reduces to `ic/(nA+nB)`, the area cancels out of `F` and the pair is scored on hydropathy alone.
4. **Choose the partition (harmonic-mean selection)** — instead of stopping the trajectory at the first
   merge with `F < t`, whole partitions are compared. Candidates are every stop point of the bottom-up trajectory
   and every top-down best-first partition, from one domain until no part can be split further (the part whose
   weakest single or double cut on the SSE order has the lowest `F` is split next; every part >= 40 residues, so
   inserted domains are reachable). A partition is
   admissible only if every cut interface is weak, `F_ij < t`, and every domain of >= 80 residues has no cut weaker
   than `t`, `Fcut_d > t`. Among admissible partitions BiHC returns the one with the highest harmonic mean of the
   margins `t - F_ij` and `Fcut_d - t` (ties -> fewer domains), i.e. the partition whose least certain decision is
   furthest from `t` (default `t = 0.365`, see below).
5. **Clean** — every sequence segment shorter than 40 residues, then every domain with fewer than 2
   secondary-structure elements (a lone helix or strand), is merged into the sequence-adjacent domain with the
   highest `F`.

Chains shorter than 80 residues are reported as a single domain (no two parts of >= 40 residues exist).

The lattice and CA-contact fallbacks below do not use the harmonic-mean selection: they cut the bottom-up trajectory
at their own `F` (0.40 and 0.30).

**Contact-area backend.** The exact tessellation needs Voronota 1.29 (MIT licence), which the user installs
(README, "Installation"). Without it the command line stops, unless `--allow-lattice` is given: then the code uses a
pure-Python Voronoi whose surface cells are bounded by a solvent lattice, prints a warning, and switches to that
backend's own threshold `F = 0.40`. The lattice areas are an approximation of the exact ones. On exact areas
`-t` / `--threshold` sets `t`; on the fallback graphs it sets the cut `F`.

**Modified residues (HETATM).** Residues written as HETATM, such as selenomethionine (MSE), are left out of the graph, as they are out of the Voronota tessellation and of STRIDE, which read ATOM records only. They are put back into the domains in every output (positions, JSON, `.tsv`, PyMOL script, structure files): a residue between two parts of one domain joins that domain; between two domains a single residue joins the larger domain and a longer run is split half / half. So a domain is never cut by the one- or two-residue gaps they leave. Water, ions and ligands are not part of the chain. Positions are written as compact ranges: two parts of a domain separated only by residues without coordinates (missing residues) are one segment (`2-226`, not `2-17+20-48+...`), while a residue of the structure in between (in another domain or unassigned) still separates them (each bridged gap is reported with a warning and in the `missing_res` column of `NAME_domains.tsv`); the segments of a domain are listed in residue-number order.

**Input requirements.** The method needs an **all-atom** structure: the tessellation is what carries the
signal. If the input is CA-only or coarse-grained (fewer than 3 heavy atoms per residue), if the chain id or
residue numbering cannot be matched, or if the tessellation cannot be built, the code prints a warning and
falls back to an 8 Å CA-contact graph cut at `F = 0.30`. A tessellation of CA atoms alone is not meaningful, so
coarse structures are detected and diverted rather than tessellated. For NMR ensembles every model is read and the one
matching the input coordinates is used.

**Non-standard residues.** Every amino-acid residue of the polymer chain is used, whatever its name: besides the
20 standard residues and the mapped modified ones (MSE, SEC, ...), any residue with backbone N, CA, C that is
peptide-bonded (C–N ≤ 2.0 Å) to its neighbour in the chain — e.g. `UNK` (unknown amino acid, usually modelled as
backbone + CB) or an unmapped modified residue — is kept, with the default hydropathy (Kyte–Doolittle −0.4).
Ligands that merely carry an amino-acid moiety (e.g. SAM / SAH cofactors) are not bonded into the chain and are
left out.

BiHC is a protein method: nucleic-acid chains in the input are skipped.

## The threshold `t`

The threshold `t` (`-t` / `--threshold`, default 0.365) is the one operating parameter; the other constants (e.g.
`beta = 0.86` and the 40-residue minimum segment) are fixed parts of the method. Higher values split more, lower values
merge more. `t` is on the scale of `F`, so it belongs to this score as defined here: changing the radii, the probe or
the hydropathy weighting changes the scale of `F`.

## Saved graphs

Everything before the partition selection -- secondary structure, Voronota contact areas, SSE units and the
bottom-up merge trajectory -- does not depend on `t`, so each run with an output directory also saves it as
`<name>_graph.npz` (one file per structure, all processed chains; `--no-graph` turns this off), together with every
candidate partition of both directions and its F values, which do not depend on `t` either. Given to `-i`, it is read back
and only re-scores the margins and re-runs the clean-up, with any `-t`, `--no-post`, `-c`; the result is identical to a full
run with the same options. BiHC tells a graph from a structure by its `.npz` ending; a directory holding `*_graph.npz` files
is read as graphs, even when it also holds structure files (an `-o` directory does). A graph input cannot write domain structure files
(`--output-pdb`, which also writes the PyMOL script), which need the original coordinates; the JSON and `.tsv` are written as usual. The file holds plain arrays
only (NumPy `.npz`, read with `allow_pickle=False`), so opening one received from someone else cannot run code.
