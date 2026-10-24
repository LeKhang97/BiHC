# BiHC

BiHC finds the structural domains of a protein chain from its 3D structure (PDB or mmCIF). It is deterministic and
**training-free**: it uses no learned model, no sequence homology and no evolutionary information, only the physics of
the chain.

BiHC works in three steps:
1. **Units.** Secondary-structure elements, assigned by STRIDE, are the units of the chain.
2. **Contact strength.** Two units are linked by their contact area in a Voronoi tessellation of the atoms (Voronota,
   1.4 Å probe), weighted by hydrophobicity.
3. **Partition.** Units are grouped bottom-up and split top-down using a self-containment score. BiHC then returns the
   partition whose least certain decision lies furthest from a single threshold `t` (default 0.365).

[`docs/METHOD.md`](docs/METHOD.md) describes the method and its implementation in detail.

## Installation

BiHC needs Python ≥ 3.8 and two external programs. They are **not included** in this repository, because their
licences differ from BiHC's, so install them yourself:

| Program | Used for | Licence |
|---|---|---|
| [STRIDE](https://github.com/MDAnalysis/stride) | secondary structure | free for academic use; commercial use needs a licence from its authors |
| [Voronota 1.29](https://github.com/kliment-olechnovic/voronota) | contact areas | MIT |

**1. BiHC and its Python packages**

```bash
git clone https://github.com/LeKhang97/BiHC.git && cd BiHC
pip install -r requirements.txt          # numpy, scipy, biopython
```

**2. Voronota 1.29.** Install it with conda (`conda install -c conda-forge -c bioconda voronota`, Linux and macOS) or build it from
the release source with a C++ compiler (about 2 minutes):

```bash
curl -LO https://github.com/kliment-olechnovic/voronota/releases/download/v1.29.4602/voronota_1.29.4602.tar.gz
tar xzf voronota_1.29.4602.tar.gz && cd voronota_1.29.4602
g++ -O3 -std=c++11 -o voronota $(find ./src/ -name '*.cpp')
```

**3. STRIDE.** Build it from source with a C compiler:

```bash
git clone https://github.com/MDAnalysis/stride && make -C stride/src      # executable: stride/src/stride
```

This copy holds the same source files as the original STRIDE distribution (Frishman & Argos). Note that the bioconda
package called `stride` is a different program, a genome assembler.

**4. Tell BiHC where they are.** BiHC looks for each program in this order:

1. `--stride PATH` / `--voronota PATH` on the command line (the executable or its directory), or the environment
   variables `BIHC_STRIDE` / `BIHC_VORONOTA`;
2. an executable called `stride` / `voronota` in `src/`;
3. the `PATH`.

So putting both on the `PATH` (or copying them into `src/`) is enough. If one is missing, BiHC stops and says which.
It never silently runs a different method: the pure-Python approximation of the contact areas is used only when you
ask for it with `--allow-lattice`.

```bash
python BiHC.py -i examples/1a45.pdb      # quick check; the header names the STRIDE and Voronota it found
```

BiHC uses Voronota 1.29; other Voronota versions may give other contact areas (BiHC prints a warning).

### Docker

```bash
docker build -t bihc .
docker run --rm -v "$PWD":/data bihc -i examples/1a45.pdb -o out
```

The image downloads and compiles STRIDE and Voronota 1.29 while it is built (step 2 and 3 above), so nothing has to
be installed beside Docker. The current directory is mounted as `/data`, so input and output paths are relative to it.

## Quick start

```bash
python BiHC.py -i examples/1a45.pdb
```

```
Number of residues of cluster 1: 81
Cluster 1 positions:
1-81

Number of residues of cluster 2: 92
Cluster 2 positions:
82-174

  WARNING: cluster 2 spans residue 85 without coordinates in the structure, bridged in its ranges
```

More examples:

```bash
python BiHC.py -i structure.pdb -c A                 # one chain (default: every chain separately)
python BiHC.py -i structures/ -o out                # every .pdb/.cif/.ent/.mmcif in a directory
python BiHC.py -i structure.pdb -o out              # write the JSON, the .tsv and the graph
python BiHC.py -i structure.pdb -o out --output-pdb  # also the domain structure file and a PyMOL script
python BiHC.py -i structure.pdb -c A -t 0.40         # another operating point (higher t splits more)
python BiHC.py -i out/structure_graph.npz -t 0.40 -o out2  # re-run from a saved graph (fast, no STRIDE/Voronota)
```

## Output

Domains are given in author residue numbering, including insertion codes. Each domain is printed as its list of
segments, so a discontinuous domain appears as e.g. `2-13, 102-335, 381-463`.
- Residues without coordinates do not break a segment: when they lie between two residues of the same domain, the
  segment runs across them (`82-174` although 85 has no coordinates in 1a45), in every output (standard output, the
  `.tsv`, the PyMOL script). Each bridged gap is reported with a warning, and in the
  `missing_res` column of the `.tsv`. Residues without coordinates between two domains belong to neither.
- Modified residues such as MSE are included in the domain that surrounds them.

| option | file | content |
|---|---|---|
| (always) | standard output | number of residues and ranges of every domain, per chain |
| `-o DIR` | `DIR/NAME.json` | per chain: CA coordinates (`data`), domain index per residue (`cluster`), residue labels (`res_labels`), missing residues |
| `-o DIR` | `DIR/NAME_domains.tsv` | tab-separated, one row per domain: `structure`, `chain`, `domain`, `n_residues`, `res_positions` (residue ranges in author numbering; the segments of a discontinuous domain joined by `,`), `missing_res` (residues without coordinates inside the domain, bridged in `res_positions`; `-` if none) |
| `-o DIR` | `DIR/NAME_graph.npz` | the saved graph, for re-running with another `t` by giving it to `-i` (`--no-graph` turns it off) |
| `--output-pdb` | `DIR/NAME.pdb` (or `.cif`) | the processed chain, in the input's format and residue order (water and ligands left out) |
| `--output-pdb` | `DIR/NAME_pymolcmd.pml` | PyMOL script that loads that file (absolute path) and colours its domains: `pymol DIR/NAME_pymolcmd.pml` |

`NAME` is the input file name by default (`-j` sets another). When several chains are processed, the JSON and the `.tsv`
stay one file each (both are keyed by chain), while `--output-pdb` writes one structure file and one PyMOL script per
chain, `NAME_<chain>.pdb` and `NAME_<chain>_pymolcmd.pml`. Any output option alone writes to the current directory.

## Options

| short | long | meaning | default |
|---|---|---|---|
| `-i` | `--input` | structure file (PDB/mmCIF), saved graph (`*_graph.npz`; re-runs only the partition selection), or a directory of either (graphs first) | required |
| `-c` | `--chain` | chain id, or `longest`; omitted: every chain separately | all chains |
| `-t` | `--threshold` | operating threshold; higher splits more, lower merges more | `0.365` |
| `-m` | `--min-length` | chains shorter than this are skipped | `30` |
| `-o` | `--outpath` | output directory | — |
| `-j` | `--json` | base name of the output files | input name |
| | `--output-pdb` | also write each processed chain as a structure file (`.pdb`, `.cif` for mmCIF input) with a PyMOL script that colours its domains | off |
| | `--no-post` | skip the post-processing (minimum segment length 40, at least 2 SSEs per domain) | off |
| | `--no-graph` | do not save the graph file | saved |
| | `--stride` | STRIDE executable or its directory | `$BIHC_STRIDE`, `src/`, `PATH` |
| | `--voronota` | Voronota 1.29 executable or its directory | `$BIHC_VORONOTA`, `src/`, `PATH` |
| | `--allow-lattice` | run without Voronota, on the lattice approximation of the contact areas | off |
| `-v` | `--verbose` | verbose output | off |

Notes:
- **Short chains.** Chains shorter than 80 residues are reported as one domain.
- **All-atom input.** BiHC needs an all-atom structure. For CA-only or coarse-grained input, it warns and falls back to
  a CA-contact graph.
- **Proteins only.** Nucleic-acid chains are skipped.

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Citation

If you use BiHC, please cite:

> *[BiHC: Protein domain segmentation by bidirectional hierarchical clustering using Voronoi tessellation and physical self-containment](https://doi.org/10.64898/2026.10.18.759833)*

BiHC runs two third-party programs; please cite them as well:
- STRIDE: Frishman D, Argos P. Knowledge-based protein secondary structure assignment. *Proteins* 1995; 23: 566–579.
- Voronota: Olechnovič K, Venclovas Č. Voronota: a fast and reliable tool for computing the vertices of the Voronoi
  diagram of atomic balls. *J Comput Chem* 2014; 35: 672–681.

## License

- **BiHC source code:** MIT licence ([`LICENSE`](LICENSE)).
- **STRIDE** (not included): free for academic use only; commercial use requires a licence from its authors.
- **Voronota** (not included): MIT licence. `src/voronota_radii` is the atomic radii file of Voronota 1.29, included
  unchanged under that licence (see [`LICENSE`](LICENSE)).
