import argparse
import os
import sys
from pathlib import Path
from src.Functions import *


def main_parser():
    parser = argparse.ArgumentParser(
        description="BiHC: detect protein domains from a given 3D structure (PDB/mmCIF). Protein chains only."
    )

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='verbose mode.')

    parser.add_argument('-i', '--input', required=True,
                        help='Input: a structure file (PDB or mmCIF), a saved graph (<name>_graph.npz, written with the '
                             'outputs of an earlier run; re-runs only the partition selection and clean-up, e.g. with a '
                             'new -t, without STRIDE / Voronota, and cannot be combined with --output-pdb), or a directory of either (a '
                             'directory holding graphs is read as graphs).')

    parser.add_argument('-t', '--threshold',
                        type=float, default=None,
                        help='Threshold t of the harmonic-mean partition selection (default 0.365). Higher splits more. '
                             'On the fallback graphs (no exact Voronota areas) it is the F at which the bottom-up merge '
                             'is cut instead.')

    parser.add_argument('-m', '--min-length', dest='min_length',
                        default=30,
                        type=int,
                        help='Minimum chain length: chains shorter than this are skipped (default 30).')

    parser.add_argument('-o', '--outpath',
                        nargs='?',
                        const='.',
                        default=None,
                        type=str,
                        help="Output directory: <name>.json, <name>_domains.tsv and <name>_graph.npz (plus the --output-pdb files). Defaults to the current directory when an output is requested.")

    parser.add_argument('-c', '--chain',
                        type=str,
                        nargs='?',
                        const=None,
                        default=None,
                        help="Chain to process: a specific chain id, or 'longest'. "
                             "If omitted, every chain is clustered independently.")

    parser.add_argument('-j', '--json',
                        type=str,
                        nargs='?',
                        const=False,
                        default=None,
                        help='Base name of the output files (<name>.json, <name>_domains.tsv, ...). Defaults to the input file name.')

    parser.add_argument('--output-pdb', action='store_true',
                        help='Also write each processed chain as a structure file (.pdb, or .cif for mmCIF input) and a '
                             'PyMOL script (<name>_pymolcmd.pml) that loads it and colours its domains; one pair per '
                             'chain (<name>_<chain>.*) when several chains are processed. Needs a structure input.')

    parser.add_argument('--no-graph',
                        action='store_true',
                        help='Do not write the graph file (<name>_graph.npz) that is otherwise saved with the outputs.')
    parser.add_argument('--no-post',
                        action='store_true',
                        help='Return the chosen partition without the post-processing. By default every sequence '
                             'segment shorter than 40 residues and every domain with fewer than 2 secondary-structure '
                             'elements is merged into the sequence-adjacent domain with the highest self-containment F.')
    parser.add_argument('--stride', metavar='PATH', default=None,
                        help='STRIDE executable (or its directory). Default: $BIHC_STRIDE, src/stride, then the PATH.')
    parser.add_argument('--voronota', metavar='PATH', default=None,
                        help='Voronota 1.29 executable (or its directory). Default: $BIHC_VORONOTA, src/voronota, '
                             'then the PATH.')
    parser.add_argument('--allow-lattice', action='store_true',
                        help='Run without Voronota, on the pure-Python lattice approximation of the contact areas '
                             '(it has its own threshold).')
    return parser


def check_programs(args):
    """Locate STRIDE and Voronota before any structure is read; exit with install instructions when one is missing.
    Without Voronota the lattice approximation is used only on request (--allow-lattice): it computes different
    contact areas, so a missing install must not switch the method silently."""
    from src import bihc_core
    for name, given in (('stride', args.stride), ('voronota', args.voronota)):
        if given:
            os.environ[bihc_core.PROGRAM_ENV[name]] = given          # also seen by the STRIDE call in Functions
    bihc_core.VORONOTA_BIN = bihc_core.find_program('voronota')
    missing = [n for n in ('stride', 'voronota') if bihc_core.find_program(n) is None]
    if 'voronota' in missing and args.allow_lattice:
        missing.remove('voronota')
        print('WARNING: Voronota not found; --allow-lattice: contact areas from the lattice approximation')
    if missing:
        sys.exit('\n'.join(bihc_core.missing_program_message(n) for n in missing))
    print(f"STRIDE: {bihc_core.find_program('stride')}")
    if bihc_core.VORONOTA_BIN:
        version = bihc_core.voronota_version()
        print(f"Voronota {version or '(version unknown)'}: {bihc_core.VORONOTA_BIN}")
        if not (version or '').startswith('1.29'):
            print('WARNING: BiHC uses Voronota 1.29; other versions may give other contact areas')


GRAPH_SUFFIX = '_graph.npz'
STRUCTURE_EXTS = ('.pdb', '.cif', '.ent', '.mmcif')


def input_kind(path):
    """'graph' for a saved graph (.npz) or a directory holding *_graph.npz files, otherwise 'structure'. A directory
    with both kinds -- the output directory of a run with -o, whose .pdb files are the written domains -- is read as
    graphs."""
    if not os.path.isdir(path):
        return 'graph' if path.lower().endswith('.npz') else 'structure'
    return 'graph' if any(n.endswith(GRAPH_SUFFIX) and os.path.isfile(os.path.join(path, n))
                          for n in os.listdir(path)) else 'structure'


def main_argument():
    return main_parser().parse_args()


def process_args():
    args = main_argument()

    algo = 'BiHC'

    t_hm = args.threshold

    graph_mode = input_kind(args.input) == 'graph'
    if graph_mode and args.output_pdb:
        main_parser().error('--output-pdb needs the structure file; it cannot be used with a saved graph as input')
    source = args.input
    # output name: the structure name the graph was saved under (out/1a4i_graph.npz -> 1a4i)
    stem = (source[:-len('_graph.npz')] if source.endswith('_graph.npz') else source) if graph_mode else source

    largs = [source, 'B']

    msg = 'Input information:'
    decorate_message(msg)

    print("Using algorithm: ", algo)
    print("Pipeline: SSE super-nodes + Voronoi contact areas + self-containment ratio F (protein chains)")
    if graph_mode:
        print(f"Input: saved graph {source} (STRIDE / Voronota are not re-run)")
        if os.path.isdir(source) and any(n.lower().endswith(STRUCTURE_EXTS) for n in os.listdir(source)):
            print(f"  the directory also holds structure files; only its {GRAPH_SUFFIX} files are read")
    else:
        check_programs(args)
    print(f"partition selection: harmonic-mean objective   threshold t: {t_hm if t_hm is not None else 0.365}"
          f"   post-processing: {'off' if args.no_post else 'MINSEGMENT 40 -> min2ss'}")

    # any output option writes the JSON, the .tsv and the graph (default directory: the current one); the structure
    # file and the PyMOL script only on request
    if args.outpath is not None or args.json is not None or args.output_pdb:
        args.outpath = args.outpath if args.outpath is not None else '.'
        args.json = args.json or stem

    largs2 = [args.outpath, args.verbose, 'CA', args.json, stem if args.output_pdb else None,
              args.chain, None, True, False, t_hm, not args.no_post,
              not args.no_graph and not graph_mode, graph_mode,
              args.output_pdb]          # y[11]: save the graph, y[12]: saved-graph input, y[13]: write the .pml

    largs += [args.min_length]                 # x[-1]: minimum chain length

    return largs, largs2
