import argparse
from pathlib import Path
from src.Functions import *

def main_argument():
    parser = argparse.ArgumentParser(description ="This tool is used to detect RNA domains from given 3D coordinate :))")

    subparsers = parser.add_subparsers(dest='subcommand')

    parser.add_argument('-v', '--verbose',
        action ='store_true', 
        help ='verbose mode.')
    
    parser.add_argument('-i',
        '--input',
        required=True,
        help ='input file. Must be in pdb format.')
    
    parser.add_argument('-at',
        '--atom_type',
        default = "C3'",
        help ="Atom types to be considered in the analysis. Default is C3'.")
    
    parser.add_argument('-t',
        '--threshold',
        default= 30,
        type= int,
        help ='Lower threshold for sequence length')

    parser.add_argument('-d',
        '--distance',
        default= None,
        type=float,
        help="Distance threshold to form edge between nodes (default = 7.5 A for protein, 15 A for RNA).")

    parser.add_argument('-w', '--weight', 
                        default= 'True',
                        nargs='?', 
                        const = 'True',
                        type= str,
                        choices= ['True', 'False'],
                        help="Apply weighted graph")

    parser.add_argument('-o', '--outpath',
                nargs='?', 
                const='.',
                default= None, 
                type=str,
                help ="path of output for json and pdb files. If not specified, the output will be saved in the current directory.")
    
    parser.add_argument('-c', '--chain',
                        type=str, 
                        nargs='?', 
                        const=False, 
                        default='all', 
                        help='Name of the chain to be processed. If not specified, all chains will be processed.')
    
    parser.add_argument('-j', '--json', 
                        type= str, 
                        nargs = '?', 
                        const = False, 
                        default = None, 
                        help='Name of the output json files. If not specified, its name will be the same as the input file')
    
    parser.add_argument('-p', '--pdb', 
                        type= str, 
                        nargs = '?', 
                        const = False, 
                        default = None, 
                        help='Name of the output pdb file(s). If not specified, its name will be the same as the input file')
    
    parser.add_argument('-a', 
					'--algorithm',
                    default = 'B',
					choices = ['B'],
					help="Clustering algorithm. (BiHC))")
    
    parser.add_argument('-pp', '--post_process', 
                        action='store_false', 
                        help="Disable post-processing")
  
    parser.add_argument('-r', '--reference',
                        type=str, 
                        nargs='?', 
                        const=False, 
                        default=None, 
                        help="Reference partitions. Domains are separated by semi-colon. Discontinuous domain has segments separated by '+' symbol")

    '''parser.add_argument('-s', '--segment',
                        type=str, 
                        nargs='?',  
                        default='30;10;100', 
                        help="Segment sizes for post-processing. Default is '30;10;100' (i.e., domain segments of length >=30 are kept, segments of length between 10 and 30 are merged, outlier segments of length <10 are discarded)")'''

    # Subparser for -a B
    parser_a_B = subparsers.add_parser('B', help='Arguments for Bidirectional Hierarchical clustering algorithm')
    parser_a_B.add_argument('-s', type = float, default= 0.4, help='Modularity threshold for segment to be splitted (default = 0.4)')
    parser_a_B.add_argument('-m', type=float,default= 0.2, help= "Contact ratio threshold to be merged then (default = 0.2)")
    parser_a_B.add_argument('-r', type=float, default= 1, help='resolution of the algorithm (default = 1)')


    args = parser.parse_args()      
    
    return args

def process_args():
    args = main_argument()

    args.algorithm = 'B' # force to use BiHC
    algo = 'BiHC' # force to use BiHC

    largs = [args.input, args.algorithm]

    msg = 'Input information:'
    decorate_message(msg)

    print('Using atom type: ', args.atom_type)
    print("Using algorithm: ", algo)
    
    print(f'Mode selected for {algo} algorithm:', end = ' ')

    if args.outpath != None and args.json == None and args.pdb == None:
        args.json = args.input
        args.pdb = args.input
    
    if args.json == False:
        args.json = args.input

    if args.pdb == False:
        args.pdb = args.input
    
    if (args.outpath == None) and (args.json != None or args.pdb != None):
        args.outpath = '.'

    largs2 = [args.outpath, args.verbose, args.atom_type, args.json, args.pdb, args.chain, args.reference, args.post_process]

    if args.algorithm == 'B':
        if not hasattr(args, 's'):
            args.s = 0.4
        
        if not hasattr(args, 'm'):
            args.m = 0.2
        if not hasattr(args, 'r'):
            args.r = 1

        print(f"split threshold: {args.s}, merge threshold: {args.m}, resolution: {args.r}")
        largs += [args.s, args.m, args.r]
    
    else:
        sys.exit("Unrecognized algorithm!")

    largs += [args.weight, args.distance, args.threshold]

    #print(largs)

    return largs, largs2