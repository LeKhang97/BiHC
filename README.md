# BiHC

## CLI arguments

### Core arguments:

-  i / --input : input file (.pdb / .cif)

- c / --chain : chain selection
  - all (default) = process all valid chains
  - longest = process the longest valid chain only
  - or a specific chain ID (e.g. A, 1, BG)

-  at / --atom_type : atom type used to build residue graph (default: C3')

-  d / --distance : distance cutoff to create edges (default: 7.5 A for protein, 15 A for RNA)

-  t / --threshold : minimum sequence length cutoff (default: 30)

-  w / --weight : apply weighted graph (True/False, default: True)

-  v / --verbose : verbose mode

-  o / --outpath : output directory (default: current directory when outputs enabled)

-  j / --json : JSON output basename (default = input name)

-  p / --pdb : structure output basename (default = input name)

BiHC parameters (subcommand B is forced internally):

-  s : split threshold (default = 0.4)

-  m : merge threshold (default = 0.2)

-  r : resolution (default = 1)

### Example (explicit BiHC params):
```
python3 BiHC.py -i example.cif -c chain -v -o output_dir B -s 0.4 -m 0.2 -r 1
```
