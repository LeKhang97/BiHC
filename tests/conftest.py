import os, sys

# the repository root must shadow any other directory called "src"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
