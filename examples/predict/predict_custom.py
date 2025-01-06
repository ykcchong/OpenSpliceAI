'''Full predict process with custom data processing, batch prediction, and file output'''

from openspliceai.predict import *
from pathlib import Path

# Resolve the absolute path to parent dir
pdir = Path(__file__).resolve().parents[2]

# Define file paths and model
input_sequence_file = f'{pdir}/examples/predict/chr22.fa'
output_dir = f'{pdir}/examples/predict/results/'
flanking_size = 10000
model = f'{pdir}/models/spliceai-mane/{flanking_size}nt/model_{flanking_size}nt_rs14.pt'

## DATA PREPROCESSING
