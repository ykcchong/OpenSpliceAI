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
# initialize global variables
initialize_globals(flanking_size, hdf_threshold_len, flush_predict_threshold, chunk_size, split_fasta_threshold)

print(f'''Running predict with SL: {sequence_length}, flanking_size: {flanking_size}, threshold: {threshold}, in {'debug, ' if debug else ''}{'turbo' if not predict_all else 'all'} mode.
        model: {model_path}, 
        input_sequence: {input_sequence}, 
        gff_file: {gff_file},
        output_dir: {output_dir},
        hdf_threshold_len: {hdf_threshold_len}, flush_predict_threshold: {flush_predict_threshold}, split_fasta_threshold: {split_fasta_threshold}, chunk_size: {chunk_size}''')

# create output directory
os.makedirs(output_dir, exist_ok=True)
output_base = initialize_paths(output_dir, flanking_size, sequence_length)
print("Output path: ", output_base, file=sys.stderr)
print("Model path: ", model_path, file=sys.stderr)
print("Flanking sequence size: ", flanking_size, file=sys.stderr)
print("Sequence length: ", sequence_length, file=sys.stderr)

# PART 1: Extracting input sequence
print("--- Step 1: Extracting input sequence ... ---", flush=True)
start_time = time.time()

# if gff file is provided, extract just the gene regions into a new FASTA file
if gff_file: 
    print('\t[INFO] GFF file provided: extracting gene sequences.')
    new_fasta = process_gff(input_sequence, gff_file, output_base)
    datafile_path, NAME, SEQ = get_sequences(new_fasta, output_base)
else:
    # otherwise, collect all sequences from FASTA into file
    datafile_path, NAME, SEQ = get_sequences(input_sequence, output_base)

# print_motif_counts()

print("--- %s seconds ---" % (time.time() - start_time))

### PART 2: Getting one-hot encoding of inputs
print("--- Step 2: Creating one-hot encoding ... ---", flush=True)
start_time = time.time()

dataset_path, LEN = convert_sequences(datafile_path, output_base, SEQ, debug=debug)

print("--- %s seconds ---" % (time.time() - start_time))


### PART 3: Loading model
print("--- Step 3: Load model ... ---", flush=True)
start_time = time.time()

# setup device
device = setup_device()

# load model from current state
model, params = load_model(device, flanking_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
print(f"\t[INFO] Device: {device}, Model: {model}, Params: {params}")

print("--- %s seconds ---" % (time.time() - start_time))

if predict_all: # predict using intermediate files

    ### PART 4: Get predictions
    print("--- Step 4: Get predictions ... ---", flush=True)
    start_time = time.time()

    predict_file = get_prediction(model, dataset_path, device, params, output_base, debug=debug)

    print("--- %s seconds ---" % (time.time() - start_time))


    ### PART 5: Generate BED report
    print("--- Step 5: Generating BED report ... ---", flush=True)
    start_time = time.time()

    generate_bed(predict_file, NAME, LEN, output_base, threshold=threshold, debug=debug)

    print("--- %s seconds ---" % (time.time() - start_time))

else: # combine prediction and output
    
    ### PART 4o: Get only predictions and write to BED
    print("--- Step 4o: Extract predictions to BED ... ---", flush=True)
    start_time = time.time()
    
    predict_file = predict_and_write(model, dataset_path, device, params, NAME, LEN, output_base, threshold=threshold, debug=debug)

    print("--- %s seconds ---" % (time.time() - start_time))  
