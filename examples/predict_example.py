from openspliceai.predict import predict
import argparse

# Replace your sequence in the file
input_sequence = './output.fasta'

# Replace with your model and flanking size
flanking_size = 10000
model = f'../models/spliceai-mane/{flanking_size}nt/model_{flanking_size}nt_rs14.pt'
print(model)

# Replace with the output directory
output_dir = './results'

# Create argparse Namespace
args = argparse.Namespace(
    model=model,             
    output_dir=output_dir,             
    flanking_size=flanking_size,            
    input_sequence=input_sequence,          
    annotation_file=None,         # Default value
    threshold=1e-6,               # Default value
    predict_all=False,            # Default value 
    debug=False,                  # Default value
    hdf_threshold=0,              # Default value
    flush_threshold=500,          # Default value
    split_threshold=1500000,      # Default value
    chunk_size=100                # Default value
)

predict.predict(args)

# Output will appear in the specified output directory


# ## To get arguments step-by-step
# predict.initialize_globals()
# os.makedirs(output_dir, exist_ok=True)
# output_base = predict.initialize_paths(output_dir, flanking_size)
