from openspliceai.predict import predict
import argparse

# Replace your sequence in the file
input_sequence_file = './test/output.fasta'
input_sequence = open(input_sequence_file).read().strip()

# Replace with your model and flanking size
flanking_size = 10000
model = f'./models/spliceai-mane/{flanking_size}nt/model_{flanking_size}nt_rs14.pth'

# Replace with the output directory
output_dir = './test/results'

predictions = predict.predict(input_sequence, model, flanking_size)
print(predictions, predictions.shape)