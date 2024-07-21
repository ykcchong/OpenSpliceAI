from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import os, sys, re
import numpy as np
import h5py

input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
# Replace this with your custom sequence

context = 10000 # [80, 400, 2000, 10000]
paths = (f'models/SpliceAI/SpliceNet{context}_c{x}.h5' for x in range(1, 6))
models = [load_model(x) for x in paths]
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = np.mean([models[m].predict(x) for m in range(5)], axis=0)

# y is gene_predictions
acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]

print(f'Acceptor probabilities: {acceptor_prob}')
print(f'Donor probabilities: {donor_prob}')