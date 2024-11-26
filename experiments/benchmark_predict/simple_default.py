from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np

with open('data/toy/human/chr1.fa','r') as f:
    f.readline()
    input_sequence = f.read().replace('\n', '')
    
context = 10000
paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]
seq_len = 5000
for i in range(0, (len(input_sequence) // 5000) * 5000, seq_len):
    block = input_sequence[i:i+seq_len]
    x = one_hot_encode('N'*(context//2) + block + 'N'*(context//2))[None, :]
    y = np.mean([models[m].predict(x) for m in range(5)], axis=0)
    
    acceptor_prob = y[0, :, 1]
    donor_prob = y[0, :, 2]
    
    print('Block:', i,'\n', 'Acceptor:', acceptor_prob,'\n', 'Donor:', donor_prob)