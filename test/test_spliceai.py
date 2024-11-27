from keras.models import load_model
from pkg_resources import resource_filename
import numpy as np

def one_hot_encode(seq):

    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]


input_sequence = open('./test/example.fa').read().split('\n')[1]
# Replace this with your custom sequence

context = 10000
paths = ('./models/SpliceAI/SpliceNet10000_c{}.h5'.format(x) for x in range(1, 6))
models = [load_model(x) for x in paths]
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = np.mean([models[m].predict(x) for m in range(5)], axis=0)

acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]

print(acceptor_prob, donor_prob)