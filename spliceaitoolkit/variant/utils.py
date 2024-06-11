from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from pyfaidx import Fasta
from keras.models import save_model, load_model
import torch
import logging
import onnx
import platform
from onnx2keras import onnx_to_keras
from spliceaitoolkit.train.spliceai import SpliceAI
from spliceaitoolkit.constants import *


def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)

def load_model(device, flanking_size):
    """Loads the given model."""
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    L = 32
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    N_GPUS = 2
    BATCH_SIZE = 18*N_GPUS

    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12*N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6*N_GPUS

    CL = 2 * np.sum(AR*(W-1))

    print(f"\t[INFO] Context nucleotides {CL}")
    print(f"\t[INFO] Sequence length (output): {SL}")
    
    model = SpliceAI(L, W, AR).to(device)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}

    return model, params


class Annotator:

    def __init__(self, ref_fasta, annotations, model=None, CL=80):

        if annotations == 'grch37':
            annotations = resource_filename(__name__, 'annotations/grch37.txt')
        elif annotations == 'grch38':
            annotations = resource_filename(__name__, 'annotations/grch38.txt')

        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy()+1
            self.tx_ends = df['TX_END'].to_numpy()
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i])+1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e))
            exit()
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()

        try:
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
        except IOError as e:
            logging.error('{}'.format(e))
            exit()
        
        if model == 'SpliceAI':
            paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
            self.models = [load_model(resource_filename(__name__, x)) for x in paths]
        else:
            self.models = []
            for m in model.split(','):
                if m.endswith('.pt'): # convert from pytorch to onnx to keras 
                    device = setup_device()
                    spliceai, _ = load_model(device, CL)
                    spliceai = spliceai.to(device)
                    spliceai.load_state_dict(torch.load(m))
                    spliceai.eval()

                    dummy_input = torch.randn(1, 4, SL+CL).to(device)
                    torch.onnx.export(spliceai, dummy_input, "spliceai.onnx")
                    onnx_model = onnx.load("spliceai.onnx")

                    input_names = [input.name for input in onnx_model.graph.input]
                    print(input_names)
                    k_model = onnx_to_keras(onnx_model, input_names) # THROWING ERROR

                    save_model(k_model, 'model_keras.h5')
                    keras_model = load_model('model_keras.h5')
                    self.models.append(keras_model)
                else:
                    self.models.append(load_model(m))

    def get_name_and_strand(self, chrom, pos):

        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                              np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                              np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs
        else:
            return [], [], []

    def get_pos_data(self, idx, pos):

        dist_tx_start = self.tx_starts[idx]-pos
        dist_tx_end = self.tx_ends[idx]-pos
        dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx])-pos, key=abs)
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)

        return dist_ann


def one_hot_encode(seq):

    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]


def normalise_chrom(source, target):

    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr'+source

    return source


def get_delta_scores(record, ann, dist_var, mask):

    cov = 2*dist_var+1
    wid = 10000+cov
    delta_scores = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return delta_scores

    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return delta_scores

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][record.pos-wid//2-1:record.pos+wid//2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return delta_scores

    if seq[wid//2:wid//2+len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref issue): {}'.format(record))
        return delta_scores

    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): {}'.format(record))
        return delta_scores

    if len(record.ref) > 2*dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return delta_scores

    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue

            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                delta_scores.append("{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], genes[i]))
                continue

            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid//2+dist_ann[0], 0), max(wid//2-dist_ann[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len-alt_len, 0)

            x_ref = 'N'*pad_size[0]+seq[pad_size[0]:wid-pad_size[1]]+'N'*pad_size[1]
            x_alt = x_ref[:wid//2]+str(record.alts[j])+x_ref[wid//2+ref_len:]

            x_ref = one_hot_encode(x_ref)[None, :]
            x_alt = one_hot_encode(x_alt)[None, :]

            if strands[i] == '-':
                x_ref = x_ref[:, ::-1, ::-1]
                x_alt = x_alt[:, ::-1, ::-1]

            y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(len(ann.models))], axis=0)
            y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(len(ann.models))], axis=0)

            if strands[i] == '-':
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]

            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov//2+alt_len],
                    np.zeros((1, del_len, 3)),
                    y_alt[:, cov//2+alt_len:]],
                    axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov//2],
                    np.max(y_alt[:, cov//2:cov//2+alt_len], axis=1)[:, None, :],
                    y_alt[:, cov//2+alt_len:]],
                    axis=1)

            y = np.concatenate([y_ref, y_alt])

            idx_pa = (y[1, :, 1]-y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1]-y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2]-y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2]-y[1, :, 2]).argmax()

            mask_pa = np.logical_and((idx_pa-cov//2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na-cov//2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd-cov//2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd-cov//2 != dist_ann[2]), mask)

            delta_scores.append("{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
                                record.alts[j],
                                genes[i],
                                (y[1, idx_pa, 1]-y[0, idx_pa, 1])*(1-mask_pa),
                                (y[0, idx_na, 1]-y[1, idx_na, 1])*(1-mask_na),
                                (y[1, idx_pd, 2]-y[0, idx_pd, 2])*(1-mask_pd),
                                (y[0, idx_nd, 2]-y[1, idx_nd, 2])*(1-mask_nd),
                                idx_pa-cov//2,
                                idx_na-cov//2,
                                idx_pd-cov//2,
                                idx_nd-cov//2))

    return delta_scores
