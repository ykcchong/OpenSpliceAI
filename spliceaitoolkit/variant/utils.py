from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import keras.models
import torch
import logging
import onnx
import platform
import os
from spliceaitoolkit.train.spliceai import SpliceAI
from spliceaitoolkit.constants import *

# from onnx2keras import onnx_to_keras

import re
from tensorflow import keras
import logging
import inspect
from onnx import numpy_helper

import onnx2keras
from onnx2keras.layers import AVAILABLE_CONVERTERS
from onnx2keras.converter import onnx_node_attributes_to_dict

def sanitize_layer_name(name):
    # Replace invalid characters with underscore
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)

def onnx_to_keras(onnx_model, input_names,
                  input_shapes=None, name_policy=None, verbose=True, change_ordering=False):
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param input_shapes: override input shapes (experimental)
    :param name_policy: override layer names. None, "short" or "renumerate" (experimental)
    :param verbose: verbose output
    :param change_ordering: change ordering to HWC (experimental)
    :return: Keras model
    """
    # Use channels first format by default.
    keras_fmt = keras.backend.image_data_format()
    keras.backend.set_image_data_format('channels_first')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger('onnx2keras')

    logger.info('Converter is called.')

    onnx_weights = onnx_model.graph.initializer
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    logger.debug('List input shapes:')
    logger.debug(input_shapes)

    logger.debug('List inputs:')
    for i, input in enumerate(onnx_inputs):
        logger.debug('Input {0} -> {1}.'.format(i, input.name))

    logger.debug('List outputs:')
    for i, output in enumerate(onnx_outputs):
        logger.debug('Output {0} -> {1}.'.format(i, output))

    logger.debug('Gathering weights to dictionary.')
    weights = {}
    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
            weights[sanitize_layer_name(onnx_extracted_weights_name)] = numpy_helper.to_array(onnx_w)
        except:
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            weights[sanitize_layer_name(onnx_extracted_weights_name)] = numpy_helper.to_array(onnx_w)

        logger.debug('Found weight {0} with shape {1}.'.format(
                     sanitize_layer_name(onnx_extracted_weights_name),
                     weights[sanitize_layer_name(onnx_extracted_weights_name)].shape))

    layers = dict()
    lambda_funcs = dict()
    keras_outputs = []
    keras_inputs = []

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                if input_shapes:
                    input_shape = input_shapes[i]
                else:
                    input_shape = [dim.dim_value for dim in onnx_i.type.tensor_type.shape.dim][1:]

                layers[sanitize_layer_name(input_name)] = keras.layers.InputLayer(
                    input_shape=input_shape, name=sanitize_layer_name(input_name)
                ).output

                keras_inputs.append(layers[sanitize_layer_name(input_name)])

                logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))

    # Convert every operation separable
    node_names = []
    for node_index, node in enumerate(onnx_nodes):
        node_type = node.op_type
        node_params = onnx_node_attributes_to_dict(node.attribute)

        # Add global converter info:
        node_params['change_ordering'] = change_ordering
        node_params['name_policy'] = name_policy

        node_name = str(node.output[0])
        keras_names = []
        for output_index, output in enumerate(node.output):
            sanitized_output = sanitize_layer_name(output)
            if name_policy == 'short':
                keras_name = keras_name_i = sanitized_output[:8]
                suffix = 1
                while keras_name_i in node_names:
                    keras_name_i = keras_name + '_' + str(suffix)
                    suffix += 1
                keras_names.append(keras_name_i)
            elif name_policy == 'renumerate':
                postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                keras_names.append('LAYER_%s' % postfix)
            else:
                keras_names.append(sanitized_output)

        if len(node.output) != 1:
            logger.warning('Trying to convert multi-output node')
            node_params['_outputs'] = list(node.output)
            node_names.extend(keras_names)
        else:
            keras_names = keras_names[0]
            node_names.append(keras_names)

        logger.debug('######')
        logger.debug('...')
        logger.debug('Converting ONNX operation')
        logger.debug('type: %s', node_type)
        logger.debug('node_name: %s', node_name)
        logger.debug('node_params: %s', node_params)
        logger.debug('...')

        logger.debug('Check if all inputs are available:')
        if len(node.input) == 0 and node_type != 'Constant':
            raise AttributeError('Operation doesn\'t have an input. Aborting.')

        for i, node_input in enumerate(node.input):
            logger.debug('Check input %i (name %s).', i, node_input)
            if sanitize_layer_name(node_input) not in layers:
                logger.debug('The input not found in layers / model inputs.')

                if sanitize_layer_name(node_input) in weights:
                    logger.debug('Found in weights, add as a numpy constant.')
                    layers[sanitize_layer_name(node_input)] = weights[sanitize_layer_name(node_input)]
                else:
                    raise AttributeError('Current node is not in weights / model inputs / layers.')
        else:
            logger.debug('... found all, continue')

        keras.backend.set_image_data_format('channels_first')
        AVAILABLE_CONVERTERS[node_type](
            node,
            node_params,
            layers,
            lambda_funcs,
            node_name,
            keras_names
        )
        if isinstance(keras_names, list):
            keras_names = keras_names[0]

        try:
            logger.debug('Output TF Layer -> ' + str(layers[sanitize_layer_name(keras_names)]))
        except KeyError:
            pass

    # Check for terminal nodes
    for layer in onnx_outputs:
        if sanitize_layer_name(layer) in layers:
            keras_outputs.append(layers[sanitize_layer_name(layer)])

    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)

    if change_ordering:
        import numpy as np
        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'shared_axes' in layer['config']:
                # TODO: check axes first (if it's not 4D tensor)
                layer['config']['shared_axes'] = [1, 2]

            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(np.array(
                        [
                            [None] +
                            list(layer['config']['batch_input_shape'][2:][:]) +
                            [layer['config']['batch_input_shape'][1]]
                        ]), -1
                    ))
            if layer['config'] and 'target_shape' in layer['config']:
                if len(list(layer['config']['target_shape'][1:][:])) > 0:
                    layer['config']['target_shape'] = \
                        tuple(np.reshape(np.array(
                                list(layer['config']['target_shape'][1:]) +
                                [layer['config']['target_shape'][0]]
                            ), -1),)

            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                if layer['config']['axis'] == 3:
                    layer['config']['axis'] = 1
                if layer['config']['axis'] == 1:
                    layer['config']['axis'] = 3

        for layer in conf['layers']:
            if 'function' in layer['config'] and layer['config']['function'][1] is not None:
                kerasf = list(layer['config']['function'])
                dargs = list(kerasf[1])
                func = lambda_funcs.get(layer['name'])

                if func:
                    if len(dargs) > 1:
                        params = inspect.signature(func).parameters                    
                        i = list(params.keys()).index('axes') if ('axes' in params) else -1

                        if i > 0:
                            i -= 1
                            axes = list(range(len(dargs[i].shape)))
                            axes = axes[0:1] + axes[2:] + axes[1:2]
                            dargs[i] = np.transpose(dargs[i], axes)

                        i = list(params.keys()).index('axis') if ('axis' in params) else -1

                        if i > 0:
                            i -= 1
                            axis = np.array(dargs[i])
                            axes_map = np.array([0, 3, 1, 2])
                            dargs[i] = axes_map[axis]
                    else:
                        if dargs[0] == -1:
                            dargs = [1]
                        elif dargs[0] == 3:
                            dargs = [1]

                kerasf[1] = tuple(dargs)
                layer['config']['function'] = tuple(kerasf)

        keras.backend.set_image_data_format('channels_last')
        model_tf_ordering = keras.models.Model.from_config(conf)

        for dst_layer, src_layer, conf in zip(model_tf_ordering.layers, model.layers, conf['layers']):
            W = src_layer.get_weights()
            # TODO: check axes first (if it's not 4D tensor)
            if conf['config'] and 'shared_axes' in conf['config']:
                W[0] = W[0].transpose(1, 2, 0)
            dst_layer.set_weights(W)

        model = model_tf_ordering

    keras.backend.set_image_data_format(keras_fmt)

    return model

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

def convert_pt_to_keras(model_path, CL, output_dir): 

    ## check compatibility ##

    # Print the keys in the state_dict
    state_dict = torch.load(model_path)
    print("State_dict keys:")
    print(', '.join(state_dict.keys()))

    # Print model's named parameters
    device = setup_device()
    model, params = load_model(device, CL)
    print("\nModel's named parameters:")
    print(','.join([name for name, param in model.named_parameters()]))

    state_dict_keys = set(state_dict.keys())
    model_param_keys = set(name for name, _ in model.named_parameters())

    # Check for keys that are in the state_dict but not in the model
    extra_keys = state_dict_keys - model_param_keys
    if extra_keys:
        print("Extra keys in state_dict:\n", extra_keys)

    # Check for keys that are in the model but not in the state_dict
    missing_keys = model_param_keys - state_dict_keys
    if missing_keys:
        print("Missing keys in state_dict:\n", missing_keys)

    ## perform conversion ##

    # load state dict, use non-strict if necessary
    try:
        model.load_state_dict(state_dict)
    except Exception as e1:
        try: 
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully with strict=False")
        except Exception as e2:
            print(f"Error loading model: {e1}, {e2}")
    else:
        print('State dict fully loaded')
    model.eval()
    
    # convert to onnx
    dummy_input = torch.randn(1, 4, SL+CL).to(device)
    torch.onnx.export(model, dummy_input, f"{output_dir}spliceai.onnx", input_names=['input'], output_names=['output'], dynamic_axes={'input': {0:'batch_size'}, 'output': {0:'batch_size'}})
    onnx_model = onnx.load(f"{output_dir}spliceai.onnx")
    print('Onnx model loaded')

    # convert onnx to keras
    input_names = [input.name for input in onnx_model.graph.input]
    print(input_names)
    k_model = onnx2keras.onnx_to_keras(onnx_model, input_names, input_shapes=[(4, SL+CL)], name_policy='renumerate', verbose=True) # THROWING ERROR

    # save as h5
    keras.models.save_model(k_model, f'{output_dir}model_keras.h5')
    keras_model = keras.models.load_model(f'{output_dir}model_keras.h5')

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
####################################################################################################################################

#######                                                 ORIGINAL                                                             #######

####################################################################################################################################
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

class Annotator:
    """
    Annotator class to handle gene annotations and reference sequences.
    It initializes with the reference genome, annotation data, and optional model configuration.
    """
    
    def __init__(self, ref_fasta, annotations, output_dir, model=None, CL=80):
        """
        Initializes the Annotator with reference genome, annotations, and model settings.
        
        Args:
            ref_fasta (str): Path to the reference genome FASTA file.
            annotations (str): Path or name of the annotation file (e.g., 'grch37', 'grch38').
            output_dir (str): Directory for output files.
            model (str, optional): Path to the model file or type of model ('SpliceAI'). Defaults to None.
            CL (int, optional): Context length parameter for model conversion. Defaults to 80.
        """

        # Load annotation file based on provided annotations type
        if annotations == 'grch37':
            annotations = resource_filename(__name__, 'annotations/grch37.txt')
        elif annotations == 'grch38':
            annotations = resource_filename(__name__, 'annotations/grch38.txt')

        # Load and parse the annotation file
        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            # Extract relevant columns into numpy arrays for efficient access
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy() + 1  # Transcription start sites (1-based indexing)
            self.tx_ends = df['TX_END'].to_numpy()  # Transcription end sites
            
            # Extract and process exon start and end sites, convert into numpy array format
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i]) + 1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e)) 
            exit()  # Exit if the file cannot be read
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()  # Exit if the file format is incorrect

        # Load the reference genome fasta file
        try:
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
        except IOError as e:
            logging.error('{}'.format(e))  # Log file read error
            exit()  # Exit if the file cannot be read

        # Load models based on the specified model type or file
        if model == 'SpliceAI':
            paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))  # Generate paths for SpliceAI models
            self.models = [load_model(resource_filename(__name__, x)) for x in paths]
        else:
            self.models = []
            for m in [path.strip() for path in model.split(',')]:
                if m.endswith('.pt'):  # Convert PyTorch model to Keras
                    keras_model = convert_pt_to_keras(m, CL, output_dir)
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
