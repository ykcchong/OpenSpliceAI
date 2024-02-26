###############################################################################
# This file has the functions necessary to create the SpliceAI model.
###############################################################################
import tensorflow as tf  # Make sure TensorFlow is imported
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv1D, Cropping1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add
import keras.backend as kb
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, add, Layer, Dense, Softmax, Multiply

class MultiHeadAttention(Layer):
    def __init__(self, num_heads, model_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        assert model_dim % self.num_heads == 0, "Model dimension must be divisible by number of heads."
        self.depth = model_dim // self.num_heads
        self.query_dense = Dense(model_dim)
        self.key_dense = Dense(model_dim)
        self.value_dense = Dense(model_dim)
        self.dense = Dense(model_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, model_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, model_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, model_dim)
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len_q, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_output = tf.reshape(output, (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, model_dim)
        output = self.dense(concat_output)  # (batch_size, seq_len_q, model_dim)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "model_dim": self.model_dim,
        })
        return config




def ResidualUnit(l, w, ar):
    def f(input_node):
        bn1 = BatchNormalization()(input_node)
        act1 = Activation('relu')(bn1)
        conv1 = Conv1D(l, kernel_size=(w,), dilation_rate=(ar,), padding='same')(act1)  # Adjusted here
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv1D(l, kernel_size=(w,), dilation_rate=(ar,), padding='same')(act2)  # Adjusted here
        output_node = add([conv2, input_node])

        return output_node

    return f


def SpliceAI_Multihead(L, W, AR):
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    assert len(W) == len(AR)
    CL = 2 * np.sum(AR*(W-1))
    input0 = Input(shape=(None, 4))
    conv = Conv1D(L, 1)(input0)
    skip = Conv1D(L, 1)(conv)
    for i in range(len(W)):
        conv = ResidualUnit(L, W[i], AR[i])(conv)
        if (((i+1) % 4 == 0) or ((i+1) == len(W))):
            # Skip connections to the output after every 4 residual units
            dense = Conv1D(L, 1)(conv)
            skip = add([skip, dense])
    # skip = Cropping1D(CL/2)(skip)

    # Calculate cropping amount, ensuring it's an integer
    cropping_amount = int(np.round(CL / 2))
    # Use a tuple with the calculated amount for both the beginning and end
    skip = Cropping1D((cropping_amount, cropping_amount))(skip)

    ############################
    # Add MultiheadAttention here; ensure embedding dim is divisible by num_heads
    ############################
    num_heads = 4  # Choose based on your model dimension L
    model_dim = L  # Assuming model_dim is equal to the number of filters L in your conv layers

    # After defining your model up to the point where you want to add attention
    attention_layer = MultiHeadAttention(num_heads=num_heads, model_dim=model_dim)
    attention_output = attention_layer(skip)  # 'skip' is the output from previous layers
    ############################
    # End of MultiheadAttention layer
    ############################

    # Final prediction layer
    output0 = Conv1D(filters=3, kernel_size=(1,), activation='softmax')(attention_output)
    model = Model(inputs=input0, outputs=output0)
    return model



def categorical_crossentropy_2d(y_true, y_pred):
    # Standard categorical cross entropy for sequence outputs
    # Convert y_true to float32 to match y_pred's type
    y_true = tf.cast(y_true, tf.float32)

    return - kb.mean(y_true[:, :, 0]*kb.log(y_pred[:, :, 0]+1e-10)
                   + y_true[:, :, 1]*kb.log(y_pred[:, :, 1]+1e-10)
                   + y_true[:, :, 2]*kb.log(y_pred[:, :, 2]+1e-10))