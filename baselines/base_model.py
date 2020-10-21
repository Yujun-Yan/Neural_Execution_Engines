from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import math

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, state_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, state_size, seq_len, d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask[:,:,tf.newaxis,:,:] * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        state_size = x.shape[-3]
        x = tf.reshape(x, (batch_size, state_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        state_size = tf.shape(q)[1]
        
        q = self.wq(q)  # (batch_size, state_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, state_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, state_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, state_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, state_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, state_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, state_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, state_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 1, 3, 2, 4])  # (batch_size, state_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, state_size, -1, self.d_model))  # (batch_size, state_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, state_size, seq_len_q, d_model)
        attention_weights = tf.reduce_mean(attention_weights, axis=2) # (batch_size, state_size, seq_len_q, seq_len_k)

        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff,  rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, 1)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        attn_output, attention_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        

        self.mha1 = MultiHeadAttention(d_model, 1)
        self.mha2 = MultiHeadAttention(d_model, 1)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, com_mask, dec_mask):
        # enc_output.shape == (batch_size, state_size, 1, d_model)
        

        attn1, attn_weights_block1 = self.mha1(x, x, x, com_mask)

        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, dec_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
    
    
def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos*angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis,:], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:,1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)
def binary_encoding(x, binary_size):
    pow_base = tf.reverse(tf.range(binary_size, dtype=tf.int64),[-1])
    out = tf.bitwise.bitwise_and(tf.expand_dims(x, -1), tf.bitwise.left_shift(tf.constant(1, dtype=tf.int64), pow_base))
    out = tf.cast(tf.greater(out, 0), tf.float32)
    return out
def back2int(x): ##x: int64
    binary_size = x.shape[-1]
    pow_base = tf.bitwise.left_shift(tf.constant(1, dtype=tf.int64),tf.reverse(tf.range(binary_size, dtype=tf.int64),[-1])[:, tf.newaxis])
    out = tf.matmul(x, pow_base)
    return tf.squeeze(out,-1)


def add_pos(x, mask, pos_enc):
    mask = tf.squeeze(1-mask, -2)
    x_ind = tf.math.cumsum(mask, -1)
    return x+tf.gather_nd(pos_enc, tf.cast(x_ind[:,:,:,tf.newaxis], dtype=tf.int64))



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, rate=0.1):
        super(Encoder, self).__init__()

        self.binary_size = binary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(pos, self.d_model)
        

        self.enc_layers = [EncoderLayer(d_model, dff,  rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask, use_pos, emb):
        x = binary_encoding(x, self.binary_size)
        x = emb(x)
            
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if use_pos:
            x = add_pos(x, mask, self.pos_encoding)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, state_size, seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, rate=0.1):
        super(Decoder, self).__init__()

        self.binary_size = binary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(pos, d_model)

        self.dec_layers = [DecoderLayer(d_model, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, 
           com_mask, dec_mask, use_pos, emb):

        attention_weights = {}
        x = binary_encoding(x, self.binary_size)
        x = emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if use_pos:
            x += tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 com_mask, dec_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, state_size, seq_len, d_model)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, 
               target_vocab_size,use_pos, out_num, out_pos, rate=0.1):
        super(Transformer, self).__init__()
        
        self.num_layers = num_layers

        self.encoder = Encoder(num_layers, d_model, binary_size, dff, pos, rate)
        self.decoder = Decoder(num_layers, d_model, binary_size, dff, pos,  rate)
        
        self.emb = tf.keras.layers.Dense(d_model, use_bias=False)

        self.use_pos = use_pos
        self.out_num = out_num
        self.out_pos = out_pos
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, enc_padding_mask, 
           com_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask, self.use_pos, self.emb)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, com_mask, dec_padding_mask, self.use_pos, self.emb)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return_list = []
        if self.out_num:
            return_list.append(final_output)
        return_list.append(attention_weights)
        if self.out_pos:
            last_att_weights = attention_weights['decoder_layer{}_block2'.format(self.num_layers)]
            last_att_weights = tf.reshape(last_att_weights, [-1, last_att_weights.shape[1]*last_att_weights.shape[2], last_att_weights.shape[-1]])
            return_list.append(last_att_weights)
        return return_list
    