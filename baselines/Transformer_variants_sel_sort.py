#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import math
import argparse


# In[2]:

parser = argparse.ArgumentParser()
parser.add_argument('-V', "--variants", choices=["all_mod", "all_mod_one_hot_output", "all_mod_one_hot_input", "all_mod_direct_binary_input", "all_mod_wo_sym", "all_mod_excp_dot_att", "all_mod_excp_res", "all_mod_excp_shared_proj", "vanilla", "vanilla_one_hot_input", "vanilla_direct_binary_input"], required=True, help='Choose a transformer variant to run. Select from "all_mod", "all_mod_one_hot_output", "all_mod_one_hot_input", "all_mod_direct_binary_input", "all_mod_wo_sym", "all_mod_excp_dot_att", "all_mod_excp_res", "all_mod_excp_shared_proj", "vanilla", "vanilla_one_hot_input", "vanilla_direct_binary_input"')
parser.add_argument('-E', "--evaluation", choices=["mixed", "hard", "random"], required=True, help='Choose evaluation difficulty level: "mixed", "hard", "random"')
parser.add_argument('--att_sup', action='store_true', help='using supervision for attention masks')
parser.add_argument('-R', "--Reload", type=str, default="No", help='Path (w.r.t current folder) to reload the model')
args = parser.parse_args() 


binary_size = 8
d_model = 16
Train_SIZE = 20000
BATCH_SIZE = 64
Val_size = 2000

num_layers = 6
dff = 128

train_sub_size = 50
val_sub_size = 10
rep_num = 20

if args.variants == "all_mod_one_hot_output":
    target_vocab_size = 2 ** binary_size + 2
else:
    target_vocab_size = binary_size + 2   #### end_token, inf
dropout_rate = 0.1
if args.variants == "all_mod_excp_res":
    res_ratio = 1
else:
    res_ratio = 1.5
if args.variants == "all_mod_wo_sym":
    make_sym = False
else:
    make_sym = True

if args.variants.startswith("vanilla") or args.variants == "all_mod_excp_dot_att":
    mha_att = True
else:
    mha_att = False

if args.variants.endswith("one_hot_input"):
    bin_enc = False
else:
    bin_enc = True

if (args.variants == "all_mod_direct_binary_input") or (args.variants == "all_mod_direct_binary_input"):
    use_emb = False
else:
    use_emb = True

if args.variants == "all_mod_excp_shared_proj":
    shared_proj = False
else:
    shared_proj = True



inf = 2 ** binary_size
end_token = 2 ** (binary_size+1)
pad_token = 2 ** (binary_size+2)
start_token_dec = 0


home_dir = "./"

file_name = "sel_sort_{}_eval_{}".format(args.variants, args.evaluation) + "_att_sup"*args.att_sup

current_time = time.strftime("%d_%H_%M_%S", time.localtime(time.time()))
current_dir = home_dir + file_name + "_" + current_time

subprocess.call("mkdir {}".format(current_dir), shell=True)
current_dir += '/'

############### task specific settings ###################

if args.Reload == "No":
    reload_from_dir_2 = False
else:
    reload_from_dir_2 = True

reload_dir_2 = home_dir + args.Reload + "/"

if reload_from_dir_2:
    EPOCHS_2 = 1
else:
    EPOCHS_2 = 300
seq_len_p2 = 8

out_num_2 = True
out_pos_2 = True
assert(out_num_2 or out_pos_2)
USE_positioning_2 = False
pos_2 = seq_len_p2 + 1

num_max_2 = 2 ** binary_size

if reload_from_dir_2:
    subprocess.call("cp -r {}model_2 {}".format(reload_dir_2, current_dir), shell=True)
    subprocess.call("cp {}*.npy {}".format(reload_dir_2, current_dir), shell=True)
with open("{}parameters.txt".format(current_dir), 'w') as fi:
    fi.write("binary_size: {}\nd_model: {}\nTrain_SIZE: {}\nBATCH_SIZE: {}\nVal_size: {}\nnum_layers: {}\ndff: {}\ntrain_sub_size: {}\nval_sub_size: {}\nrep_num: {}\ntarget_vocab_size: {}\ndropout_rate: {}\nmake_sym: {}\nEPOCHS_2: {}\nseq_len_p2: {}\nout_num_2: {}\nout_pos_2: {}\nUSE_positioning_2: {}\nnum_max_2: {}".format(
        binary_size, d_model, Train_SIZE, BATCH_SIZE, Val_size, num_layers, dff, train_sub_size, val_sub_size, rep_num, target_vocab_size, dropout_rate, make_sym, EPOCHS_2, seq_len_p2, out_num_2, out_pos_2, USE_positioning_2, num_max_2) + reload_from_dir_2 * "\nreload_dir_2: {}".format(reload_dir_2))
    


# In[3]:
if not reload_from_dir_2:
    probs = np.array([1.0]*num_max_2+[2.0]*2)
    probs = probs/np.sum(probs)
    Train_exmp = np.random.choice(list(range(num_max_2))+[end_token]+[inf], size=(Train_SIZE, seq_len_p2), p=probs)
    Val_exmp = np.random.choice(list(range(num_max_2))+[end_token]+[inf], (Val_size, seq_len_p2), p=probs)
    train_special = []
    val_special = []
    for i in range(4):
        step = i + 1
        start_tr_val = np.random.choice(range(num_max_2-(seq_len_p2-1)*step),(train_sub_size + val_sub_size, 1), replace=False)
        tr_val = np.tile(start_tr_val, [1, seq_len_p2])
        tr_val += np.arange(0, seq_len_p2*step, step)
        train_special.append(tr_val[:train_sub_size, :])
        val_special.append(tr_val[train_sub_size:, :])
    train_special = np.concatenate(train_special, axis=0)
    train_special = np.tile(train_special, (rep_num, 1))
    val_special = np.concatenate(val_special, axis=0)
    val_special = np.tile(val_special, (rep_num, 1))
    np.apply_along_axis(np.random.shuffle, 1, train_special)
    np.apply_along_axis(np.random.shuffle, 1, val_special)
    Train_exmp = np.concatenate([Train_exmp, train_special], axis=0)
    Val_exmp = np.concatenate([Val_exmp, val_special], axis=0)
    ###### randomly hide some numbers ############
    hide_num_train = np.random.choice(range(seq_len_p2), Train_exmp.shape[0]) 
    hide_num_val = np.random.choice(range(seq_len_p2), Val_exmp.shape[0]) 
    def rand_hide(row, hide_num):
        if hide_num:
            row[-hide_num:] = pad_token
    for i, row in enumerate(Train_exmp):
        rand_hide(row, hide_num_train[i])
    for i, row in enumerate(Val_exmp):
        rand_hide(row, hide_num_val[i])
    Sorted_train = np.sort(Train_exmp)
    Sorted_val = np.sort(Val_exmp)
    
    np.save("{}Train_exmp".format(current_dir), Train_exmp)
    np.save("{}Sorted_train".format(current_dir), Sorted_train)
    np.save("{}Val_exmp".format(current_dir), Val_exmp)
    np.save("{}Sorted_val".format(current_dir), Sorted_val)
    
    if args.att_sup:
        Sorted_train_ind = np.argsort(Train_exmp)
        Sorted_val_ind = np.argsort(Val_exmp)
        train_mask = np.zeros((Train_exmp.shape[0], seq_len_p2, seq_len_p2), dtype=np.float32)
        for i in range(seq_len_p2):
            train_mask[np.arange(Train_exmp.shape[0]), i, Sorted_train_ind[:,i]] = 1
        
        val_mask = np.zeros((Val_exmp.shape[0], seq_len_p2, seq_len_p2), dtype=np.float32)
        for i in range(seq_len_p2):
            val_mask[np.arange(Val_exmp.shape[0]), i, Sorted_val_ind[:,i]] = 1
        np.save("{}train_mask".format(current_dir), train_mask)
        np.save("{}val_mask".format(current_dir), val_mask)
        
    print("hide_num_train:")
    print(hide_num_train[:3])
    print("hide_num_val")
    print(hide_num_val[:3])
else:
    Train_exmp = np.load("{}Train_exmp.npy".format(reload_dir_2))
    Val_exmp = np.load("{}Val_exmp.npy".format(reload_dir_2))
    Sorted_train = np.load("{}Sorted_train.npy".format(reload_dir_2))
    Sorted_val = np.load("{}Sorted_val.npy".format(reload_dir_2))
    if args.att_sup:
        train_mask = np.load("{}train_mask.npy".format(reload_dir_2))
        val_mask = np.load("{}val_mask.npy".format(reload_dir_2))
    
print("Train_example:")
print(Train_exmp[:3,:])
print("Sorted_train:")
print(Sorted_train[:3])
print("Val_example:")
print(Val_exmp[:3,:])
print("Sorted_val:")
print(Sorted_val[:3])
if args.att_sup:
    print("train_mask:")
    print(train_mask[:3,:,:])
    print("val_mask:")
    print(val_mask[:3,:,:])

if args.att_sup:
    Train_dataset_2 = tf.data.Dataset.from_tensor_slices((Train_exmp, train_mask, Sorted_train))    
    Val_dataset_2 = tf.data.Dataset.from_tensor_slices((Val_exmp, val_mask, Sorted_val))  
else:
    Train_dataset_2 = tf.data.Dataset.from_tensor_slices((Train_exmp, Sorted_train))    
    Val_dataset_2 = tf.data.Dataset.from_tensor_slices((Val_exmp, Sorted_val))   
    
    
    
    


# In[4]:

def encode_t2(tr, mask, srt):
    srt = np.hstack(([start_token_dec] , srt))
    return tr, mask, srt
def tf_encode_t2(tr, mask, srt):
    return tf.py_function(encode_t2, [tr, mask, srt], [tf.int64, tf.float32, tf.int64])

def encode_wo_mask_w_endtoken(tr, srt):
    tr = np.hstack((tr, [end_token]))
    srt = np.hstack(([start_token_dec], srt, [end_token]))
    return tr, srt

def encode_wo_mask(tr, srt):
    srt = np.hstack(([start_token_dec] , srt))
    return tr, srt

def tf_encode_wo_mask_w_endtoken(tr, srt):
    return tf.py_function(encode_wo_mask_w_endtoken, [tr, srt], [tf.int64, tf.int64])

def tf_encode_wo_mask(tr, srt):
    return tf.py_function(encode_wo_mask, [tr, srt], [tf.int64, tf.int64])

if args.att_sup:
    train_dataset_2 = Train_dataset_2.map(tf_encode_t2)
else:
    train_dataset_2 = Train_dataset_2.map(tf_encode_wo_mask)
train_dataset_2 = train_dataset_2.cache()
train_dataset_2 = train_dataset_2.shuffle(Train_exmp.shape[0]).batch(BATCH_SIZE)
train_dataset_2 = train_dataset_2.prefetch(tf.data.experimental.AUTOTUNE)

if args.att_sup:
    val_dataset_2 = Val_dataset_2.map(tf_encode_t2)
else:
    val_dataset_2 = Val_dataset_2.map(tf_encode_wo_mask)
val_dataset_2 = val_dataset_2.batch(BATCH_SIZE)

exmp = next(iter(train_dataset_2))
print(exmp[0][0,:])
print(exmp[-1][0,:])
if args.att_sup:
    print(exmp[1][0,:,:])

exmp = next(iter(val_dataset_2))
print(exmp[0][0,:])
print(exmp[-1][0,:])
if args.att_sup:
    print(exmp[1][0,:,:])


# In[5]:


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, state_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, state_size, seq_len, d_model)
  ])


def generate_similarity_score(q, k, NN, make_sym):
    seq_len_k = tf.shape(k)[-2]
    seq_len_q = tf.shape(q)[-2]
    q_inp = tf.tile(q[:,:,:,tf.newaxis,:],[1,1,1,seq_len_k,1])
    k_inp = tf.tile(k[:,:,tf.newaxis,:,:],[1,1,seq_len_q,1,1])
    combined = tf.concat([q_inp, k_inp], -1)
    sim_weights = NN(combined)
    if make_sym:
        combined_2 = tf. concat([k_inp, q_inp],-1)
        sim_weights_2 = NN(combined_2)
        sim_weights += sim_weights_2
    return tf.squeeze(sim_weights, [-1])

def scaled_general_attention(q, k, v, mask, NN, make_sym):
    scaled_attention_logits = generate_similarity_score(q, k, NN, make_sym)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


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





# In[6]:

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





class Attention(tf.keras.layers.Layer):
    def __init__(self, d_model, make_sym, shared_proj):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.make_sym = make_sym
        self.shared_proj = shared_proj
        if shared_proj:
            self.w = tf.keras.layers.Dense(d_model)
        else:
            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
    
    def call(self, v, k, q, mask, NN):
        batch_size = tf.shape(q)[0]
        
        if self.shared_proj:
            q = self.w(q)  # (batch_size, state_size, seq_len, d_model)
            k = self.w(k)  # (batch_size, state_size, seq_len, d_model)
            v = self.w(v)  # (batch_size, state_size, seq_len, d_model)
        else:
            q = self.wq(q)  # (batch_size, state_size, seq_len, d_model)
            k = self.wk(k)  # (batch_size, state_size, seq_len, d_model)
            v = self.wv(v)  # (batch_size, state_size, seq_len, d_model)
    
        scaled_attention, attention_weights = scaled_general_attention(
            q, k, v, mask, NN, self.make_sym)
        output = self.dense(scaled_attention)  # (batch_size, state_size, seq_len_q, d_model)
        return output, attention_weights


# In[7]:


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff,  mha_att, make_sym=None, shared_proj=None, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        if mha_att:
            self.mha = MultiHeadAttention(d_model, 1)
        else:
            self.mha = Attention(d_model, make_sym, shared_proj)
        
        self.mha_att = mha_att
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask, NN=None):
        if self.mha_att:
            attn_output, attention_weights = self.mha(x, x, x, mask)
        else:
            attn_output, attention_weights = self.mha(x, x, x, mask, NN)  # (batch_size, state_size, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(res_ratio*x + (2-res_ratio)*attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(res_ratio*out1 + (2-res_ratio)*ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# In[8]:


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, mha_att, make_sym=None, shared_proj=None, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        if mha_att:
            self.mha1 = MultiHeadAttention(d_model, 1)
            self.mha2 = MultiHeadAttention(d_model, 1)
        else:  
            self.mha1 = Attention(d_model, make_sym, shared_proj)
            self.mha2 = Attention(d_model, make_sym, shared_proj)
        
        self.mha_att = mha_att
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, com_mask, dec_mask, NN=None):
        # enc_output.shape == (batch_size, state_size, 1, d_model)
        
        if self.mha_att:
            attn1, attn_weights_block1 = self.mha1(x, x, x, com_mask)
        else:
            attn1, attn_weights_block1 = self.mha1(x, x, x, com_mask, NN)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1((2-res_ratio)*attn1 + res_ratio*x)
        
        if self.mha_att:
            attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, dec_mask)
        else:
            attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, dec_mask, NN)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2((2-res_ratio)*attn2 + res_ratio*out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3((2-res_ratio)*ffn_output + res_ratio*out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# In[9]:


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
    binary_size += 2   
    pow_base = tf.reverse(tf.range(binary_size, dtype=tf.int64),[-1])
    out = tf.bitwise.bitwise_and(tf.expand_dims(x, -1), tf.bitwise.left_shift(tf.constant(1, dtype=tf.int64), pow_base))
    out = tf.cast(tf.greater(out, 0), tf.float32)
    return out
def back2int(x): ##x: int64
    binary_size = x.shape[-1]
    pow_base = tf.bitwise.left_shift(tf.constant(1, dtype=tf.int64),tf.reverse(tf.range(binary_size, dtype=tf.int64),[-1])[:, tf.newaxis])
    out = tf.matmul(x, pow_base)
    return tf.squeeze(out,-1)

# In[10]:


def add_pos(x, mask, pos_enc):
    mask = tf.squeeze(1-mask, -2)
    x_ind = tf.math.cumsum(mask, -1)
    return x+tf.gather_nd(pos_enc, tf.cast(x_ind[:,:,:,tf.newaxis], dtype=tf.int64))


# In[11]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, binary_size, dff, pos,  mha_att, bin_enc, use_emb=None, make_sym=None, shared_proj=None,
               rate=0.1):
        super(Encoder, self).__init__()

        self.binary_size = binary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(pos, self.d_model)
        self.bin_enc = bin_enc
        self.use_emb = use_emb
        

        self.enc_layers = [EncoderLayer(d_model, dff,  mha_att, make_sym, shared_proj, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask, use_pos, NN=None, emb=None):
        if self.bin_enc:
            if self.use_emb:
                x = binary_encoding(x, self.binary_size)
                x = emb(x)
            else:
                x = binary_encoding(x, d_model-2)
        else:
            x = tf.one_hot(x, 2 ** self.binary_size+3, axis=-1)
            x = emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if use_pos:
            x = add_pos(x, mask, self.pos_encoding)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask, NN)

        return x  # (batch_size, state_size, seq_len, d_model)


# In[12]:


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, mha_att, bin_enc, use_emb=None, make_sym=None, shared_proj=None,
               rate=0.1):
        super(Decoder, self).__init__()

        self.binary_size = binary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(pos, d_model)
        self.bin_enc = bin_enc
        self.use_emb = use_emb

        self.dec_layers = [DecoderLayer(d_model, dff, mha_att, make_sym, shared_proj, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, 
           com_mask, dec_mask, use_pos, NN=None, emb=None):

        attention_weights = {}
        if self.bin_enc:
            if self.use_emb:
                x = binary_encoding(x, self.binary_size)
                x = emb(x)
            else:
                x = binary_encoding(x, d_model-2)
        else:
            x = tf.one_hot(x, 2 ** self.binary_size+3, axis=-1)
            x = emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if use_pos:
            x += tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 com_mask, dec_mask, NN)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, state_size, seq_len, d_model)
        return x, attention_weights


# In[13]:


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, 
               target_vocab_size, mha_att, bin_enc, use_emb, use_pos, out_num, out_pos, make_sym=None, shared_proj=None, rate=0.1):
        super(Transformer, self).__init__()
        
        self.num_layers = num_layers
        if mha_att:
            self.NN = None
        else:
            self.NN = point_wise_feed_forward_network(1, dff)
        self.encoder = Encoder(num_layers, d_model, binary_size, dff, pos, mha_att, bin_enc, use_emb, make_sym, shared_proj, rate)
        self.decoder = Decoder(num_layers, d_model, binary_size, dff, pos,  mha_att, bin_enc, use_emb, make_sym, shared_proj, rate)
        if bin_enc:
            if use_emb:
                self.emb = tf.keras.layers.Dense(d_model, use_bias=False)
            else:
                self.emb = None
        else:
            self.emb = tf.keras.layers.Dense(d_model, use_bias=False)
        self.use_pos = use_pos
        self.out_num = out_num
        self.out_pos = out_pos
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, enc_padding_mask, 
           com_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask, self.use_pos, self.NN, self.emb)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, com_mask, dec_padding_mask, self.use_pos, self.NN, self.emb)

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


# In[14]:


transformer_2 = Transformer(num_layers, d_model, binary_size, dff, pos_2, target_vocab_size, mha_att, bin_enc, use_emb, USE_positioning_2, out_num_2, out_pos_2, make_sym, shared_proj, dropout_rate)


# In[15]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# In[16]:


learning_rate = CustomSchedule(d_model)

optimizer_2 = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


# In[17]:


checkpoint_path_2 = current_dir + "model_2"
if not reload_from_dir_2:
    subprocess.call("mkdir {}".format(checkpoint_path_2), shell=True)

ckpt_2 = tf.train.Checkpoint(transformer_2=transformer_2,
                           optimizer_2=optimizer_2)

ckpt_manager_2 = tf.train.CheckpointManager(ckpt_2, checkpoint_path_2, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager_2.latest_checkpoint:
    ckpt_2.restore(ckpt_manager_2.latest_checkpoint)
    print ('Model_2 checkpoint restored!!')


# In[18]:


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, pad_token), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[-1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
     


# In[19]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
d_loss = tf.keras.metrics.Mean(name='d_loss')
if args.variants != "all_mod_one_hot_output": 
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    d_accuracy = tf.keras.metrics.Accuracy(name='d_accuracy')
else:
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    d_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='d_accuracy')





# In[20]:

def loss_function(real, pred, sample_weight):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred, sample_weight)
    return tf.reduce_mean(loss_)


def loss_function_bin(real, pred, sample_weight):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred, sample_weight)
    return tf.reduce_mean(loss_)


def loss_pos(real, pred, sample_weight):
    loss_obj = tf.keras.losses.CategoricalCrossentropy()
    loss_ = loss_obj(real, pred, sample_weight)
    return loss_


# In[32]:


summary_writer_2 = tf.summary.create_file_writer(current_dir + 'logs')
@tf.function
def train_step_2(inp_list):
    if args.att_sup:
        inp, msk_real, tar = inp_list ##### msk_real: (batch, seq_len, seq_len)
    else:
        inp, tar = inp_list
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    enc_inp = inp[:, tf.newaxis, :]
    tar_inp = tar_inp[:, tf.newaxis, :]
    tar_real = tar_real[:, tf.newaxis, :]
    
    mask = tf.cast(tf.math.equal(tar_real, pad_token), dtype=tf.float32) #### (batch, 1, seq_len)
    if args.variants != "all_mod_one_hot_output":
        tar_bin = binary_encoding(tar_real, binary_size)*(1-mask[:,:,:,tf.newaxis])
    else:
        tar_real = tf.where(tf.equal(tar_real, inf), tf.cast(2 ** binary_size, tf.int64), tar_real)
        tar_real = tf.where(tf.equal(tar_real, end_token), tf.cast(2 ** binary_size+1, tf.int64), tar_real)
    
    with tf.GradientTape() as tape:
        predictions, _, predicted_mask = transformer_2(enc_inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        if args.variants != "all_mod_one_hot_output":
            loss = loss_function_bin(tar_bin, predictions+mask[:,:,:,tf.newaxis]*-1e9, sample_weight=1-mask[:,:,:,tf.newaxis])
        else:
            loss = loss_function(tf.cast(tar_real, dtype=tf.float32)*(1-mask), predictions*(1-mask[:,:,:,tf.newaxis]), sample_weight=1-mask[:,:,:,tf.newaxis])
        if args.att_sup:
            mask_tr = tf.transpose(mask, perm=[0,2,1])
            msk_masked = msk_real*(1-mask_tr) + mask_tr*tf.one_hot(tf.ones((tar_real.shape[0], seq_len_p2), dtype=tf.int64),seq_len_p2)
            predicted_mask_masked = predicted_mask*(1-mask_tr) + mask_tr*tf.one_hot(tf.ones((tar_real.shape[0], seq_len_p2), dtype=tf.int64),seq_len_p2)
            loss += loss_pos(msk_masked, predicted_mask_masked, sample_weight=1-mask_tr)
    gradients = tape.gradient(loss, transformer_2.trainable_variables)
    optimizer_2.apply_gradients(zip(gradients, transformer_2.trainable_variables))
    train_loss(loss)
    tf.summary.scalar("loss", train_loss.result(), step=optimizer_2.iterations)
    if args.variants != "all_mod_one_hot_output":
        pred_binary = tf.cast(tf.greater(predictions, 0), tf.int64)
        pred_binary = tf.cast(back2int(pred_binary), dtype=tf.float32)
        train_accuracy(tf.cast(tar_real, dtype=tf.float32), pred_binary, sample_weight=1-mask)
    else:
        train_accuracy(tf.cast(tar_real, dtype=tf.float32), predictions, sample_weight=1-mask)

# In[33]:


def eval_val_2(dataset, test_mod=False, name='Validation'):
    d_loss.reset_states()
    d_accuracy.reset_states()
    for inp_list in dataset:
        if args.att_sup and (not test_mod):
            inp, msk_real, tar = inp_list ##### msk_real: (batch, seq_len, seq_len)
        else:
            inp, tar = inp_list
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        enc_inp = inp[:, tf.newaxis, :]
        tar_inp = tar_inp[:, tf.newaxis, :]
        tar_real = tar_real[:, tf.newaxis, :]
        if args.variants == "all_mod_one_hot_output":
            tar_real = tf.where(tf.equal(tar_real, inf), tf.cast(2 ** binary_size, tf.int64), tar_real)
            tar_real = tf.where(tf.equal(tar_real, end_token), tf.cast(2 ** binary_size+1, tf.int64), tar_real)

        predictions, _, predicted_mask = transformer_2(enc_inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        mask = tf.cast(tf.math.equal(tar_real, pad_token), dtype=tf.float32)
        if args.variants != "all_mod_one_hot_output":
            loss = loss_function_bin(binary_encoding(tar_real, binary_size)*(1-mask[:,:,:,tf.newaxis]), predictions+mask[:,:,:,tf.newaxis]*-1e9, sample_weight=1-mask[:,:,:,tf.newaxis])
        else:
            loss = loss_function(tf.cast(tar_real, dtype=tf.float32)*(1-mask), predictions*(1-mask[:,:,:,tf.newaxis]), sample_weight=1-mask[:,:,:,tf.newaxis])
        if args.att_sup and (not test_mod):
            mask_tr = tf.transpose(mask, perm=[0,2,1])
            msk_masked = msk_real*(1-mask_tr) + mask_tr*tf.one_hot(tf.ones((tar_real.shape[0], seq_len_p2), dtype=tf.int64),seq_len_p2)
            predicted_mask_masked = predicted_mask*(1-mask_tr) + mask_tr*tf.one_hot(tf.ones((tar_real.shape[0], seq_len_p2), dtype=tf.int64),seq_len_p2)
            loss += loss_pos(msk_masked, predicted_mask_masked, sample_weight=1-mask_tr)
        d_loss(loss)
        if args.variants != "all_mod_one_hot_output":
            pred_binary = tf.cast(tf.greater(predictions, 0), tf.int64)
            pred_binary = tf.cast(back2int(pred_binary), dtype=tf.float32)
            d_accuracy(tf.cast(tar_real, dtype=tf.float32), pred_binary, sample_weight=1-mask)
        else:
            d_accuracy(tf.cast(tar_real, dtype=tf.float32), predictions, sample_weight=1-mask)
    print('{}_Loss {:.4f} {}_Accuracy {:.4f}'.format(name, d_loss.result(), name, d_accuracy.result()))
    return d_accuracy.result()


# In[ ]:


for epoch in range(EPOCHS_2):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    with summary_writer_2.as_default():
        for (batch, inp_list) in enumerate(train_dataset_2):
            
            train_step_2(inp_list)
            
            if batch % 500 == 0:
                print('Epoch {} Batch {}:\nTraining_Loss {:.4f} Training_Accuracy {:.4f}'.format(epoch+1, batch, train_loss.result(), train_accuracy.result()))
                eval_val_2(val_dataset_2)
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager_2.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch + 1, ckpt_save_path))
        print('Epoch {}:\nTraining_Loss {:.4f} Training_Accuracy {:.4f}'.format(epoch+1, train_loss.result(), train_accuracy.result()))
        eval_val_2(val_dataset_2)
        print('Time taken for 1 epoch: {} secs\n'.format(time.time()-start))
        


# In[ ]:


max_seq_len = 100
max_seq_step = 60
if args.evaluation == "random":
    test_size_random = 100
elif args.evaluation == "mixed":
    test_size_random = 60
    test_size_small_steps = 10
else:
    max_seq_step = 50
    test_size_small_steps = 25
acc = []
for i in range(8, max_seq_len+1):
    if args.evaluation != "hard":
        test_random = np.random.choice(range(num_max_2), (test_size_random, i))
    if args.evaluation != "random":
        test_special = []
        if tf.less_equal(i, max_seq_step):
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max_2-(i-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, i])
                test += np.arange(0, i*step, step)
                test_special.append(test)
        else:
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max_2-(max_seq_step-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, max_seq_step])
                test += np.arange(0, max_seq_step*step, step)
                padded_random = np.random.choice(range(num_max_2), (test_size_small_steps, i-max_seq_step))
                test = np.concatenate((test, padded_random), axis=-1)
                test_special.append(test)
        test_special = np.concatenate(test_special, axis=0)
        np.apply_along_axis(np.random.shuffle, 1, test_special)
        if args.evaluation == "mixed":
            test = np.concatenate([test_random, test_special], axis=0)
        else:
            test = test_special
    else:
        test = test_random
    Sorted_test = np.sort(test)
    test_dataset = tf.data.Dataset.from_tensor_slices((test, Sorted_test))
    test_dataset = test_dataset.map(tf_encode_wo_mask_w_endtoken)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    acc.append(eval_val_2(test_dataset, test_mod=True, name="Test_seq{}".format(i)))


# In[ ]:


np.save("{}acc".format(current_dir), acc)
plt.figure()
plt.plot(range(8, len(acc)+8), acc)
plt.xlabel('seq_len')
plt.ylabel('acc')
filename='acc_plot.png'
plt.savefig(current_dir+filename)

