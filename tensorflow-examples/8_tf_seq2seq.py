# Author - Rajesh Borade
# Description - Simple seq2seq generator

# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd

# ------------------------ Helper Functions ------------------------

# Convert ASCII code to text
def decode_to_text(bytes):
    return "".join(map(chr, bytes)).replace('\x00', '').replace('\n', '').replace('\xc3', '').replace('\x81','').replace('\xe3','').replace('\x8b','').replace('\xb3','')

# Convert ASCII code to text
def encode_to_ascii(_string):
    _values = []
    _string = [m.encode('utf-8') for m in _string]
    for ii in range(len(_string)) :
        try :
            _values.append(ord(_string[ii]))
        except:
            pass
    return _values

# Pad the values to given sequence
PAD = 0
def do_padding(_arr, total_length):
    while(len(_arr) < total_length):
        _arr.append(PAD)
    return _arr

# ------------------------------------------------------------------

# Number of ASCII Code
vocab_size = 256
# the model selects (classify) one of 256 classes (ASCII codes) per time unit
target_vocab_size = vocab_size 
learning_rate = 0.1

INPUT_LENGTH = 35
OUTPUT_LENGTH = 65
batch_size = 1

buckets=[(INPUT_LENGTH, OUTPUT_LENGTH)]
# Because seq2seq does batch learning, it buckets input by length.

in_arr = []
out_arr = []

df = pd.read_csv("./data/encryption/aes128.txt")
plain_arr = df['plain']
enc_arr = df['enc']

for ii in range(len(plain_arr)):
    input_string = plain_arr[ii]
    target_string = enc_arr[ii]
    # print(input_string)
    # print(target_string)
    x_values = encode_to_ascii(input_string)
    x_values = do_padding(x_values, INPUT_LENGTH)
    y_values = encode_to_ascii(target_string)
    y_values = do_padding(y_values, OUTPUT_LENGTH)
    in_arr.append(x_values)
    out_arr.append(y_values)

# The number of actual valid (loss counted) number of characters 
target_weights = ([1.0]*OUTPUT_LENGTH + [0.0]*0) * batch_size

# ------------------------------------------------------------------

class Seq2Seq(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
            self.buckets = buckets
            self.batch_size = batch_size
            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size

            cell = single_cell = tf.contrib.rnn.GRUCell(size)
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

            # The seq2seq function
            # encoder_inputs: A list of ASCII codes in the input sentence.
            # decoder_inputs: A list of ASCII codes in the target sentence.
            # cell: RNN cell to use for seq2seq.
            # num_encoder_symbols, num_decoder_symbols: The number of symbols in the input sentence and the target sentence.
            # embedding_size: Size to embed each ASCII code.
            # feed_previous: Inference (true for learning / false for Inference)
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                    return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                    encoder_inputs, decoder_inputs, cell,
                                    num_encoder_symbols=source_vocab_size,
                                    num_decoder_symbols=target_vocab_size,
                                    embedding_size=size,
                                    feed_previous=do_decode)
                
            # computational graph 
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            
            # Bucket size + one as decoder input node. 
            # (One additional creation is because the target symbol is equivalent to the decoder input shifting one space)
            for i in range(buckets[-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))

            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name='weights{0}'.format(i)))
                
            # The target symbol is equivalent to the decoder input shifted by one space.
            targets = [self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs) - 1)]
            
            # Using seq2seq with buckets
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                            self.encoder_inputs, self.decoder_inputs, targets,
                            self.target_weights, buckets,
                            lambda x, y: seq2seq_f(x, y, False))
                            
            # Gradient
            self.updates = []
            self.updates.append(tf.train.AdamOptimizer(learning_rate).minimize(self.losses[0]))

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
            bucket_id=0 # Choosing bukcet to use
            encoder_size, decoder_size = self.buckets[bucket_id]

            # Input feed: encoder inputs, decoder inputs, target_weights 
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = [encoder_inputs[l]]
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = [decoder_inputs[l]]
                input_feed[self.target_weights[l].name] = [target_weights[l]]

            # Insert a value because there is one more decoder input node created.
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
            last_weight = self.target_weights[decoder_size].name
            input_feed[last_weight] = np.zeros([self.batch_size], dtype=np.int32)

            if not test:
                output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
            else:
                output_feed = [self.losses[bucket_id]]  # Loss for this batch.
                for l in range(decoder_size):  # Output logits.
                    output_feed.append(self.outputs[bucket_id][l])

            outputs = session.run(output_feed, input_feed)
            if not test:
                    return outputs[0], outputs[1] # loss
            else:
                    return outputs[0], outputs[1:] # loss, outputs.

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# with tf.Session(config=config) as session:

with tf.Session() as session:
    current_step = 0
    model= Seq2Seq(vocab_size, target_vocab_size, buckets, size=5, num_layers=1, batch_size=batch_size)
    session.run(tf.global_variables_initializer())
    _iteration = 0
    success_count = 0
    while True:
            if(current_step == len(in_arr)) :
                current_step = 0
                if (success_count == len(in_arr)) :
                    print("*** All ciphers are decrypted ***")
                    break
                else :
                    success_count = 0
                    print("*** Starting again ***")
            input_data = in_arr[current_step]
            target_data = out_arr[current_step]
            # train
            model.step(session, input_data, target_data, target_weights, test=False)
            # test
            losses, outputs = model.step(session, input_data, target_data, target_weights, test=True)
            text_bytes = np.argmax(outputs, axis=2)  # shape (12, 12, 256)
            word = decode_to_text(text_bytes)
            try:
                print("_iteration %d, losses %f, success_count %d, plain %s, cipher %s, -> seq %s" % (_iteration, losses, success_count, plain_arr[current_step], enc_arr[current_step], word))
            except:
                pass
            if word == enc_arr[current_step][1:]:
                print(">>>>> success <<<<<<<")
                success_count = success_count + 1
            current_step = current_step + 1
            _iteration = _iteration + 1

''' SAMPLE RESULTS

_iteration 0, losses 4.306795, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq
_iteration 1, losses 0.189915, success_count 0, plain abcd, cipher  qrst, -> seq r
*** Starting again ***
_iteration 2, losses 1.012399, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qqsssssssssss
_iteration 3, losses 0.133701, success_count 0, plain abcd, cipher  qrst, -> seq qqr
*** Starting again ***
_iteration 4, losses 0.297331, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq rrrvwwwx
_iteration 5, losses 0.320977, success_count 0, plain abcd, cipher  qrst, -> seq qrrvv
*** Starting again ***
_iteration 6, losses 0.856554, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qq
_iteration 7, losses 0.310849, success_count 0, plain abcd, cipher  qrst, -> seq rrrrr
*** Starting again ***
_iteration 8, losses 1.797428, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq rrrrrssrr
_iteration 9, losses 1.513077, success_count 0, plain abcd, cipher  qrst, -> seq sssssr
*** Starting again ***
_iteration 10, losses 1.362721, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qtttttt
_iteration 11, losses 0.450369, success_count 0, plain abcd, cipher  qrst, -> seq qqq
*** Starting again ***
_iteration 12, losses 1.632811, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qqttuv
_iteration 13, losses 0.391719, success_count 0, plain abcd, cipher  qrst, -> seq uuuu
*** Starting again ***
_iteration 14, losses 0.447815, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq vrrtttvvy
_iteration 15, losses 0.576941, success_count 0, plain abcd, cipher  qrst, -> seq rrrrr
*** Starting again ***
_iteration 16, losses 0.520672, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrrrxtxxx
_iteration 17, losses 0.535167, success_count 0, plain abcd, cipher  qrst, -> seq qqwwx
*** Starting again ***
_iteration 18, losses 0.357097, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qqwwwwwwy
_iteration 19, losses 0.393591, success_count 0, plain abcd, cipher  qrst, -> seq uuuu
*** Starting again ***
_iteration 20, losses 0.370067, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq yuuuuuuuy
_iteration 21, losses 0.293021, success_count 0, plain abcd, cipher  qrst, -> seq vv
*** Starting again ***
_iteration 22, losses 0.285116, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qqsttvvvy
_iteration 23, losses 0.049900, success_count 0, plain abcd, cipher  qrst, -> seq qqtt
*** Starting again ***
_iteration 24, losses 0.321211, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qqttttvyy
_iteration 25, losses 0.028217, success_count 0, plain abcd, cipher  qrst, -> seq qrss
*** Starting again ***
_iteration 26, losses 0.169508, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrsssvvyy
_iteration 27, losses 0.067625, success_count 0, plain abcd, cipher  qrst, -> seq rrsw
*** Starting again ***
_iteration 28, losses 0.114849, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrsswwvvy
_iteration 29, losses 0.046967, success_count 0, plain abcd, cipher  qrst, -> seq qrss
*** Starting again ***
_iteration 30, losses 0.152686, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstruuxyy
_iteration 31, losses 0.032329, success_count 0, plain abcd, cipher  qrst, -> seq qrtt
*** Starting again ***
_iteration 32, losses 0.130146, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qsttuuuxyy
_iteration 33, losses 0.082816, success_count 0, plain abcd, cipher  qrst, -> seq qstt
*** Starting again ***
_iteration 34, losses 0.077015, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuwwxy
_iteration 35, losses 0.036368, success_count 0, plain abcd, cipher  qrst, -> seq qrsw
*** Starting again ***
_iteration 36, losses 0.066422, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuvvxy
_iteration 37, losses 0.021327, success_count 0, plain abcd, cipher  qrst, -> seq qrss
*** Starting again ***
_iteration 38, losses 0.072914, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuvvxy
_iteration 39, losses 0.014715, success_count 0, plain abcd, cipher  qrst, -> seq qrst
>>>>> success <<<<<<<
*** Starting again ***
_iteration 40, losses 0.053671, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuwwxy
_iteration 41, losses 0.011267, success_count 0, plain abcd, cipher  qrst, -> seq qrst
>>>>> success <<<<<<<
*** Starting again ***
_iteration 42, losses 0.057543, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuwwyy
_iteration 43, losses 0.008682, success_count 0, plain abcd, cipher  qrst, -> seq qrst
>>>>> success <<<<<<<
*** Starting again ***
_iteration 44, losses 0.057729, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuwwxy
_iteration 45, losses 0.007610, success_count 0, plain abcd, cipher  qrst, -> seq qrst
>>>>> success <<<<<<<
*** Starting again ***
_iteration 46, losses 0.051057, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuvwxy
>>>>> success <<<<<<<
_iteration 47, losses 0.022098, success_count 1, plain abcd, cipher  qrst, -> seq qrsu
*** Starting again ***
_iteration 48, losses 0.047504, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrsuuvwxy
_iteration 49, losses 0.007130, success_count 0, plain abcd, cipher  qrst, -> seq qrst
>>>>> success <<<<<<<
*** Starting again ***
_iteration 50, losses 0.054787, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstvvwxy
_iteration 51, losses 0.003985, success_count 0, plain abcd, cipher  qrst, -> seq qrst
>>>>> success <<<<<<<
*** Starting again ***
_iteration 52, losses 0.035684, success_count 0, plain abcdefghi, cipher  qrstuvwxy, -> seq qrstuvwxy
>>>>> success <<<<<<<
_iteration 53, losses 0.005776, success_count 1, plain abcd, cipher  qrst, -> seq qrst
>>>>> success <<<<<<<
*** All ciphers are decrypted ***

'''