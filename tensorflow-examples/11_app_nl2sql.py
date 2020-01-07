# Author - Rajesh Borade
# Description - English to SQL generator using seq2sq model with Tensorflow library

# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd

# ------------------------ Functions ------------------------


def decode_to_text(bytes_string):
    return "".join(map(chr, bytes_string)).replace('\x00', '').replace('\n', '').replace('\xc3', '').replace('\x81','').replace('\xe3','').replace('\x8b','').replace('\xb3','')


def encode_to_ascii(_string):
    _values = []
    _string = [m.encode('utf-8') for m in _string]
    for ii in range(len(_string)) :
        try :
            _values.append(ord(_string[ii]))
        except:
            pass
    return _values


PAD = 0
def do_padding(_arr, total_length):
    while(len(_arr) < total_length):
        _arr.append(PAD)
    return _arr

# ------------------------------------------------------------------


vocab_size = 256

target_vocab_size = vocab_size 
learning_rate = 0.03

INPUT_LENGTH = 40
OUTPUT_LENGTH = 70
TOTAL_EPOCHS = 2000
batch_size = 1

buckets=[(INPUT_LENGTH, OUTPUT_LENGTH)]


#-----------------------DATA PREP-----------------------------------------

in_arr = []
out_arr = []
test_in_arr = []
test_out_arr = []

df = pd.read_csv('./data/train10.tsv', sep='\t', names=['nl', 'sql'], header=None)
plain_arr = df['nl']
enc_arr = df['sql']

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

df = pd.read_csv('./data/test10.tsv', sep='\t', names=['nl', 'sql'], header=None)
test_plain_arr = df['nl']
test_enc_arr = df['sql']

for ii in range(len(test_plain_arr)):
    input_string = test_plain_arr[ii]
    target_string = test_enc_arr[ii]
    # print(input_string)
    # print(target_string)
    x_values = encode_to_ascii(input_string)
    x_values = do_padding(x_values, INPUT_LENGTH)
    y_values = encode_to_ascii(target_string)
    y_values = do_padding(y_values, OUTPUT_LENGTH)
    test_in_arr.append(x_values)
    test_out_arr.append(y_values)

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
            # cell = single_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(size), 0.1)
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                    return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                    encoder_inputs, decoder_inputs, cell,
                                    num_encoder_symbols=source_vocab_size,
                                    num_decoder_symbols=target_vocab_size,
                                    embedding_size=size,
                                    feed_previous=do_decode)
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in range(buckets[-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))
            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name='weights{0}'.format(i)))
            targets = [self.decoder_inputs[i] for i in range(len(self.decoder_inputs) )]
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                            self.encoder_inputs, self.decoder_inputs, targets,
                            self.target_weights, buckets,
                            lambda x, y: seq2seq_f(x, y, False))
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
    
    model= Seq2Seq(vocab_size, target_vocab_size, buckets, size=5, num_layers=1, batch_size=batch_size)
    session.run(tf.global_variables_initializer())
    
    # training
    print("***************************************")
    print("************** TRAINING ***************")
    print("***************************************")
    train_step = 0
    _iteration = 0
    while _iteration < TOTAL_EPOCHS:
            if(train_step == len(in_arr)) :
                train_step = 0
            input_data = in_arr[train_step]
            target_data = out_arr[train_step]
            # train
            model.step(session, input_data, target_data, target_weights, test=False)
            # test
            losses, outputs = model.step(session, input_data, target_data, target_weights, test=True)
            text_bytes = np.argmax(outputs, axis=2) 
            word = decode_to_text(text_bytes)
            try:
                if losses < 1.0 :
                    print("_iteration %d, loss %f, \n NL ==> %s, \nSQL ==> %s, \nGEN ==> %s" % (_iteration, losses, plain_arr[train_step], enc_arr[train_step], word))
            except:
                pass
            train_step = train_step + 1
            _iteration = _iteration + 1
    
    # testing
    print("***************************************")
    print("************** TESTING ****************")
    print("***************************************")
    test_step = 0
    while test_step < len(test_in_arr):
        input_data = test_in_arr[test_step]
        target_data = test_out_arr[test_step]
        # test
        losses, outputs = model.step(session, input_data, target_data, target_weights, test=True)
        text_bytes = np.argmax(outputs, axis=2)
        _gen_text = decode_to_text(text_bytes)
        try:
            if losses < 1.0 :
                print("test_step %d, loss %f, \n NL ==> %s, \nSQL ==> %s, \nGEN ==> %s" % (test_step, losses, test_plain_arr[test_step], test_enc_arr[test_step], _gen_text))
        except:
            pass
        test_step = test_step + 1