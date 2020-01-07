# Author -  rajesh.a.borade
# Description - Sentiment Classification

import tensorflow as tf
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import codecs
import pandas as pd

# -------- Data preparation Vectorization--------

stemmer = LancasterStemmer()

def prepare_dataset(fileName) :
    training_data_normal_json = []
    df = pd.read_csv(fileName, sep=":", header=None)
    df = df.replace(np.nan, '', regex=True)
    class_arr = df[df.columns[0]]
    sentence_arr = df[df.columns[1]]
    for i in range(len(class_arr)):
        training_data_normal_json.append({"class":"{}".format(class_arr[i]), "sentence":"{}".format(sentence_arr[i])})
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    for pattern in training_data_normal_json:
        w = nltk.word_tokenize(pattern['sentence'])
        words.extend(w)
        documents.append((w, pattern['class']))
        if pattern['class'] not in classes:
            classes.append(pattern['class'])
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = list(set(words))
    classes = list(set(classes))
    training_vector_x = []
    training_vector_y = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        training_vector_x.append(bag)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training_vector_y.append(output_row)
    return words, classes, training_vector_x, training_vector_y, training_data_normal_json

def find_largest_in(res_arr):
    m = max(res_arr)
    index_of = [i for i, j in enumerate(res_arr) if j == m]
    return index_of

def tokenize(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def convert_to_vector(sentence, words):
    sentence_words = tokenize(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

file_name = './data/sentiment/sentiment_data.txt'
words, classes, training_vector_x, training_vector_y, training_data_normal_json = prepare_dataset(file_name)

test_vector_x = training_vector_x[39:44]
test_vector_y = training_vector_y[39:44]
training_vector_x = training_vector_x[0:39]
training_vector_y = training_vector_y[0:39]
'''
print('--------------------------------')
counter = 0
for ii in range(len(training_vector_x)) :
    print(training_data_normal_json[counter])
    print(training_vector_x[ii])
    print(training_vector_y[ii])
    counter += 1
print('--------------------------------')
for ii in range(len(test_vector_x)) :
    print(training_data_normal_json[counter])
    print(test_vector_x[ii])
    print(test_vector_y[ii])
    counter += 1
print('--------------------------------')
'''
# ---------------- Hyperparameters ----------------

TOTAL_ARRAY_SIZE = len(words)
NUM_OF_CLASSES = len(classes)
HIDDEN_LAYER_NODES = 150
DROPOUT_LAYER_NODES = 150
EPOCHS = 1500

# ---------- Variables, placeholders and Data ----------

x = tf.placeholder('float', [1, TOTAL_ARRAY_SIZE])
y = tf.placeholder('float', [1, NUM_OF_CLASSES])
pkeep = tf.placeholder(tf.float32)

# ---------------- Model ----------------

def neural_network_model(data) :
    with tf.name_scope('neural_network_model'):
        in_layer =  {'weights' : tf.Variable(tf.random_normal([TOTAL_ARRAY_SIZE, HIDDEN_LAYER_NODES])),
                            'biases' : tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]))
                        }
        dropout_layer =  {'weights' : tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES, DROPOUT_LAYER_NODES])),
                            'biases' : tf.Variable(tf.random_normal([DROPOUT_LAYER_NODES]))
                        }
        output_layer =  {'weights' : tf.Variable(tf.random_normal([DROPOUT_LAYER_NODES, NUM_OF_CLASSES])),
                            'biases' : tf.Variable(tf.random_normal([NUM_OF_CLASSES]))
                        }
        # 1 x 52 * 52 x 7  = 1 x 7
        input_layer = tf.matmul(data, in_layer['weights']) + in_layer['biases']
        input_layer = tf.nn.tanh(input_layer)
        d = tf.matmul(input_layer, dropout_layer['weights']) + dropout_layer['biases']
        d = tf.nn.tanh(d)
        d = tf.nn.dropout(d, pkeep)
        output = tf.matmul(d, output_layer['weights']) + output_layer['biases']
        output = tf.nn.relu(output)
        return output

# ---------------- Train and Test ----------------

def run_neural_network(x) :
     
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCHS) :
            epoch_loss = 0
            for ii in range(len(training_vector_x)) :
                epoch_x = training_vector_x[ii]
                epoch_y = training_vector_y[ii]
                epoch_x = np.expand_dims(epoch_x, axis=1)
                epoch_x = np.transpose(epoch_x)
                epoch_y = np.asarray(epoch_y)
                epoch_y = np.expand_dims(epoch_y, axis=1)
                epoch_y = np.transpose(epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y, pkeep:0.5})
                epoch_loss += c
            print('Epoch ', epoch, ' completed out of ', EPOCHS, ' loss : ', epoch_loss)
        
        saver = tf.train.Saver()
        saver.save(sess, './data/model/sentiment_model.ckpt')
        total_test_accuracy = 0
        for ii in range(len(test_vector_x)) :
            epoch_x = test_vector_x[ii]
            epoch_y = test_vector_y[ii]
            epoch_x = np.expand_dims(epoch_x, axis=1)
            epoch_x = np.transpose(epoch_x)
            epoch_y = np.asarray(epoch_y)
            epoch_y = np.expand_dims(epoch_y, axis=1)
            epoch_y = np.transpose(epoch_y)
            test_accuracy = accuracy.eval(feed_dict = {x:epoch_x, y:epoch_y, pkeep:1.0})
            print('Test Accuracy : ', test_accuracy)
            total_test_accuracy += test_accuracy
        avg_test_accuracy = (total_test_accuracy/len(test_vector_x))
        print('Avg. Test Accuracy : ', avg_test_accuracy)        

run_neural_network(x)

'''
def predict(test_text):

    tf.reset_default_graph()
    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
         
        sess.run(tf.global_variables_initializer())
        test_accuracy = 0
        saver.restore(sess, "./data/model/sentiment_model.ckpt")
        test_vector = convert_to_vector(test_text, words)
        _x = test_vector
        _x = np.expand_dims(_x, axis=1)
        _x = np.transpose(_x)
        res = sess.run(prediction, feed_dict={x: _x})
        test_vector = convert_to_vector('i think i lost my phone', words)
        _x = np.expand_dims(test_vector, axis=1)
        _x = np.transpose(_x)
        res = sess.run(prediction, feed_dict={x: _x})
        res = np.asarray(res)
        res = res.flatten()
        res = find_largest_in(res)
        res = classes[res[0]]
        print(res)
        return res
'''