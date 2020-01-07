# Author -  rajesh.a.borade
# Description - Convolutional NN to classify MNIST images

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# ---------------- Hyperparameters ----------------

TOTAL_ARRAY_SIZE = 784
HIDDEN_LAYER_1_NODES_32 = 32
HIDDEN_LAYER_2_NODES_64 = 64
HIDDEN_LAYER_3_NODES_128 = 128
NUM_OF_CLASSES = 10
BATCH_SIZE = 100
EPOCHS = 10
DROPOUT = 0.75

# ---------- Variables, placeholders and Data ----------

x = tf.placeholder(tf.float32, [None, TOTAL_ARRAY_SIZE])
y = tf.placeholder('float', [None, NUM_OF_CLASSES])
mnist = input_data.read_data_sets("./data/", one_hot=True)
 
# ---------------- Model ----------------

def convolution_with_relu(_x, _w, _b, strides=1):
    _x = tf.nn.conv2d(_x, _w, strides=[1, strides, strides, 1], padding='SAME')
    _x = tf.nn.bias_add(_x, _b)
    return tf.nn.relu(_x)

def max_pool(_x, _k=2):
    return tf.nn.max_pool(_x, ksize=[1, _k, _k, 1], strides=[1, _k, _k, 1], padding='SAME')

def convolutional_neural_network(x) :
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, HIDDEN_LAYER_1_NODES_32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, HIDDEN_LAYER_1_NODES_32, HIDDEN_LAYER_2_NODES_64])),
        # fully connected, 7*7*64 inputs, 128 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*HIDDEN_LAYER_2_NODES_64, HIDDEN_LAYER_3_NODES_128])),
        # 128 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([HIDDEN_LAYER_3_NODES_128, NUM_OF_CLASSES]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([HIDDEN_LAYER_1_NODES_32])),
        'bc2': tf.Variable(tf.random_normal([HIDDEN_LAYER_2_NODES_64])),
        'bd1': tf.Variable(tf.random_normal([HIDDEN_LAYER_3_NODES_128])),
        'out': tf.Variable(tf.random_normal([NUM_OF_CLASSES]))
    }     
    x = tf.reshape(x, shape=[-1, 28, 28, 1]) 
    conv1 = convolution_with_relu(x, weights['wc1'], biases['bc1'])
    conv1 = max_pool(conv1, _k=2) 
    conv2 = convolution_with_relu(conv1, weights['wc2'], biases['bc2'])
    conv2 = max_pool(conv2, _k=2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, DROPOUT)
    output = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return output
 
# ---------------- Train and Test ----------------

def run_neural_network(x) :
     
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        test_accuracy = 0
        for epoch in range(EPOCHS) :
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/BATCH_SIZE)) :
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, ' completed out of ', EPOCHS, ' loss : ', epoch_loss)
            train_accuracy = accuracy.eval({x:epoch_x, y:epoch_y})
            print('Train Accuracy : ', train_accuracy)
        
        test_accuracy = accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
        print('Test Accuracy : ', test_accuracy)
 
run_neural_network(x)
