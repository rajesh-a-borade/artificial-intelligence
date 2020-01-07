# Author -  rajesh.a.borade
# Description - Simple NN to classify MNIST images

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
 
# ---------------- Hyperparameters ----------------

TOTAL_ARRAY_SIZE = 784
NUM_OF_CLASSES = 10
BATCH_SIZE = 100
EPOCHS = 10

# ---------- Variables, placeholders and Data ----------

x = tf.placeholder('float', [None, TOTAL_ARRAY_SIZE])
y = tf.placeholder('float', [None, NUM_OF_CLASSES])
mnist = input_data.read_data_sets("./data/", one_hot=True)

# ---------------- Model ----------------

def neural_network_model(data) :
    with tf.name_scope('neural_network_model'):     
        output_layer =  {'weights' : tf.Variable(tf.random_normal([TOTAL_ARRAY_SIZE, NUM_OF_CLASSES])),
                            'biases' : tf.Variable(tf.random_normal([NUM_OF_CLASSES]))
                        }
        output = tf.matmul(data, output_layer['weights']) + output_layer['biases']
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
