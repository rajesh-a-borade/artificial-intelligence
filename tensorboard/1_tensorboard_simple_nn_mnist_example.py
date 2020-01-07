# Author -  rajesh.a.borade
# Description - Simple NN to classify MNIST images

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
 
TOTAL_ARRAY_SIZE = 784
NUM_OF_CLASSES = 10
BATCH_SIZE = 100
EPOCHS = 10
TFBOARD_LOG_DIR = './data/1/'

x = tf.placeholder('float', [None, TOTAL_ARRAY_SIZE])
y = tf.placeholder('float', [None, NUM_OF_CLASSES])
mnist = input_data.read_data_sets("./data/", one_hot=True)
 
def neural_network_model(data) :
    with tf.name_scope('neural_network_model'):     
        output_layer =  {'weights' : tf.Variable(tf.random_normal([TOTAL_ARRAY_SIZE, NUM_OF_CLASSES])),
                            'biases' : tf.Variable(tf.random_normal([NUM_OF_CLASSES]))
                        }
        output = tf.matmul(data, output_layer['weights']) + output_layer['biases']
        tf.summary.histogram("weights", output_layer['weights'])
        tf.summary.histogram("biases", output_layer['biases'])
        tf.summary.histogram("activations", output)
        return output
 
def train_neural_network(x) :
     
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    writer = tf.summary.FileWriter(TFBOARD_LOG_DIR)
    summ = tf.summary.merge_all()

    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        test_accuracy = 0
        for epoch in range(EPOCHS) :
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/BATCH_SIZE)) :
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c, _summ_run = sess.run([optimizer, cost, summ], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, ' completed out of ', EPOCHS, ' loss : ', epoch_loss)
            train_accuracy = accuracy.eval({x:epoch_x, y:epoch_y})
            print('Train Accuracy : ', train_accuracy)
            writer.add_summary(_summ_run, epoch)
        
        test_accuracy = accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
        print('Test Accuracy : ', test_accuracy)
        
        writer.add_graph(sess.graph)    

train_neural_network(x)
