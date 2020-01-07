# Author - Rajesh Borade
# Desc - Simple gan with MNIST dataset


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('./data/mnist/')
batch_size = 64

'''
  Helper functions for network
'''
def binarize(img):
    return (np.random.uniform(size=img.shape) < img).astype(np.float32)
def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
def get_bias(shape):
    return tf.Variable(tf.constant(shape=shape, value=0.1))

def generator(g_input):
    with tf.name_scope('G'):
        l1_size, l2_size = 1200, 1200
        g_w1 = get_weights([28*28, l1_size])
        g_b1 = get_bias([l1_size])
        g_w2 = get_weights([l1_size, l2_size])
        g_b2 = get_bias([l2_size])
        g_w3 = get_weights([l2_size, 28*28])
        g_b3 = get_bias([28*28])

        l1 = tf.nn.relu(tf.add(tf.matmul(X, g_w1), g_b1))
        l2 = tf.nn.relu(tf.add(tf.matmul(l1, g_w2), g_b2))
        logits = tf.sigmoid(tf.add(tf.matmul(l2, g_w3), g_b3))
        return logits
    
class Discriminator():
    def __init__(self):
        with tf.name_scope('D'):
            d_l1_size, d_l2_size = 500, 100
            self.d_w1 = get_weights([28*28, d_l1_size])
            self.d_b1 = get_bias([d_l1_size])
            self.d_w2 = get_weights([d_l1_size, d_l2_size])
            self.d_b2 = get_bias([d_l2_size])
            self.d_w3 = get_weights([d_l2_size, 1])
            self.d_b3 = get_bias([1])

    def network(self, d_input):
        d_l1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(d_input, self.d_w1), self.d_b1)), 0.5)
        d_l2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(d_l1, self.d_w2), self.d_b2)), 0.5)
        pred = tf.sigmoid(tf.add(tf.matmul(d_l2, self.d_w3), self.d_b3))
        return pred

X = tf.placeholder(shape=[None, 28*28], dtype=tf.float32)
Z = tf.placeholder(shape=[None, 28*28], dtype=tf.float32)

g_z = generator(Z)
decoder = Discriminator()
d_x = decoder.network(X)
d_z = decoder.network(g_z)

g_batch = tf.Variable(0)
d_batch = tf.Variable(0)

g_loss = - tf.reduce_mean(tf.reduce_sum(tf.log(d_z), reduction_indices=[1]))
g_learning_rate = tf.train.exponential_decay(0.01, g_batch, 100, 0.95, staircase=True)
g_optimizer = tf.train.MomentumOptimizer(g_learning_rate, 0.9).minimize(g_loss)

d_loss = - tf.reduce_mean(tf.reduce_sum(tf.log(d_x) + tf.log(1-d_z), reduction_indices=[1]))
d_learning_rate = tf.train.exponential_decay(0.01, g_batch, 100, 0.95, staircase=True)
d_optimizer = tf.train.MomentumOptimizer(d_learning_rate, 0.9).minimize(d_loss)

epochs = 100
k = 5

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(epochs):
        for j in range(k):
            # NOTE using an arbitary distribution as noise
            batch_z = np.random.normal(0, 0.1, (batch_size,28*28))
            batch_x = binarize(data.train.next_batch(batch_size)[0])
            _, d_l = sess.run([d_optimizer, d_loss], feed_dict={X:batch_x, Z:batch_z})
            
        batch_z = np.random.normal(0, 0.1, batch_size*28*28).reshape((batch_size, 28*28))
        batch_x = data.train.next_batch(batch_size)[0]
        _t, g_l = sess.run([g_optimizer, g_loss], feed_dict={X:batch_x, Z:batch_z})
        if i%10 ==0:
            print("generator_loss:%f, decoder_loss:%f"%(g_l, d_l))