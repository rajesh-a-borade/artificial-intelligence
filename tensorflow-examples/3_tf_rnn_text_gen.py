# Author -  rajesh.a.borade
# Description - Recurrent Neural network to generate text
# Replaced the original Basic LSTM cell GRU Cell and changed the activation
# original work by https://gist.github.com/mikalv
# dataset was taken from https://github.com/martin-gorner/tensorflow-rnn-shakespeare

import numpy as np
import random
import tensorflow as tf
import datetime
import codecs

# ---------------- Hyperparameters ----------------

max_len = 40
step = 2
num_units = 10
learning_rate = 0.001
batch_size = 200
epoch = 2
temperature = 0.5

# -------- Data preparation helper methods --------

def read_data(file_name):
    text = open(file_name, 'r').read()
    return text.lower()

def featurize(text):
    unique_chars = list(set(text))
    len_unique_chars = len(unique_chars)
    input_chars = []
    output_char = []
    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])
    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    target_data = np.zeros((len(input_chars), len_unique_chars))
    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

# ---------------- Model ----------------

def rnn(x, weight, bias, len_unique_chars):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_unique_chars])
    x = tf.split(x, max_len, 0)
    # cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    cell = tf.nn.rnn_cell.GRUCell(num_units)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction

def sample(predicted):
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

# ---------------- Train and Test ----------------

def run(train_data, target_data, unique_chars, len_unique_chars):
    
    x = tf.placeholder("float", [None, max_len, len_unique_chars])
    y = tf.placeholder("float", [None, len_unique_chars])
    weight = tf.Variable(tf.random_normal([num_units, len_unique_chars]))
    bias = tf.Variable(tf.random_normal([len_unique_chars]))
 
    prediction = rnn(x, weight, bias, len_unique_chars)
    softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
 
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
 
    num_batches = int(len(train_data)/batch_size)
 
    for i in range(epoch):
        print("----------- Epoch {0}/{1} -----------".format(i+1, epoch))
        count = 0
        for _ in range(num_batches):
            train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
            count += batch_size
            sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})
 
        # One of training set as seed
        seed = train_batch[:1:]
        seed_chars = ''
        for each in seed[0]:
            seed_chars += unique_chars[np.where(each == max(each))[0][0]]
 
        # Predicting the next 1000 characters
        for i in range(1000):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print(seed_chars)
    sess.close()

filepath = './data/shakespeare/hamlet.txt'
text = read_data(filepath)
train_data, target_data, unique_chars, len_unique_chars = featurize(text)
run(train_data, target_data, unique_chars, len_unique_chars)


'''

----------- SAMPLE RESULTS -----------

----------- Epoch 1/60 -----------
ame be presently perform'd,
        even while  hne -ms tita aehlt   lhnre om em[th  aoianto e  han ohe  [tit i t igaptea tlrtah he   ret  ast
toe a ate&h e  rn e -noon [hl h  it o e
tino eeria wso  a ohth  e h s   n ltth  h  r e tta ]oe t t hn&  len     a        u  tthmeytat e
t esno  w en e gt the wt. 'to ht t i
tafptane plehneaune
ue,  th  g t wtitt i oa m       eem  a
tte tat h ti ilpe  eee s
 ren nh hea
le,&
eawaii tth  he amee   he ee  w| &s e e t
e c
 le apst&elf s eae hi a ss t  eit tohe  atfh e   e  isa meaur  oenat
 he   u e aa e aths
 m  he?ie i hea  a oiei   h a i ta o st e o  at oo h&ea tlta  t   e  ty h  h t tath othe  kit   t  ow tihflte oi  oi   et r o&  etau h ee o  eea so -on e la me ieau     re ?s
thittae tena sh  lt le[h t]i i in e  matau e ue n aeh  th ea ite ,n e e
-h hoi s iptutnne at hah  os t
ti t  thanh e]&
e a wtee honen eth  t h]t
ites
ea th   o tuh whie  i  h  or tam tyh e  i hoe et-ioo eleei  le ap iooo(hnes e tolh   i tuh t
:|htstt ht hreiatwol  ?h
 ni)  etheo ehis  e tt meishe  t ihr    tt llohee
ae be ehe e gaans



----------- Epoch 24/60 -----------

hamlet  how doon this to be my lord;
        and grast she falled it be the hearty of shall and some,
        that have should wates the spurpse of shall seaven him
        the place of the hearthoras, and string to the counted.

hamlet  i dead dest a wands the such and for my pronsier;
        it be most from his mouth and ear of his
        argand in the disole in him the such and it priase,
        than the thrie change and bellower made,
        i have speak of i am sucked for his shall him
        the pracked than i should more distace of it,
        and the cansion to the fastle.

        [exeunt all his sprich, the pursonit of liat,
        and met a hands and the fasth and the father.

hamlet  i lay think it is the place,
        that he will some this calse of the father;
        and the passand on a fasicion of this light:
        he should have love shome of my that in the hands:
        the canness of this sucked to his hand sir,
        and whose so hall his pase, as hath breaked to heaven.
        that is the canker with usome of placed him

'''