
# Author -  rajesh.a.borade
# Description - Autoencoder for fraud detection
# Dataset - dataset was taken from https://www.kaggle.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -------- Data preparation --------

df_train = pd.read_csv("./data/fraud_data/creditcard_train.csv")
data_train = df_train.drop(['Time'], axis=1)
y_arr_train = data_train['Class']
y_arr_train = np.asarray(y_arr_train)
y_train = []
for _item in y_arr_train :
    if _item == 0 :
        y_train.append([0,1])
    if _item == 1 :
        y_train.append([1,0])
x_train = data_train.drop(['Class'], axis=1)

df_test = pd.read_csv("./data/fraud_data/creditcard_test.csv")
data_test = df_test.drop(['Time'], axis=1)
y_arr_test = data_test['Class']
y_arr_test = np.asarray(y_arr_test)
y_test = []
for _item in y_arr_test :
    if _item == 0 :
        y_test.append([0,1])
    else :
        y_test.append([1,0])
x_test = data_test.drop(['Class'], axis=1)

# -------- Hyperparameters --------

TOTAL_TRAIN_EXAMPLES = x_train.shape[0]

INPUT_DIMENSION_29 = x_train.shape[1]
DIMENSION_14 = int(INPUT_DIMENSION_29 / 2)
OUTPUT_DIMENSION_29 = INPUT_DIMENSION_29
NUM_OF_CLASSES = 2
BATCH_SIZE = 100
EPOCHS = 1

x = tf.placeholder('float', [None, INPUT_DIMENSION_29])
y = tf.placeholder('float', [None, NUM_OF_CLASSES])

# -------- Neural Network --------

def neural_network(x) :
    weights = {
        'in': tf.Variable(tf.random_normal([INPUT_DIMENSION_29, DIMENSION_14])),
        'e1': tf.Variable(tf.random_normal([DIMENSION_14, DIMENSION_14])),
        'd1': tf.Variable(tf.random_normal([DIMENSION_14, OUTPUT_DIMENSION_29])),
        'out': tf.Variable(tf.random_normal([OUTPUT_DIMENSION_29, NUM_OF_CLASSES]))
    }
    biases = {
        'b_in': tf.Variable(tf.random_normal([DIMENSION_14])),
        'b_e1': tf.Variable(tf.random_normal([DIMENSION_14])),
        'b_d1': tf.Variable(tf.random_normal([OUTPUT_DIMENSION_29])),
        'b_out': tf.Variable(tf.random_normal([NUM_OF_CLASSES]))
    }     
    input_layer =  tf.add(tf.matmul(x, weights['in']), biases['b_in'])
    input_layer = tf.nn.tanh(input_layer)
    encoder = tf.add(tf.matmul(input_layer, weights['e1']), biases['b_e1'])
    encoder = tf.nn.relu(encoder)
    decoder =  tf.add(tf.matmul(encoder, weights['d1']), biases['b_d1'])
    decoder = tf.nn.tanh(decoder)
    output = tf.add(tf.matmul(decoder, weights['out']), biases['b_out'])
    output = tf.nn.relu(output)
    return output
 
# ---------------- Train and Test ----------------

def run_neural_network(x) :
     
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        test_accuracy = 0
        for epoch in range(EPOCHS) :
            epoch_loss = 0
            for i in range(int(TOTAL_TRAIN_EXAMPLES/BATCH_SIZE)) :
                off_1 = i * BATCH_SIZE
                off_2 = i * BATCH_SIZE + BATCH_SIZE
                # print('Training batch - ', off_1, ' : ', off_2)
                epoch_x = x_train[off_1:off_2]
                epoch_y = y_train[off_1:off_2]
                epoch_x = np.asarray(epoch_x)
                epoch_y = np.asarray(epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, ' completed out of ', EPOCHS, ' loss : ', epoch_loss)
            train_accuracy = accuracy.eval({x:epoch_x, y:epoch_y})
            print('Train Accuracy : ', train_accuracy)
        
        test_accuracy = accuracy.eval({x:x_test, y:y_test})
        print('Test Accuracy : ', test_accuracy)
        # ------------- Validate the results -------------
        y_pred = sess.run(prediction, feed_dict = {x:x_test})
        tensor = tf.constant(y_pred)
        y_pred_numpy_array = tensor.eval()
        y_arr = []
        p_arr = []
        y_frauds = 0
        for _item in y_test :
            _item_np = np.array(_item)
            if(_item_np[0] == 0) :
                y_arr.append(0)
            else :
                y_arr.append(1)        
                y_frauds = y_frauds + 1
        p_frauds = 0
        for _item in y_pred_numpy_array :
            _item_np = np.array(_item)
            if(_item_np[0] == 0) :
                p_arr.append(0)
            else :
                p_arr.append(1)
                p_frauds = p_frauds + 1
        print('-----------------------------------')
        print('Actual fraud cases - ', y_frauds)
        confusion = tf.confusion_matrix(labels=y_arr, predictions=y_arr, num_classes=NUM_OF_CLASSES)
        print('Objective Confusion matrix - ')
        print(confusion.eval())
        print('-----------------------------------')
        print('Predicted fraud cases - ', p_frauds)
        print('Prediction Confusion matrix - ')
        confusion = tf.confusion_matrix(labels=y_arr, predictions=p_arr, num_classes=NUM_OF_CLASSES)
        print(confusion.eval())
        
        
run_neural_network(x)


''' 
-------------- SAMPLE RESULT --------------

Train Accuracy :  0.99
Epoch  48  completed out of  50  loss :  7.466624181273801
Train Accuracy :  0.99
Epoch  49  completed out of  50  loss :  7.260177716470935
Train Accuracy :  1.0
Test Accuracy :  0.99935997
-----------------------------------
Actual fraud cases -  75
[[57734     0]
 [    0    75]]
-----------------------------------
Predicted fraud cases -  428
Prediction Confusion matrix -
[[57727     7]
 [   30    45]]

---------------------------------------------
 '''