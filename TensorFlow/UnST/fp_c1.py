# Classification Model [4885 rows x 1221 columns]
# tensorboard --logdir=.\my_graph\0F\
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fp_dr import get_data, next_batch

import os
import urllib.request as urllib7m   
import pandas as pd
import numpy as np
import tensorflow as tf
# Datasets 
xt          = [] 
yt          = []
xtt         = [] 
ytt         = []
xtp1        = []  
ytp1        = []

#Directories
LOGDIR      = "./my_graph/0FCR/"
TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"
TEST_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNE.csv"
model_path = LOGDIR + "model.ckpt"


# Parameters
dv = 2   
learning_rate = 0.001
batch_size = 128
training_iters = 1000 #200000
display_step = training_iters*0.1 #10%
record_step  = training_iters*0.005
# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 1221 # MNIST data input (img shape: 28*28)
n_classes = 4 # MNIST total classes (0-9 digits)

# Model variables
x = tf.placeholder(tf.float32,   shape=[None, n_input],   name="x")
y = tf.placeholder(tf.int16,     shape=[None, n_classes], name="cat")
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
xt, yt      = get_data(TRAI_DS)
xtt, ytt    = get_data(TEST_DS)
# String for the logs
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    #conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s" % (learning_rate, fc_param) 

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# - declaration of model and global attributes 
pred = multilayer_perceptron(x, weights, biases)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
#
def train_model(): 
    # Running first session
    print("Starting 1st session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        for i in range(training_iters):  
            xtb, ytb = next_batch(batch_size, xt, yt)
            if i % record_step == 0:
                #[train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xt, y: yt }) 
                [train_accuracy] = sess.run([accuracy], feed_dict={x: xtb, y: ytb }) 
            if i % display_step == 0:
                print("step %d, training accracy %g" %(i, train_accuracy))
            sess.run(optimizer, feed_dict={x: xtb, y: ytb})
        print("Optimization Finished!")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: xtt, y: ytt}))
#
def test_model():
    # Running a new session
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
        # test the model
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: xt, y: yt}))
        print("Real value: %d", ytp1  )
        print("Predicted value:", sess.run(pred, feed_dict={x: xtp1}) ) 
#
#
def main():
    # Construct model
    xtp1.append(xtt[1]);    ytp1.append(ytt[1])

    if dv == 1:
        train_model()
    elif dv == 2: 
        test_model()
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == "__main__":
    main()    
