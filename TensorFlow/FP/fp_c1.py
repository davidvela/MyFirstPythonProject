# Classification Model [4885 rows x 1221 columns] - reuse the model! 

# tensorboard --logdir=.\my_graph\0FCR\
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fp_dr import get_data, next_batch
import sys
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

LOGDIR      = "./my_graph/0FCR/"
TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"
TEST_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNE.csv"
model_path  = LOGDIR + "model.ckpt"

# Parameters
learning_rate = 0.001
batch_size = 128
training_iters = 1000 #200000
display_step = training_iters*0.1 #10%
record_step  = training_iters*0.005
# Network Parameters
n_hidden_1  = 256   # 1st layer number of features
n_hidden_2  = 256   # 2nd layer number of features
n_input     = 1221  # data input (img shape: 28*28)
n_classes   = 4     # total classes 

# Model variables
x = tf.placeholder(tf.float32,   shape=[None, n_input],   name="x")
y = tf.placeholder(tf.int16,     shape=[None, n_classes], name="cat")
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),    name="Weights_1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="Weights_2"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="Weights_out"),
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="Bias_1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="Bias_2"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="Bias_out"),
}
xt, yt      = get_data(TRAI_DS, 0)
xtt, ytt    = get_data(TEST_DS, 0)
# String for the logs
def make_hparam_string(learning_rate, no_fc):
    return "lr_%.0E,fc=%d" % (learning_rate, no_fc) 

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    with tf.name_scope("fc_1"):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    with tf.name_scope("fc_2"):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    with tf.name_scope("fc_output"):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# - declaration of model and global attributes 
pred = multilayer_perceptron(x, weights, biases)
softmaxT = tf.nn.softmax(pred)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
with tf.name_scope("xent"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar("xent", cost)
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
summ = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#
def train_model(hparam): 
    # Running first session
    print("Starting 1st session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        writer = tf.summary.FileWriter(LOGDIR + hparam)
        writer.add_graph(sess.graph)
        for i in range(training_iters):  
            xtb, ytb = next_batch(batch_size, xt, yt)
            if i % record_step == 0:
                [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xtb, y: ytb }) 
                writer.add_summary(s, i)
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
        print("Real value: {}", (ytp1)  )
        print("Predicted:", (sess.run([pred, softmaxT], feed_dict={x: xtp1}) )) 
#
#
def main(dv):
    # Construct model
    xtp1.append(xtt[1]);    ytp1.append(ytt[1])
    hparam = make_hparam_string(learning_rate, 3)

    if dv == 1:
        train_model(hparam)
    elif dv == 2: 
        test_model()
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dv = int(sys.argv[1])
        if dv > 0 and dv < 3 :
            main(dv)    
        else: print ("please type 1 for training and 2 for testing")
    else: print ("please type 1 for training and 2 for testing")

#end 