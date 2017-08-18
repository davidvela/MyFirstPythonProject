# DATA HANDLING... 
import pandas as pd 
import tensorflow as tf 
import numpy as np 

def network(n_input, n_classes, lr): 
    x = tf.placeholder(tf.float32,   shape=[None, n_input],   name="x")
    y = tf.placeholder(tf.int16,     shape=[None, n_classes], name="cat")

    n_hidden_1  = 256   # 1st layer number of features
    n_hidden_2  = 256   # 2nd layer number of features

    self.weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),    name="Weights_1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="Weights_2"),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="Weights_out"),
    }

    self.biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1]), name="Bias_1"),
        'b2': tf.Variable(tf.zeros([n_hidden_2]), name="Bias_2"),
        'out': tf.Variable(tf.zeros([n_classes]), name="Bias_out"),
    }
    
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

    pred = out_layer

    softmaxT = tf.nn.softmax(pred)
    prediction=tf.reduce_max(y,1)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("xent"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("xent", cost)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

return pred, optimizer, accuracy





