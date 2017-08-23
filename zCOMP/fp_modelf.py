import pandas as pd 
import tensorflow as tf 
import numpy as np 

def nn_input(shape):
    return tf.placeholder(tf.float32,   shape=[None, shape],   name="x")
def nn_output(n_classes): 
    return tf.placeholder(tf.float32,   shape=[None, n_classes],   name="y")
def nn_prob():
    return tf.placeholder(dtype=tf.float32, name="keep_prob")
 

def nn(x_tensor, keep_prob, n_classes ): 
    n_hidden_1  = 256   # 1st layer number of features
    n_hidden_2  = 256   # 2nd layer number of features
    weights = {
        'h1': tf.Variable(tf.random_normal([ x_tensor.get_shape().as_list()[1], n_hidden_1]),    name="Weights_1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1,  n_hidden_2]), name="Weights_2"),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="Weights_out"),
    }

    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1]), name="Bias_1"),
        'b2': tf.Variable(tf.zeros([n_hidden_2]), name="Bias_2"),
        'out': tf.Variable(tf.zeros([n_classes]), name="Bias_out"),
    }
    
    # Hidden layer with RELU activation
    with tf.name_scope("fc_1"):
        layer_1 = tf.add(tf.matmul(x_tensor, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    with tf.name_scope("fc_2"):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        
    # Output layer with linear activation
    with tf.name_scope("fc_output"):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    pred = out_layer
    return pred




