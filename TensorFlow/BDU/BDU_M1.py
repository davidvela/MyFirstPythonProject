import tensorflow as tf
import numpy as np
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D

# TensorFlow - lesson 1 Big Data University
#----------------------------------------------
#       Module 1 - logistic regression = classification
#       three function set: 
#           1 weighs_Matrix_Multiplication: tf.matmul(X, weights)
#           2 Bias_Addition: tg.add(weighted_X, bias)
#           3 Fitting to Sigmoid Probabillity curve tf.nn.sigmoid(wieghted_X_with_bias)

#       Module 1 - logistic regression = Activation functions

def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)

    X,Y = np.meshgrid(ws, bs)
    
    os = np.array([actfunc(tf.constant(w*1 + b)).eval(session=sess) \
        for w,b in zip(np.ravel(X), np.ravel(Y))    ])
    
    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1)


sess = tf.Session();
i = tf.constant([1.0, 2.0, 3.0], shape=[1,3])
w = tf.random_normal(shape=[3,3]) #matrix of weights
b = tf.random_normal(shape=[1,3]) #vector of biases

def func(x): return x # dummy activation function

act = func(tf.matmul(i,w)+b)
result = act.eval(session=sess)
print( result )
#plot_act(1.0, func)
#plot_act(1, tf.sigmoid)


# ReLU - Rectified Linear Unit (Linear Unit Functions) (0->infinite)
#      - best activation function!
plot_act(1, tf.nn.relu)
act = tf.nn.relu(tf.matmul(i,w) + b)
print(act.eval(session=sess))