import tensorflow as tf
import numpy as np
#import matplotlib.patches as mpatches 
#import matplotlib.pyplot as plt 

# TensorFlow - lesson 1 Big Data University - tablet!
#----------------------------------------------

hello = tf.constant('Hello, TenshorFlow!')
a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a, b)
sess = tf.Session()
with tf.Session() as sess:
    result = sess.run(c)
    print(result)



