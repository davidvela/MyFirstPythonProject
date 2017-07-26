import tensorflow as tf
import numpy as np

# TensorFlow - lesson 1 and 2 Standfor University
# # visualize graphs in tf: writer ​=​ tf​.​summary​.​FileWriter​(​'./graphs'​,​ sess​.​graph)
# $ python ​[​yourprogram​.​py​]
# $ tensorboard ​--​logdir​=​"./graphs" --port 6006
#   tensorboard --logdir=.\my_graph	
#   http://localhost:6006/

# tensorboard ​--​logdir​="C:\_bd\" --port 6006
#----------------------------------------------

hello = tf.constant('Hello, TensorFlow!')
a = tf.constant([2], name = "a" )
b = tf.constant([3], name = "b")
c = tf.add(a, b, name = "add")
sess = tf.Session()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./my_graph/01', sess.graph)
    result = sess.run(c)
    writer.close( )
print(result)

# create tensors: 
# tf.constant([2,2]), 
# f​.​zeros​([​2​,​ ​3​],​ tf​.​int32​)​ ​==>​ ​[[​0​,​ ​0​,​ ​0​],​ ​[​0​,​ ​0​,​ ​0​]]   // ones
# tf​.​zeros_like​(​input_tensor​)​  //ones_like
# tf.fill(dims, value)
# ..

