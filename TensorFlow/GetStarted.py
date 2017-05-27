import tensorflow as tf
import numpy as np

# TensorFlow - lesson 1 Big Data University
#----------------------------------------------

hello = tf.constant('Hello, TenshorFlow!')
a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a, b)
sess = tf.Session()
# print(sess.run(hello))
print(sess.run(c))
sess.close()

with tf.Session() as sess:
    result = sess.run(c)
    print(result)

# Introduction: Variables and placeholders 
#----------------------------------------------
# Multidimensional Arrays
Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[4,5,6]])
Tensor = tf.constant([  [[1,2,3],[2,3,4],[4,5,6]], [[1,2,3],[2,3,4],[4,5,6]] ] )

Matrix_one = tf.constant([ [1,1],[2,2] ])
Matrix_two = tf.constant([ [2,3],[2,3] ])
# Regular Matrix product : - * | = [4, 6], [8, 12] 
first_operation = tf.matmul(Matrix_one, Matrix_two)

# Variables: - variables need to be initialized
state   = tf.Variable(0)
one     = tf.constant(1)
new     = tf.add(state, one)
update  = tf.assign(state, new)
init_op = tf.initialize_all_variables()


with tf.Session() as session:
    result = session.run(Tensor)
    print( "Tensor (3x3x3 entries): \n" ,  result )

    result = session.run(first_operation)
    print( " \n Multiplication:  \n" ,  result )

    session.run(init_op)
    print("Counter: " , session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))
    
# Placeholder: fedding data from outside he model 
# __ Data types: DT_FLOAT, DT_DOUBLE, DT_INT8..., DT_STRING, DT_BOOL, 
#                DT_COMPLEX64,128, DT_QUINT, 
a = tf.placeholder(tf.float32)
b = a * 2
dictionary = { a: [  [[1,2,3],[2,3,4],[4,5,6]] ] }
with tf.Session() as sess:
    result = sess.run(b, feed_dict={a:3.5})
    print(result)    
    
    result = sess.run(b, feed_dict=dictionary)
    print(result)


# Introduction: Linear regression: Y = aX + b
#----------------------------------------------

# two variables - dependent  : state final goal 
#               - independent: explanatory - clauses states
# Multiple linear regresion => more than 1 independent variables
# Multivariante linear regresion => more than 1 dependent variables

# a slope or gradient and b intercept




