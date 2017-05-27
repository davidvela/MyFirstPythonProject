# TensorFlow - lesson 1 Big Data University

import tensorflow as tf
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
        
        


# Placeholder are like variables -> you have to define the type: 
# placeholder = tf.placeholder()
