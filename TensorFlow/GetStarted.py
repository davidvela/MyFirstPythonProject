import tensorflow as tf
import numpy as np
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt 

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
    print("placeholder: ", result)    
    
    result = sess.run(b, feed_dict=dictionary)
    print("placeholder DIC: ", result)


# Introduction: Linear regression: Y = aX + b
#----------------------------------------------

# two variables - dependent  : state final goal 
#               - independent: explanatory - clauses states
# Multiple linear regresion => more than 1 independent variables
# Multivariante linear regresion => more than 1 dependent variables

# a slope or gradient and b intercept

plt.rcParams['figure.figsize'] = (10,6)
X = np.arange(0.0, 5.0, 0.1)
a = 1 
b = 0
Y = a*X + b
plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
# plt.show()

x_data = np.random.rand(100).astype(np.float)
# model: Y = 3X + 2
y_data = x_data*3 + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale = 0.1 ))(y_data)  # with error 

zip(x_data, y_data)
a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b      # Model y values 

# Equation to be minimized as loss 
loss        = tf.reduce_mean(tf.square(y - y_data))
optimizer   = tf.train.GradientDescentOptimizer(0.5)
train       = optimizer.minimize(loss)

init        = tf.initialize_all_variables( )
sess        = tf.Session()
sess.run(init)
train_data = []
#for step in range(100)
