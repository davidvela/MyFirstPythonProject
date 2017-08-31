# Linear Regr. - lesson 3 TF Standfor University
# ph1 - assemble a graph
# ph2 - use session to execute operations in a graph 
#. 
# Feed values into placeholders or variables 
# .
# 03_linear_regr. fire_thef.xls x = fires y = theft.
# want: Predict thefts from fires 
# model = W*x + b.   (Y-YP)^2 
# . 
# tensor board it: tensorboard --logdir='./my_graph' 
# 		Linux/Mac: tensorboard --logdir='./my_graph/03/linear_reg'
# 		Windows:   tensorboard --logdir=.\my_graph\03\linear_reg
#				   tensorboard --logdir=C:\_bd	
# 				   tensorboard --logdir=.\my_graph		
# http://localhost:6006/
# it worked on mac
#----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import sys

dv = 0
if len(sys.argv) > 1:
    dv = int(sys.argv[1])
    if dv > 0 and dv < 2:
	    print("ok")
else: dv = 0    

# Phase 1: Assemble the graph
#------------------------------------------------
# Step 1: read in data from the .xls file
if dv == 0:
	DATA_FILE = 'data/fire_theft.xls'
	book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
	sheet = book.sheet_by_index(0)
	data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
	# generate test data: y = 3x + b
	n_samples = sheet.nrows - 1
else: 
	 n_samples = 99
	 X_input = np.linspace(- 1 , 1 , 100)
	 Y_input = X_input * 3 + np.random.randn( X_input.shape[ 0 ]) * 0.5
	 data = np.column_stack((X_input, Y_input))
	#  data.T[0] = X_input . 
	#  data.T[1] = Y_input


# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')
# Step 4: build model to predict Y
Y_predicted = X * w + b 
# Step 4: Try a Quadratic function 
u = tf.Variable(0.0, name="weights2")
# Y_predicted = X*X*w + X*u + b

# Step 5: use the square error
#  as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
# Step 5: Use Huber loss instead of MSE
# loss = tf.losses.huber_loss(Y, Y_predicted)
# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

y_mean = tf.reduce_mean(Y)
total_error = tf.reduce_sum(tf.square(tf.subtract(Y, tf.reduce_mean(Y))))
unexplained_error = tf.reduce_sum(tf.square(tf.subtract(Y, Y_predicted)))
R_squared = tf.subtract(tf.to_float(1), tf.div(total_error, unexplained_error))

# Phase 1: Train our model - 
#------------------------------------------------
with tf.Session() as sess:
# with tf.InteractiveSession() as sess: # Test interactive session.
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer()) 
	writer = tf.summary.FileWriter('./my_graph/03/linear_reg', sess.graph)
	# Step 8: train the model
	print(data)
	for i in range(100): # train the model 100 times
		total_loss = 0
		for x, y in data:
			# Session runs train_op and fetch values of loss
			_, l = sess.run([optimizer, loss ], feed_dict={X: x, Y:y}) 
			total_loss += l
		# print 'Epoch {0}: {1}'.format(i, total_loss/n_samples)
		print('Epoch %d: %d' % (i, total_loss/n_samples) )
		R2 = sess.run([R_squared], feed_dict={X: data[0], Y:data[1]}) 
		print(R2 )

	
	
	
	# close the writer when you're done using it
	writer.close() 
	# Step 9: output the values of w and b
	w_value, b_value = sess.run([w, b]) 
print("value and error: ", w_value, b_value)

# plot the results
X, Y = data.T[0], data.T[1]

plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()