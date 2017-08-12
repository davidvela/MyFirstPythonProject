import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')
a = tf.constant(1, tf.int32)
b = tf.constant(1, tf.float64)
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)


# LESSON 13 - LINEAR FUNCTION - LOGISTIC CLASSIFIER 
# tf.truncated_normal()
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
# tf.zeros()
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))