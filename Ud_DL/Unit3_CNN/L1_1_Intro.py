import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')
a = tf.constant(1, tf.int32)
b = tf.constant(1, tf.float64)

output = None
logit_data = [2.0, 1.0, 0.1]
logits = tf.placeholder(tf.float32)

softmax = tf.nn.softmax(logits)


softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)
# ToDo: Print cross entropy from session
cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    output = sess.run(softmax, feed_dict={logits: logit_data})
    print(output)
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))


# LESSON 13 - LINEAR FUNCTION - LOGISTIC CLASSIFIER 
# tf.truncated_normal()
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
# tf.zeros()
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))