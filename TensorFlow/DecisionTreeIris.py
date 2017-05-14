import tensorflow as tf
hello = tf.constant('Hello, TenshorFlow!')
sess = tf.Session()
print(sess.run(hello))
