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
