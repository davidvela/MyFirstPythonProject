# Udacity Lesson 1: From ML to DL 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 1.05 - Supervised classification

# y = wx + b 


# 1.10 - Softmax
scores  = [3.0, 1.0, 0.2]
x       = np.arange(-2, 6, 0.1)
scores2 = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)] ) 

def softmax(x):
 return np.exp(x) / np.sum(np.exp(x), axis = 0)

sess          = tf.Session()
softmaxTensor = tf.nn.softmax(scores)
print( softmax(scores) )
print( softmaxTensor  )
print(sess.run(softmaxTensor))

plt.plot(x, softmax(scores2).T, linewidth = 2 )
plt.ylabel('Probabillities')
plt.xlabel('x values')
plt.title("Softmax function man.")

plt.show()

# 1.15 - Cross Entropy 
#

# 1.16 - Loss (l) = 1/N SUM(D(S(wxi + b), Li ))
#      - Gradient Descent / L(w1,w2) Derivative(L(w1,w2))


