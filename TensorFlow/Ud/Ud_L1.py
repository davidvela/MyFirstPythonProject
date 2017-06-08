# Udacity Lesson 1: From ML to DL 

import numpy as np
import matplotlib.pyplot as plt

# 1.10 - Softmax
scores  = [3.0, 1.0, 0.2]
x       = np.arange(-2, 6, 0.1)
scores2 = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)] ) 

def softmax(x):
 return np.exp(x) / np.sum(np.exp(x), axis = 0)

print( softmax(scores) )

plt.plot(x, softmax(scores2).T, linewidth = 2 )
plt.show()

