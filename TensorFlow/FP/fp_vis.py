import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

# 1- Ger data: 

path = "../../knime-workspace/Data/FP/TFFRAL_ALSNAC.csv" 
dst  =  pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python")
#C1 M; C2 FP; C3 FN; C4 SFM; C5 AP
Y  = dst.loc[:,'FP'].as_matrix().tolist()
X  = dst.loc[:, 'M'].as_matrix().tolist()

# d = {ni: indi for indi, ni in enumerate(set(names))}
# numbers = [d[ni] for ni in names]

# 2 - Create plot. 
a = 1
if a == 0: 
    plt.hist(Y)
else: 
    plt.xlabel('M')
    plt.ylabel('FP')
    plt.grid(True)


    plt.plot(X, Y, 'bo', label='FP Comp')
    plt.legend()

# 3 - Display plot. 

plt.show()
