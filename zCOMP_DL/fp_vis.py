import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import time 

start = time.time()
# 1- Get data: 
# path = outfile = '../../_zfp/data/FRFLO/datasc.csv' 
path = outfile = '../../_zfp/data/FLALL/datasc.csv' 
dst  =  pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python")
print(dst.describe())

# path = "../../knime-workspace/Data/FP2/TFFRAL_ALSNN.xlsx"  #160s longer! 
# dst  =  pd.read_excel( path )
#end reading 
elapsed_time = float(time.time() - start)
print(elapsed_time)
# reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
# sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
#                 + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
#                 + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
#                 + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
# if(i % 2500 == 0):
# print("")


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
    N = 2 #50
    colors = np.random.rand(N)
    area = 1 #np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

    #plt.plot(X, Y, color='blue', marker='o', label='FP Comp')
    #plt.plot(X, Y, 'bo', label='FP Comp')
    plt.scatter(X,Y, s=6, c='b', marker='o', cmap=None, norm=None, vmin=60, vmax=101, alpha=None,  label='FP Comp')
    
    plt.legend()

# 3 - Display plot. 

plt.show()
