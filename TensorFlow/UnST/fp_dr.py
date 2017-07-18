# DATA HANDLING... 
import pandas as pd 
import tensorflow as tf 
import numpy as np 

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def classif(df):
    if( df < 40 ): return [0,0,0,1] 
    elif( df >= 40 and df < 60 ): return [0,0,1,0]
    elif( df >= 60 and df < 90 ): return [0,1,0,0] 
    elif( df >= 90 ): return [1,0,0,0]
def regress(df):
    return [df]

def get_data(path, type):
    dst =  pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python")
    dst = dst.fillna(0)

    if type == 0:       # Classification in categories
        dst.insert(2, 'FP_C', dst['FP'].map(lambda x: classif(x)))
        cat  = dst.loc[:,'FP_C']
        dat  = dst.iloc[:, 3:]
        # catL = list(cat.values.flatten())
        catL = cat.as_matrix().tolist()
        datL = dat.as_matrix().tolist()
    elif type == 1:     # Regression
        dst.insert(2, 'FP_R', dst['FP'].map(lambda x: regress(x)))
        cat  = dst.loc[:,'FP_R']
        dat  = dst.iloc[:, 3:]
        #print(cat)
        catL = cat.as_matrix().tolist()
        datL = dat.as_matrix().tolist()
    
    return datL, catL

# test logic: 
# TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"
# xt, yt      = get_data(TRAI_DS, 1)
# print(yt)
# for i in range(2):  
#   xtb, ytb = next_batch(10, xt, yt)
#   print(ytb)
#   print("--new--")
