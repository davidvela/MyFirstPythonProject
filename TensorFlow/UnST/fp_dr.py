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

def get_data(path):
    dst =  pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python")
    dst = dst.fillna(0)

    dst.insert(2, 'FP_C', dst['FP'].map(lambda x: classif(x)))


    cat  = dst.loc[:,'FP_C']
    dat  = dst.iloc[:, 3:]
    # catL = list(cat.values.flatten())
    catL = cat.as_matrix().tolist()
    datL = dat.as_matrix().tolist()
        
    return datL, catL