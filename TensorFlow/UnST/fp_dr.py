# DATA HANDLING... 
import pandas as pd 
import tensorflow as tf 
import numpy as np 

# Ordered batch... 
def next_sbatch(num, data, labels):
    return True
    
# Random batch... 
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

# Normalization
def normalization(type,  dts ):
    if type == "min_max":
        cat_nmax = np.max(dts)
        cat_nmin = np.min(dts)
        cat_m  = np.mean(dts)
        cat_nn = dts.apply(lambda x: ( (x - cat_nmin)) / (cat_nmax - cat_nmin) )
        return cat_nn
    elif type == 'standardization': 
        cat_m  = np.mean(dts)
        cat_st = np.std(dts)
        cat_nn = dts.apply(lambda x: ( (x - cat_m) /cat_st ))
        return cat_nn

# List formatting 
def classif(df):
    if( df < 40 ): return [0,0,0,1] 
    elif( df >= 40 and df < 60 ): return [0,0,1,0]
    elif( df >= 60 and df < 90 ): return [0,1,0,0] 
    elif( df >= 90 ): return [1,0,0,0]

def regress(df): #
    return [df]
# Split 
def split_lab_dat(dst, label_col, col_dn):
    cat  = dst.loc[:, label_col]
    dat  = dst.iloc[:, col_dn:]
    return {'label' : cat, 'data' : dat}

''' ----------     Get data      ----------
    this is the main method of this file. 
    it will read a csv file pased in the path parameter where: 
        1st column = ID, 2ndC = class, 3rdC = Class normalized and restC => inputs 
    type:   1 =  Classificaiton(C)  => 4Categories(C);  Return(rr) = tupple data list and class list 
            2 =  Regression(R) => 1C; rr = tupple training list and evaluation list 
            3 =  R - reading improved separated by 3 column type = T or E; => 1C; rr = tupple T dict and E dict *data 0, class 1 (norm) 
            4 =  C => 100C; rr T and E dict. 

    norm:   1 = min - max 
            2 =  
'''
def get_data(path, type, norm = 1):
    dst =  pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python")
    dst = dst.fillna(0)

    if type == 0:       # Classification in 4 categories
        dst.insert(2, 'FP_C', dst['FP'].map(lambda x: classif(x)))
        cat  = dst.loc[:,'FP_C']
        dat  = dst.iloc[:, 3:]
        # catL = list(cat.values.flatten())
        catL = cat.as_matrix().tolist()
        datL = dat.as_matrix().tolist()
        return datL, catL
    elif type == 1:     # Regression
        dst.insert(2, 'FP_R', dst['FP'].map(lambda x: regress(x)))
        cat  = dst.loc[:,'FP_R']
        dat  = dst.iloc[:, 3:]
        #print(cat)
        catL = cat.as_matrix().tolist()
        datL = dat.as_matrix().tolist()
        return datL, catL
    elif type == 2:     # separate T and E and then -> Regression with normalization! 
        print("in process")

        cat_n  = dst.loc[:,'FP'] 
        cat_nn = normalization( norm, cat_n )
        dst['FP'] = cat_nn
        dst.insert(2, 'FP_R', dst['FP'].map(lambda x: regress(x)))

        dst_tmp = [rows for _, rows in dst.groupby('Type')]
        data_e  = split_lab_dat(dst_tmp[0], 'FP_R', 3)
        data_t  = split_lab_dat(dst_tmp[1], 'FP', 3)

        # print(cat_nn)
        # print( data_e['label']) #print( data_e['data'])

        # cat  = dst.loc[:,'FP_R']
        # for idx in range(40):
        #     print("{} -> {}".format(cat_n[idx], cat_nn[idx]))  
        
        return data_t, data_e

# test logic: 
TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"

ALL_DS     = "../../knime-workspace/Data/FP/TFFRGR_ALSN.csv"
dtt, dte = get_data( ALL_DS, 2, 'min_max') #'standardization'

  

# xt, yt      = get_data( TRAI_DS, 1)
# print(yt)
# for i in range(2):  
#   xtb, ytb = next_batch(10, xt, yt)
#   print(ytb)
#   print("--new--")
