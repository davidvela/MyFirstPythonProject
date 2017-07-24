# DATA HANDLING... 
import pandas as pd 
import tensorflow as tf 
import numpy as np 

class fpDataModel:
    def __init__(self, path, norm, batch_size, dType):
        self.path = path
        self.norm = norm
        self.batch_size = batch_size
        self.dType = dType

    # Ordered batch... 
    def next_sbatch(self, num, data, labels):
        return True

    # Random batch... 
    def next_batch(self, num, data, labels):
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
    def normalization(self, dts ):
        if self.norm == "min_max":
            self._cat_nmax = np.max(dts)
            self._cat_nmin = np.min(dts)
            self._cat_m  = np.mean(dts)
            cat_nn = dts.apply(lambda x: ( (x - self._cat_nmin)) / (self._cat_nmax - self._cat_nmin) )
            return cat_nn
        elif self.norm == 'standardization': 
            self._cat_m  = np.mean(dts)
            self._cat_st = np.std(dts)
            cat_nn = dts.apply(lambda x: ( (x - self._cat_m) /self._cat_st ))
            return cat_nn
    def denormalize(self, x):
        x_pd= pd.DataFrame(x)

        if self.norm == "min_max":
            cat_nn = x_pd.apply(lambda x: ( (x * (self._cat_nmax - self._cat_nmin) + self._cat_nmin) ))
            return cat_nn
        elif self.norm == 'standardization': 
            cat_nn = x_pd.apply(lambda x: ( (x * self._cat_st ) + self._cat_m  ))
            return cat_nn

    # List formatting 
    def classif(self, df):
        if( df < 40 ): return [0,0,0,1] 
        elif( df >= 40 and df < 60 ): return [0,0,1,0]
        elif( df >= 60 and df < 90 ): return [0,1,0,0] 
        elif( df >= 90 ): return [1,0,0,0]

    def regress(self, df): #
        return [df]
    def classifN(self, df):
        return 0
    # Split 
    def split_lab_dat(self, dst, label_col, col_dn):
        cat  = dst.loc[:, label_col].as_matrix().tolist()
        dat  = dst.iloc[:, col_dn:].as_matrix().tolist()

        return {'label' : cat, 'data' : dat}

    #Get Data
    def get_data(self):
        dst =  pd.read_csv( tf.gfile.Open(self.path), sep=None, skipinitialspace=True,  engine="python")
        dst = dst.fillna(0)
        if self.norm != "":
            cat_n  = dst.loc[:,'FP'] 
            dst['FP'] = self.normalization( cat_n )

        if self.dType == 0:       # Classification in 4 categories
            dst.insert(2, 'FP_C', dst['FP'].map(lambda x: classif(x)))
            data  = self.split_lab_dat(dst_tmp[0], 'FP_C', 3)
            return data
        elif self.dType == 1:     # Regression
            dst.insert(2, 'FP_R', dst['FP'].map(lambda x: regress(x)))
            data  = self.split_lab_dat(dst_tmp[0], 'FP_R', 3)
            return data
        elif self.dType == 2:     # separate T and E and then -> Regression with normalization! 
            print("in process")

            dst.insert(2, 'FP_R', dst['FP'].map(lambda x: self.regress(x)))

            dst_tmp = [rows for _, rows in dst.groupby('Type')]
            data_e  = self.split_lab_dat(dst_tmp[0], 'FP_R', 4)
            data_t  = self.split_lab_dat(dst_tmp[1], 'FP', 4)
            return data_t, data_e

# test logic: 
TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"

ALL_DS     = "../../knime-workspace/Data/FP/TFFRGR_ALSN.csv"

dataClass = fpDataModel(ALL_DS, 'min_max', 128, 2)
dtt, dte = dataClass.get_data( ) #'standardization'

# print(dte['label'])
# print( dataClass.denormalize(dte['label']))
    

# xt, yt      = get_data( TRAI_DS, 1)
# print(yt)
# for i in range(2):  
#   xtb, ytb = next_batch(10, xt, yt)
#   print(ytb)
#   print("--new--")
