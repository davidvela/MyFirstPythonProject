# DATA HANDLING... 
import pandas as pd 
import tensorflow as tf 
import numpy as np 

class fpDataModel:
    def __init__(self, path, norm, batch_size, dType, labelCol, dataCol = 4, nC=100, nRange=1 , toList = True):
        self.path           = path
        self.norm           = norm
        self.batch_size     = batch_size
        self.dType          = dType
        self.labelCol       = labelCol
        self.dataCol        = dataCol
        self.nC             = nC
        self.nRange         = nRange
        self.toList         = toList 
    # Ordered batch... 
    def next_obatch(self, num, data, labels):
        return True

    # Random batch... 
    def next_batch(self, num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

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
    def declassif(self, df): 
        if  ( df == [0,0,0,1] ):   return 1 
        elif( df == [0,0,1,0] ):   return 2
        elif( df == [0,1,0,0] ):   return 3  
        elif( df == [1,0,0,0] ):   return 4      
    def regress(self, df): #
        return [df]
    def classifN(self, df):
        listofzeros = [0] * self.nC
        dfIndex = df//self.nRange
        # print('{} and {}', (df,dfIndex))
        if dfIndex < self.nC:
            listofzeros[dfIndex] = 1 
        return listofzeros
    def deClassifN(self, df):
        return df.index(1)
    # Split  
    def split_lab_dat(self, dst):
        cat  = dst.loc[:, self.labelCol]
        dat  = dst.iloc[:, self.dataCol:]
        if (self.toList): 
            cat = cat.as_matrix().tolist()
            dat = dat.as_matrix().tolist()

        return {'label' : cat, 'data' : dat}

    #Get Data
    def get_data(self, typeSep = True):
        dst =  pd.read_csv( tf.gfile.Open(self.path), sep=None, skipinitialspace=True,  engine="python")
        dst = dst.fillna(0)
        
        if self.norm != "":
            cat_n  = dst.loc[:,'FP'] 
            dst['FP'] = self.normalization( cat_n )

        if self.dType == 'class':       # Classification in 4 categories
            dst.insert(2, 'FP_C', dst['FP'].map(lambda x: self.classif(x)))
        elif self.dType == 'reg':     # Regression
            dst.insert(2, 'FP_R', dst['FP'].map(lambda x: self.regress(x)))
        elif self.dType == 'classN':   # Classification in N categories  
            dst.insert(2, 'FP_C', dst['FP'].map(lambda x: self.classifN(x))) 
        
        # 3 if no type and 4 if type
        if typeSep == True:
            dst_tmp = [rows for _, rows in dst.groupby('Type')]
            data_e  = self.split_lab_dat(dst_tmp[0])
            data_t  = self.split_lab_dat(dst_tmp[1])
            return data_t, data_e
        else :   return  self.split_lab_dat(dst_tmp[0])
# main 
def main():
    # test logic: 
    TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"
    # dataClass = fpDataModel(TRAI_DS, 'min_max', 128, 1)
    # dte = dataClass.get_data( )
    # print(dte['label'])


    ALL_DS     = "../../knime-workspace/Data/FP/TFFRGR_ALSN.csv"
    # dataClass = fpDataModel( path= ALL_DS, norm = 'min_max', batch_size = 128, dType="reg", labelCol = 'FP_R', dataCol = 4,   nC=100, nRange=1, toList = True )#'standardization'
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_C', dataCol = 4,   nC=100, nRange=1, toList = True )
    dtt, dte = dataClass.get_data( ) 
    # print(dte['label'])
    # print(dataClass.deClassifN(dte['label'][1]))
    # print( dataClass.denormalize(dte['label']))
        

    # xt, yt      = get_data( TRAI_DS, 1)
    # print(yt)
    # for i in range(2):  
    #   xtb, ytb = next_batch(10, xt, yt)
    #   print(ytb)
    #   print("--new--")

if __name__ == '__main__':
    main()