# DATA HANDLING... 
import pandas as pd 
import tensorflow as tf 
import numpy as np 
import requests
import json
from types import *
from collections import Counter

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
    #
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
    #
    def declassif(self, df): 
        if  ( df == [0,0,0,1] ):   return 1 
        elif( df == [0,0,1,0] ):   return 2
        elif( df == [0,1,0,0] ):   return 3  
        elif( df == [1,0,0,0] ):   return 4      
    #
    def regress(self, df): #
        return [df]
    
    #
    def classifN(self, df):
        listofzeros = [0] * self.nC
        dfIndex = df//self.nRange
        # print('{} and {}', (df,dfIndex))
        if dfIndex < self.nC:
            listofzeros[dfIndex] = 1 
        return listofzeros
    
    #
    def deClassifN(self, df, val = 1 ):
        return df.index(val)
    
    # Split  
    def split_lab_dat(self, dst):
        cat  = dst.loc[:, self.labelCol]
        dat  = dst.iloc[:, self.dataCol:]
        if (self.toList): 
            cat = cat.as_matrix().tolist()
            dat = dat.as_matrix().tolist()

        return {'label' : cat, 'data' : dat}
    
    #Get Data
    def get_data(self, typeSep = True, pathA = "", filter = ""):
        if pathA != "":
            dst =  pd.read_csv( tf.gfile.Open(pathA), sep=None, skipinitialspace=True,  engine="python")
        else: 
            dst =  pd.read_csv( tf.gfile.Open(self.path), sep=None, skipinitialspace=True,  engine="python")
        
        dst = dst.fillna(0)
        
        if filter == '>23':
            dst = dst[dst["FP"]>23]
        elif filter == '>60':
            dst = dst[dst["FP"]>60]


        if self.norm != "":
            cat_n  = dst.loc[:,'FP'] 
            dst['FP'] = self.normalization( cat_n )

        if   self.dType == 'class':       # Classification in 4 categories
            dst.insert(2, 'FP_C', dst['FP'].map(lambda x: self.classif(x)))
        elif self.dType == 'reg':       # Regression
            dst.insert(2, 'FP_R', dst['FP'].map(lambda x: self.regress(x)))
        elif self.dType == 'classN':    # Classification in N categories  
            dst.insert(2, 'FP_C', dst['FP'].map(lambda x: self.classifN(x))) 
        
        self.dst = dst

        # 3 if no type and 4 if type
        if typeSep == True:
            dst_tmp = [rows for _, rows in dst.groupby('Type')]
            data_e  = self.split_lab_dat(dst_tmp[0])
            data_t  = self.split_lab_dat(dst_tmp[1])
            return data_t, data_e
        else :   return  self.split_lab_dat(dst_tmp[0])
    
    # WS - Conversion
    
    def set_columns(self, url ):        # set the main data frame from the class: 
        columns_path = url
        self.col_df = pd.read_csv(columns_path, index_col=0, sep=',', usecols=[0,1,2,3])
        return(len(self.col_df))
        
    def feed_data(self, url , type="", d_st = False):
        json_df = pd.DataFrame(columns=self.col_df.index) 
        df_entry = pd.Series(index=self.col_df.index)

        df_entry = df_entry.fillna(0) 
        comp_out_count = Counter()
        
        if(isinstance( url, list)):json_data = url
        else:   
            json_str=open(url).read()
            json_data = json.loads(json_str)

        for i in range(len(json_data)):
            df_entry *= 0
            m = str(json_data[i]["m"])
            df_entry.name = m
            for key in json_data[i]:
                if key == "m":  
                    pass            
                else: 
                    #key_wz = str(int(key)
                    key_wz = int(key)
                    try:
                        ds_col = self.col_df.loc[key_wz]
                        #df_entry.loc[key_wz]
                        df_entry[key_wz] =  np.float32(json_data[i][key])
                    except: 
                        if d_st == True: 
                            print("column: {} not included in the input of: {}" .format(key_wz, m))
                        # comp_out_count[key_wz] +=1
            json_df = json_df.append(df_entry,ignore_index=False)
        # print("Counter of comp. not included :")
        # print(len(comp_out_count))
        # return json_df  
        return json_df.as_matrix().tolist()  
    #
    def get_labels(self, url, type=""):
        pass


    def check_perf(self, lA, lB):
        assert(len(lA) == len(lB))
        gt3  = 0
        gtM = 0
        num = 0
        for i in range(len(lA)):
            num = abs(lA[i]-lB[i])
            if num > 3: gt3+=1
            if num > 10: gtM+=1
        return gt3, gtM
    
#  main 
def main():
    # test logic: 
    TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"
    # dataClass = fpDataModel(TRAI_DS, 'min_max', 128, 1)
    # dte = dataClass.get_data( )
    # print(dte['label'])


    ALL_DS     = "../../knime-workspace/Data/FP/TFFRGR_ALSN.csv"
    ALL_COL     = "../_zfp/data/TFFRFLO_COL.csv"

    # dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="reg", labelCol = 'FP_R', dataCol = 4,   nC=100, nRange=1, toList = False )#'standardization'
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_C', dataCol = 4,   nC=100, nRange=1, toList = True )
    # dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="class", labelCol = 'FP_C', dataCol = 4,   nC=100, nRange=1, toList = True )
    
    print(dataClass.set_columns(ALL_COL))
    json_path    = "../_zfp/data/JSON.txt"
    json_path    = "../_zfp/data/json_fflo_ex.txt"
    
    json_str="""[{"m":"000","100214":1} ]"""
    json_data = json.loads(json_str)
    json_df = dataClass.feed_data(json_data) 
    # json_df = dataClass.feed_data(json_path) 
    el = json_df[0]
    print(np.max(el))    
    # dtt, dte = dataClass.get_data( True,  filter = ">60" ) 
    # print(dte['data'])
    # print(len(dtt[0])   )
    
    # print(dataClass.deClassifN(dte['label'][0]))
    # print( dataClass.denormalize(dte['label']))
        

    # xt, yt      = get_data( TRAI_DS, 1)
    # print(yt)
    # for i in range(2):  
    #   xtb, ytb = next_batch(10, xt, yt)
    #   print(ytb)
    #   print("--new--")

def read_json():    
    file_directory    = "../../knime-workspace/Data/FP2/JSON.txt"
    json_data=open(file_directory).read()
    data = json.loads(json_data)
    # print(data)
    print(data[1]['m'])
def read_json_url(url):
        url_oData_people = "http://services.odata.org/TripPinRESTierService/(S(pk4yy1pao5a2nngmm2ecx0hy))/People"
        # response = requests.get( url_oData_people )
        # people   = response.json();   # print(people)
        # CONVERT JOSN into object -> Pandas or dictionary array.7
        movie_json = """
        {
        "Title":"Johnny 5",
        "Year":"2001",
        "Runtime":"119 min",
        "Country":"USA"
        }
        """
        movie_data = json.loads(movie_json) # <class 'dict'>
        print("The title is {}".format(movie_data.get('Title')))    
        # add new elements to the dataset

def test_iris():
    dst  =  pd.read_csv( tf.gfile.Open('./data/iris/iris_test.csv'), sep=None, skipinitialspace=True,  engine="python")
    print(dst[-5:])
    #columns = ['30',    '4',  'setosa',  'versicolor',  'virginica'] #dst.columns
    
    def add_newIris(dst, jsonDat):
        # entry = pd.Series([0,0,0,0,0], index=columns)
        entry = pd.Series(index=dst.columns)
        entry = entry.fillna(0)
        # entry['setosa'] = 8

        # typeJ = type(jsonDat)
        # print(typeJ)

        if(isinstance( jsonDat, dict  )):
            iris_data = jsonDat
        else:  
            iris_data = json.loads(jsonDat) # <class 'dict'>
        
        # print("The title is {}".format(movie_data.get('Title')))
        for key in iris_data:
            #print( '{} corresponds to {}'.format(key,iris_data[key]))
            entry[key] = iris_data[key]
        
        dst = dst.append(entry,ignore_index=True)
        return dst
    
    dst = add_newIris(dst,iris_json)        
    print(dst[-5:])

    iris_data = json.loads(iris_json) 
    iris_data["30"]=2000
    dst = add_newIris(dst,iris_data)        
    print(dst[-5:])



iris_json = """
    {
        "30":"5.5",
        "4":"3.5",
        "setosa":"0",
        "versicolor":"33"
    }
"""

LOG        = "../../_zfp/LOGT.txt"
from datetime import datetime
def recordLogF():
    f= open(LOG ,"a+") #w,a,
    f.write("This is \t line %d\n" % (1)  + datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
    f.close()
    print("test")


if __name__ == '__main__':
    # main()
    recordLogF()
    # read_json()
    # test_iris()