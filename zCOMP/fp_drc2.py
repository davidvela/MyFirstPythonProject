import pandas as pd 
import tensorflow as tf 
import numpy as np 

import requests
import json
import sys
import os
import time
from types import *
from collections import Counter
from datetime import datetime

#version 2
#read data => class - COM or FP input // output - CAT 100 or 4! 
class fpDataModel:
    def __init__(self, path, norm, batch_size, dType, labelCol, dataCol = 4, nC=100, nRange=1 , toList = True, pathFile= "/TFFRFLO_ALSN.csv"):
        self.path           = path
        self.norm           = norm
        self.batch_size     = batch_size
        self.dType          = dType
        self.labelCol       = labelCol
        self.dataCol        = dataCol
        self.nC             = nC
        self.nRange         = nRange
        self.toLs           = toList 
        self.DSC            = pathFile
    def next_batch(self, num, data, labels):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    def get_batches(self, x, y, batch_size=128):
        n_batches = len(y)//batch_size
        x,y = x[:n_batches*batch_size], y[:n_batches*batch_size]
        for ii in range(0, len(x), batch_size ):
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]
    def classif(self, df, rv=False):
        if rv == False: 
            if( df < 23 ): return [1,0,0,0]                 #0
            elif( df >= 23 and df < 60 ): return [0,1,0,0]  #1
            elif( df >= 60 and df < 93 ): return [0,0,1,0]  #2
            elif( df >= 93 ): return [0,0,0,1]              #3
        else: 
            if( df < 23 ): return 0
            elif( df >= 23 and df < 60 ): return 1
            elif( df >= 60 and df < 93 ): return 2
            elif( df >= 93 ): return 3
    def declassif(self, df): 
        if  ( df == [1,0,0,0] ):   return 0 
        elif( df == [0,1,0,0] ):   return 1
        elif( df == [0,0,1,0] ):   return 2  
        elif( df == [0,0,0,1] ):   return 3      
    def regress(self, df): 
        return [df]
    def classifN(self, df):
        listofzeros = [0] * self.nC
        dfIndex = df//self.nRange
        # print('{} and {}', (df,dfIndex))
        if 0 < dfIndex < self.nC:   listofzeros[dfIndex] = 1
        elif dfIndex < 0:           listofzeros[0] = 1
        elif  dfIndex >= self.nC:   listofzeros[self.nC-1] = 1
        return listofzeros
    def deClassifN(self, df, val = 1 ):
        if self.dType == 'class':       # CLASS in 4C
            return df.index(val) #self.declassif(df) 
        elif self.dType == 'reg':       # Regression
            return df
        elif self.dType == 'classN':    # CLASS in NC  
            return df.index(val)
    def classify(self, x, rv=False):
        if  self.dType == 'class':     # Classification in 4 categories
            return self.classif(x, rv)
        elif self.dType == 'reg':       # Regression
            return self.regress(x)
        elif self.dType == 'classN':    # Classification in N categories  
            return self.classifN(x) 
    def split_lab_dat(self, dst):
        cat  = dst.loc[:, self.labelCol]
        dat  = dst.iloc[:, self.dataCol:]
        if (self.toLs): 
            cat = cat.as_matrix().tolist()
            dat = dat.as_matrix().tolist()
        return {'label' : cat, 'data' : dat}
    #Get Data
    def convert_2List(self, dst):
        cat = dst["label"].as_matrix().tolist()
        dat = dst["data"].as_matrix().tolist()
        return {'label' : cat, 'data' : dat}
    def read_dst(self, pathA = ""):
        if hasattr(self, 'dst'): go = 1 
        else: go = 0

        if go == 1 and  pathA != "" and pathA != self.path :
            self.dst =  pd.read_csv( tf.gfile.Open(pathA), sep=None, skipinitialspace=True,  engine="python")
            self.dst = self.dst.fillna(0)
        elif go == 0: 
            self.dst =  pd.read_csv( tf.gfile.Open(self.path), sep=None, skipinitialspace=True,  engine="python")
            self.dst = self.dst.fillna(0)
        return self.dst
    def get_data(self, typeSep = True, pathA = "", filt = "", filtn = 0, pand=True ):
        # dst = self.read_dst(pathA = pathA )
        if pathA != "": dst =  pd.read_csv( tf.gfile.Open(pathA), sep=None, skipinitialspace=True,  engine="python")
        else: dst =  pd.read_csv( tf.gfile.Open(self.path), sep=None, skipinitialspace=True,  engine="python")
        dst = dst.fillna(0)

        if filt == '>':
            dst = dst[dst["FP"]>filtn]
        elif filt == '<':
            dst = dst[dst["FP"]<filtn]
        if self.norm != "": # normalization not longer used since I am using always classification! 
            cat_n  = dst.loc[:,'FP'] 
            dst['FP'] = self.normalization( cat_n )               
        dst.insert(2, 'FP_P', dst['FP'].map(lambda x: self.classify(x)))        
        # self.dst = dst
        # 3 if no type and 4 if type
        if typeSep == True:
            self.dataCol = 4 # M F T C1 ... 
            dst_tmp = [rows for _, rows in dst.groupby('Type')]
            data_e  = self.split_lab_dat(dst_tmp[0])
            data_t  = self.split_lab_dat(dst_tmp[1])
            return data_t, data_e
        else :  
            self.dataCol = 3 # M F C1...  
            dst = dst.sample(frac=1).reset_index(drop=True)
            return  self.split_lab_dat(dst)
    def get_data2(self, colu="", datu=""):
        pass
    #json
    def set_columns(self, url , p_abs = False):        # set the main data frame from the class: 
        columns_path = url
        n=0
        if p_abs == False:
            n=2
        self.col_df = pd.read_csv(columns_path, index_col=n, sep=',', usecols=[0,1,2,3])
        return(len(self.col_df))
        
    def feed_data(self, url, abstract, type="", d_st = False, p_exp=False, pand=False):
        if p_exp == True:  #fp key - 0-102  print(ds_comp['fp'])
            indx = []
            for i in range(103): indx.append(i) 
        else:
            indx = self.col_df.index
        json_df = pd.DataFrame(columns=indx) 
        df_entry = pd.Series(index=indx)

        df_entry = df_entry.fillna(0) 
        comp_out_count = Counter()
        if(isinstance(url, list)):json_data = url
        else:   
            json_str=open(url).read()
            json_data = json.loads(json_str)

        # for i in range(20):
        for i in range(len(json_data)):
            # print(i)
            df_entry *= 0
            m = str(json_data[i]["m"])
            df_entry.name = m
            for key in json_data[i]:
                if key == "m":  
                    pass            
                else: 
                    if abstract == True :  key_wz = key      #int(key)      #abstract == True
                    else:                  key_wz = int(key) #str(int(key)) #abstract == False
                    try: #filling of key - experimental or components 
                        ds_comp = self.col_df.loc[key_wz]
                        if p_exp == True:  #fp key - 0-102   
                            co = str(ds_comp['FP'])
                            if co == 'nan': 
                                col_key = 102
                            else: 
                                col_key = int(ds_comp['FP'])
                                if col_key>101: col_key = 101
                                if col_key<0: col_key = 0
                        else: col_key = key_wz      
                        # df_entry.loc[col_key]
                        df_entry[col_key] =  np.float32(json_data[i][key])
                    except: 
                        if d_st == True: 
                            print("m:{}-c:{} not included" .format(m, key_wz))
                            comp_out_count[key_wz] +=1
            json_df = json_df.append(df_entry,ignore_index=False)
            if i % 1000 == 0: print("cycle: {}".format(i))
        print("Counter of comp. not included :"); print(comp_out_count) # print(len(comp_out_count))
        if pand == True:  return json_df  
        else:             return json_df.as_matrix().tolist()  
    def read_json(url_col, url_comp, url_lab, url_json=""):
        self.col_df = pd.read_csv(url_comp, index_col=0, sep=',', usecols=[0,1,2,3])
        print("columns: " + str(len(self.col_df)))
        # self.col_df  = pd.read_csv(url_col,  index_col=0, sep=',', usecols=[0])
        # print("columns: " + str(len(self.col_df)))
        self.lab_df = pd.read_csv(url_lab, index_col=0, sep=',', usecols=[0])
        print("labels: " + str(len(self.col_df)))
        #hot encode labels: 
        filter = 0
        if filter != 0:
            self.lab_df =  self.lab_df[self.lab_df["FP"]>filter]
        self.lab_df.insert(2, 'FP_C', self.lab_df['FP'].map(lambda x: self.classify(x))) 

        if url_json != "": self.npData = self.feed_data(url_json)
        return


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
# print("data class build")
# Notes: 

# test:  
def main():
    COM_DS     = "../_zfp/data/TFFRFLO_COM.csv"
    COL_DS     = "../_zfp/data/TFFRFLO_COL.csv"

    dataTest = {'label' : [] , 'data' :  [] }
    
    # ALL_DS     = "../../_zfp/data/FRFLO/datasc.csv"
    # dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_P', 
    #                          dataCol = 4,   nC=101, nRange=1, toList = False, pathFile= 'FRFLO'  )
    # dst = dataClass.read_dst()
    # dt, de =  dataClass.get_data(pathA=ALL_DS, typeSep = True, ) 

    return
    # COL_DS
    dataClassTest = fpDataModel( path= COL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_P', 
                            dataCol = 4,   nC=100, nRange=1, toList = True )
    
    n_input2 = dataClassTest.set_columns(COL_DS, p_abs = False)
    print(n_input2)
    json_str="""[{"m":"000","100028":9}
    ,{ "m":"000000000000636978", "100000000000100028" :0.009 , "000000000000668567" :0.008 , "000000000000131503" :0.008 }
    ]"""
    json_data = json.loads(json_str)
    dataTest['data'] = dataClassTest.feed_data(json_data, d_st =True , p_exp=True, ) 
    print(  dataTest['data'][1]  )

if __name__ == '__main__':
    # main()
    pass

def backup(): pass
    # def write_data( ): #codes, labels): 
    #     # write codes to file
    #     with open('codes', 'w') as f:
    #         codes.tofile(f)
    #     # write labels to file
    #     with open('labels', 'w') as f:
    #         writer = csv.writer(f, delimiter='\n')
    #         writer.writerow(labels)
    #     return True

    # def read_data( ):
    #     with open('labels') as f:
    #         reader = csv.reader(f, delimiter='\n')
    #         labels = np.array([each for each in reader if len(each) > 0]).squeeze()
    #     with open('codes') as f:
    #         data = np.fromfile(f, dtype=np.float32)
    #         data = codes.reshape((len(labels), -1))
    #     return codes, labels
#end

