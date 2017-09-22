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

    def next_batch(self, num, data, labels):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    def classif(self, df, rv=False):
        if rv == False: 
            if( df < 23 ): return [0,0,0,1] 
            elif( df >= 23 and df < 60 ): return [0,0,1,0]
            elif( df >= 60 and df < 93 ): return [0,1,0,0] 
            elif( df >= 93 ): return [1,0,0,0] 
        else: 
            if( df < 23 ): return 1
            elif( df >= 23 and df < 60 ): return 2
            elif( df >= 60 and df < 93 ): return 3
            elif( df >= 93 ): return 4

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
    def deClassifN(self, df, val = 1 ):
        
        if self.dType == 'class':     # Classification in 4 categories
            # return self.declassif(df) 
             return df.index(val)
        elif self.dType == 'reg':       # Regression
            return df
        elif self.dType == 'classN':    # Classification in N categories  
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
        if (self.toList): 
            cat = cat.as_matrix().tolist()
            dat = dat.as_matrix().tolist()
        return {'label' : cat, 'data' : dat}
    #Get Data
    def get_data(self, typeSep = True, pathA = "", filt = "", filtn = 0 ):
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
        self.dst = dst
        # 3 if no type and 4 if type
        if typeSep == True:
            dst_tmp = [rows for _, rows in dst.groupby('Type')]
            data_e  = self.split_lab_dat(dst_tmp[0])
            data_t  = self.split_lab_dat(dst_tmp[1])
            return data_t, data_e
        else :   return  self.split_lab_dat(dst_tmp[0])
    def get_data2(self, colu="", datu=""):
        pass
    #json
    def set_columns(self, url ):        # set the main data frame from the class: 
        columns_path = url
        self.col_df = pd.read_csv(columns_path, index_col=0, sep=',', usecols=[0,1,2,3])
        return(len(self.col_df))
        
    def feed_data(self, url , type="", d_st = False, p_exp=False, pand=False):
        json_df = pd.DataFrame(columns=self.col_df.index) 
        df_entry = pd.Series(index=self.col_df.index)
        df_entry = df_entry.fillna(0) 
        comp_out_count = Counter()
        if(isinstance(url, list)):json_data = url
        else:   
            json_str=open(url).read()
            json_data = json.loads(json_str)

        for i in range(len(json_data)):
            # print(i)
            df_entry *= 0
            m = str(json_data[i]["m"])
            df_entry.name = m
            for key in json_data[i]:
                if key == "m":  
                    pass            
                else: 
                    key_wz = key #int(key) #str(int(key))
                    try:
                        ds_comp = self.col_df.loc[key_wz]
                        if p_exp == True:  #fp key - 0-102  print(ds_comp['fp'])
                            if ds_comp['fp'] == NaN: col_key = 102
                            else: 
                                col_key = int(ds_comp['fp'])
                                if col_key>101: col_key = 101
                                if col_key<0: col_key = 0
                        else: col_key = key_wz
                        # df_entry.loc[col_key]
                        df_entry[col_key] =  np.float32(json_data[i][key])
                    except: 
                        if d_st == True: 
                            print("m{}: {} - c {} not included" .format(m, key_wz))
                        # comp_out_count[key_wz] +=1
            json_df = json_df.append(df_entry,ignore_index=False)
        # print("Counter of comp. not included :")  # print(len(comp_out_count))
        if pand == True:  return json_df  
        else:           return json_df.as_matrix().tolist()  
    def read_json(url_col, url_comp, url_lab, url_json=""):
        self.comp_df = pd.read_csv(url_comp, index_col=0, sep=',', usecols=[0,1,2,3])
        print("columns: " + str(len(self.col_df)))
        self.col_df  = pd.read_csv(url_col,  index_col=0, sep=',', usecols=[0])
        print("columns: " + str(len(self.col_df)))
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
# Notes: 
#       I don't really like the fact that I am returning lists! \
#       I would like to have more flexibillity in my program (PANDAS!).
# test:  
def main():
    COM_DS     = "../_zfp/data/TFFRFLO_COM.csv"
    COL_DS     = "../_zfp/data/TFFRFLO_COL.csv"
    dataTest = {'label' : [] , 'data' :  [] }

    dataClassTest = fpDataModel()

    json_str="""[{"m":"000","100028":9}
    ,{ "m":"000000000000636978", "100000000000100028" :0.009 , "000000000000668567" :0.008 , "000000000000131503" :0.008 }
    ]"""
    json_data = json.loads(json_str)
    dataTest['data'] = dataClassTest.feed_data(json_data, d_st =True) 
    print(  dataTest['data'][1]  )

if __name__ == '__main__':
    main()

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

