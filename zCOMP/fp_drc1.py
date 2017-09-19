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
    
    def deClassifN(self, df, val = 1 ):
        return df.index(val)
    
    def split_lab_dat(self, dst):
        cat  = dst.loc[:, self.labelCol]
        dat  = dst.iloc[:, self.dataCol:]
        if (self.toList): 
            cat = cat.as_matrix().tolist()
            dat = dat.as_matrix().tolist()
        return {'label' : cat, 'data' : dat}
    
    #Get Data
    def get_data(self, typeSep = True, pathA = "", filt = "", filtn = 0 ):
        if pathA != "":
            dst =  pd.read_csv( tf.gfile.Open(pathA), sep=None, skipinitialspace=True,  engine="python")
        else: 
            dst =  pd.read_csv( tf.gfile.Open(self.path), sep=None, skipinitialspace=True,  engine="python")
        
        dst = dst.fillna(0)
        
        if filt == '>':
            dst = dst[dst["FP"]>filtn]
        elif filt == '<':
            dst = dst[dst["FP"]<filtn]


        if self.norm != "":
            cat_n  = dst.loc[:,'FP'] 
            dst['FP'] = self.normalization( cat_n )

        if   self.dType == 'class':     # Classification in 4 categories
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
    
    
    
    def get_data2(self, colu="", datu=""):
        pass
    
    def set_columns(self, url ):        # set the main data frame from the class: 
        columns_path = url
        self.col_df = pd.read_csv(columns_path, index_col=0, sep=',', usecols=[0,1,2,3])
        return(len(self.col_df))
        
    def feed_data(self, url , type="", d_st = False):
        json_df = pd.DataFrame(columns=self.col_df.index) 
        df_entry = pd.Series(index=self.col_df.index)

        df_entry = df_entry.fillna(0) 
        comp_out_count = Counter()

        if(isinstance(url, list)):json_data = url
        else:   
            json_str=open(url).read()
            json_data = json.loads(json_str)
        
        print("start loop")
        for i in range(len(json_data)):
            print(i)
            df_entry *= 0
            m = str(json_data[i]["m"])
            df_entry.name = m
            for key in json_data[i]:
                if key == "m":  
                    pass            
                else: 
                    #key_wz = str(int(key)
                    key_wz = key
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
    
