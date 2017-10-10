import pandas as pd 
import tensorflow as tf 
import numpy as np 

# my notes: new customer to get benefits (france)... 
# problem sym that there was no german speaker person(project india)

import requests
import json
import sys
import os
import time
from types import *
from collections import Counter
from datetime import datetime

LOG        = "../../_zfp/LOG.txt"
LOGDIR     = "../../_zfp/data/my_graph/"
LOGDAT     = "../../_zfp/data/"
DSJ        = "/data_json.txt"
DSC        = "/datasc.csv"   
DC         = "/datac.csv"
DL         = "/datal.csv"

#---------------------------------------------------------------------
MMF        = "MOD1"
DESC       = "FRFLO"
DESC       = "FLALL"
dType      = "C1" #C1 or C4
type_sep   = False
spn        = 5000


# lr         = 0.01
# batch_size = 128
# epochs     = 100
# network    = [40 , 10]
#---------------------------------------------------------------------

LAB_DS     = LOGDAT + DESC + DL #"../../_zfp/data/FRFLO/datal.csv"
COL_DS     = LOGDAT + DESC + DC 
ALL_DSJ    = LOGDAT + DESC + DSJ 
ALL_DS     = LOGDAT + DESC + DSC 
MODEL_P    = LOGDIR + DESC + '/' + DESC +  MMF +"/model.ckpt"  

no     = 100
ni     = 0
dataE  = {'label' : [] , 'data' :  [] }
dataT  = {'label' : [] , 'data' :  [] }

def des():  return DESC+'_'+dType

def c4(df, rv=1):
    if rv == 1:
        if( df < 23 ):                  return [1,0,0,0]  #0
        elif( df >= 23 and df < 60 ):   return [0,1,0,0]  #1
        elif( df >= 60 and df < 93 ):   return [0,0,1,0]  #2
        elif( df >= 93 ):               return [0,0,0,1]  #3    
    elif rf==2: 
        if( df < 23 ):                  return 0
        elif( df >= 23 and df < 60 ):   return 1
        elif( df >= 60 and df < 93 ):   return 2
        elif( df >= 93 ):               return 3
    # elif rf==3: 
    #     if  ( df == [1,0,0,0] ):        return 0 
    #     elif( df == [0,1,0,0] ):        return 1
    #     elif( df == [0,0,1,0] ):        return 2  
    #     elif( df == [0,0,0,1] ):        return 3  
def cN(df):
    global no 
    listofzeros = [0] * no
    dfIndex = df #//nRange
    # print('{} and {}', (df,dfIndex))
    if    0 < dfIndex < no: listofzeros[dfIndex] = 1
    elif  dfIndex < 0:      listofzeros[0] = 1
    elif  dfIndex >= no:    listofzeros[no-1] = 1
    
    return listofzeros 
# Maybe I can do this with hot-encoder in sckitlearn
def cc(x, rv=1):
    global no
    if   dType == 'C4':  no = 4;   return c4(x, rv);
    elif dType == 'C1':  no = 100; return cN(x); 
def dc(df, val = 1 ): return df.index(val)

def read_data2():
    columns = pd.read_csv( tf.gfile.Open(ALL_DS), sep=None, skipinitialspace=True,  engine="python" ,skiprows=0, nrows=1)
    dst = pd.read_csv( tf.gfile.Open(ALL_DS), sep=None, skipinitialspace=True,  engine="python" , skiprows=128, nrows=128)


def read_data1( typeSep = True, filt = "", filtn = 0, pand=True): 
    global dataT; global dataE; global type_sep;
    #read excel by batchs
    dst = pd.read_csv( tf.gfile.Open(ALL_DS), sep=None, skipinitialspace=True,  engine="python" )# ,skiprows=1, nrows=2)
    dst = dst.fillna(0)
    if filt == '>':
            dst = dst[dst["FP"]>filtn]
    elif filt == '<':
        dst = dst[dst["FP"]<filtn]

    dst.insert(2, 'FP_P', dst['FP'].map(lambda x: cc(x)))  
    if typeSep == True:
        dataCol = 4 # M F T Cx
        dst_tmp = [rows for _, rows in dst.groupby('Type')]
        del dst
        #dst.loc[:,'FP_P']//[:, dataCol:]  # .as_matrix().tolist()
        dataE   =  {'label' : dst_tmp[0].loc[:,'FP_P'], 'data' : dst_tmp[0].iloc[:,dataCol:]  }
        dataT   =  {'label' : dst_tmp[1].loc[:,'FP_P'], 'data' : dst_tmp[1].iloc[:,dataCol:]  }
    else :  
        dataCol = 3 # M F Cx  
        dst = dst.sample(frac=1).reset_index(drop=True) #shuffle
        # dataA   =  {'label' : dst.loc[:,'FP_P'], 'data' : dst.loc[:,dataCol:]  }
        dataT  = {'label' : dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, dataCol:] }
        dataE  = {'label' : dst.loc[:spn,'FP_P'] , 'data' :  dst.iloc[:spn, dataCol:] }

def get_data(filt=["", 0]):
    global dataT; global dataE; global ni
    print(des())
    start = time.time()
    read_data1(typeSep = type_sep, filt=filt[0], filtn=filt[1] ) 

    # if type_sep == True: 
    #     read_data1(typeSep = type_sep, filt=filt[0], filtn=filt[1] ) 
    # else: 
    #     dataAll     = {'label' : [] , 'data' :  [] }
    #     dataAll    =  read_data1( typeSep = False, filt=filt[0], filtn=filt[1] ) 
    #     dataT  = {'label' : dataAll['label'][spn:] , 'data' :  dataAll['data'][spn:] }
    #     dataE  = {'label' : dataAll['label'][:spn] , 'data' :  dataAll['data'][:spn]  }
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataT["label"]),len(dataE["label"]),elapsed_time ))
    ni = len(dataE["data"].columns)
    print("N of columns: {}" .format( str(ni) ) )

def main():             
    get_data()
    return 
    mlp =  fpModel( MODEL_P, ni,  network , no)
    print(mlp.nn.get_nns())

if __name__ == '__main__':
    print("hi")
    main()
