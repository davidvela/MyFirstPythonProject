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

LOG        = "../../_zfp/LOG.txt"
LOGDIR     = "../../_zfp/data/my_graph/"
LOGDAT     = "../../_zfp/data/"
DSJ        = "/data_json.txt"
DSC        = "/datasc.csv"   
DC         = "/datac.csv"
DL         = "/datal.csv"

#---------------------------------------------------------------------
MMF        = "0F2CV5"
DESC       = "FRFLO"
#DESC       = "FLALL"
type_sep   = True 
lr         = 0.01
batch_size = 128
epochs     = 100
network    = [40 , 10]
#---------------------------------------------------------------------

LAB_DS     = LOGDAT + DESC + DL #"../../_zfp/data/FRFLO/datal.csv"
COL_DS     = LOGDAT + DESC + DC 
ALL_DSJ    = LOGDAT + DESC + DSJ 
ALL_DS     = LOGDAT + DESC + DSC 
MODEL_P    = LOGDIR + DESC + '/' + DESC +  MMF +"/model.ckpt"  


dataEv     = {'label' : [] , 'data' :  [] }
dataTrain  = {'label' : [] , 'data' :  [] }

def classify(x, rv=False):
    if  self.dType == 'class':     # Classification in 4 categories
        return self.classif(x, rv)
    elif self.dType == 'reg':       # Regression
        return self.regress(x)
    elif self.dType == 'classN':    # Classification in N categories  
        return self.classifN(x) 

def read_data1( typeSep = True, filt = "", filtn = 0, pand=True): 
    global dataTrain; global dataEv; global type_sep;
    dst = pd.read_csv( tf.gfile.Open(ALL_DS), sep=None, skipinitialspace=True,  engine="python" ,skiprows=1, nrows=2)
    return
    if filt == '>':
            dst = dst[dst["FP"]>filtn]
    elif filt == '<':
        dst = dst[dst["FP"]<filtn]
               
    dst.insert(2, 'FP_P', dst['FP'].map(lambda x: classify(x)))  


def read_data2(path):
    pass 

def get_data(filt=["", 0]):
    global dataTrain; global dataEv; global type_sep;
    start = time.time()
    if type_sep == True: 
        read_data1(typeSep = True, filt=filt[0], filtn=filt[1] ) 
    else: 
        dataAll     = {'label' : [] , 'data' :  [] }
        dataAll    =  read_data1( typeSep = False, filt=filt[0], filtn=filt[1] ) 
        spn = 5000
        dataTrain  = {'label' : dataAll['label'][spn:] , 'data' :  dataAll['data'][spn:] }
        dataEv     = {'label' : dataAll['label'][:spn] , 'data' :  dataAll['data'][:spn]  }
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))

def main():     
    get_data()
    return 
    mlp =  fpModel( MODEL_P, ex['n_i'],  network , ex['n_o'])
    print(mlp.nn.get_nns())

if __name__ == '__main__':
    print("hi")
    main()
