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


MMF        = "0F2CV5"
DESC       = "FRFLO"
type_sep   = True 
lr         = 0.01
batch_size = 128
epochs     = 100

LAB_DS     = LOGDAT + DESC + DL #"../../_zfp/data/FRFLO/datal.csv"
COL_DS     = LOGDAT + DESC + DC 
ALL_DSJ    = LOGDAT + DESC + DSJ 
ALL_DS     = LOGDAT + DESC + DSC 
MODEL_P    = LOGDIR + DESC + '/' + DESC +  MMF +"/model.ckpt"  


dataEv     = {'label' : [] , 'data' :  [] }
dataTrain  = {'label' : [] , 'data' :  [] }

def get_data1(pathA, typeSep = True, filt = "", filtn = 0, pand=True): 
    dst = pd.read_csv( tf.gfile.Open(pathA), sep=None, skipinitialspace=True,  engine="python")
    if filt == '>':
            dst = dst[dst["FP"]>filtn]
    elif filt == '<':
        dst = dst[dst["FP"]<filtn]
    if self.norm != "": # normalization not longer used since I am using always classification! 
        cat_n  = dst.loc[:,'FP'] 
        dst['FP'] = self.normalization( cat_n )               
    dst.insert(2, 'FP_P', dst['FP'].map(lambda x: self.classify(x)))  


def get_data2(path):
    pass 

def tests_classif(filt=["", 0]):
    global dataTrain; global dataEv; global type_sep;
    start = time.time()
    if type_sep == True: 
        dataTrain,  dataEv =  dataClass.get_data(pathA=ALL_DS, typeSep = True, filt=filt[0], filtn=filt[1] ) 
    else: 
        dataAll     = {'label' : [] , 'data' :  [] }
        dataAll    =  dataClass.get_data( typeSep = False, filt=filt[0], filtn=filt[1] ) 
        spn = 5000
        dataTrain  = {'label' : dataAll['label'][spn:] , 'data' :  dataAll['data'][spn:] }
        dataEv     = {'label' : dataAll['label'][:spn] , 'data' :  dataAll['data'][:spn]  }
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))

def main(): 
    
mlp =  fpModel( MODEL_P, ex['n_i'],  [40 , 10] , ex['n_o'])
print(mlp.nn.get_nns())

if __name__ == '__main__':
    main()