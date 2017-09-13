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
from fp_drc1     import fpDataModel


LOG        = "../../_zfp/LOG.txt"
LOGDIR     = "../../_zfp/data/my_graph/"
LOGDAT     = "../../_zfp/data/"

DESC       = "FRFLO"

DSJ        = "/data_json.txt"
DSC        = "/TFFRFLO_ALSN.csv"   
DC         = "/colcom.csv"
DL         = "/datal.csv"

LAB_DS     = LOGDAT + DESC + DL
COL_DS     = LOGDAT + DESC + DC  #"../../_zfp/data/FRFLO/colcom.csv"
ALL_DSJ    = LOGDAT + DESC + DSJ #"../../_zfp/data/FRFLO/datac.csv"

ALL_DS     = LOGDAT + DESC + DSC #"../../_zfp/data/FRFLO/TFFRFLO_ALSN.csv"

get_datat = [ "", "class", "reg", "classN" ]


def tests_json():
    print("tests JSON");     n_classes   = 100    
    # col_df = pd.read_csv(COL_DS, index_col=0, sep=',', usecols=[0,1,2,3])  #; print(col_df)

    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_C', 
                            dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    # get data from type = T, E 
    # print("start reading...")
    # start = time.time()
    # dataTrain,  dataEv =  dataClass.get_data(pathA=ALL_DS ) 
    # elapsed_time = float(time.time() - start)
    # print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))

    dataAll  = {'label' : [] , 'data' :  [] }
    n_input2 = dataClass.set_columns(COL_DS)
    print("input-no={}".format( n_input2))
    json_str = '''[{ "m":"8989", "c1" :0.5 },
                { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
    # json_data = json.loads(json_str)  #;print(json_data[0]['m'])

    print(ALL_DSJ)
    start = time.time()
    dataAll['data'] = dataClass.feed_data(ALL_DSJ);
    elapsed_time = float(time.time() - start)
    # separate between training and evaluation! 
    print("data read - time:{}" .format(elapsed_time ))

    

    # Create the excel with the new layout! 


def tests_classif():
    print("tests C4");     n_classes   = 4    


def tests_classifN_100(filt=''):
    n_classes   = 100    
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType=get_datat[3], labelCol = 'FP_C', 
                            dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    # get data from type = T, E 
    print("start reading...")
    start = time.time()

    filt = filt 
    print("tests C100 - filter:" + filt );    
    # def get_data(self, typeSep = True, pathA = "", filter = ""):
    dataTrain,  dataEv =  dataClass.get_data(pathA=ALL_DS, filter=filt ) 
    
    
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))
        
if __name__ == '__main__':
    # tests_json()
    
    filters = ["", ">23",">60"]
    
    for i in range(len(filters)):
        tests_classifN_100(filters[i])
    
    
    
    # tests_classif()