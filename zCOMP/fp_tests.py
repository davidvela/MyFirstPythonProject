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

from fp_drc2     import fpDataModel
from fp_drn1     import  *

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

    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_P', 
                            dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    # get data from type = T, E 
    # print("start reading...")
    # start = time.time()
    # dataTrain,  dataEv =  dataClass.get_data(pathA=ALL_DS ) 
    # elapsed_time = float(time.time() - start)
    # print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))
    n_input2 = dataClass.set_columns(COL_DS)
    print("input-no={}".format( n_input2))
    dataAll     = {'label' : [] , 'data' :  [] }
    json_str = '''[{ "m":"8989", "c1" :0.5, "c3" :0.5  },
                { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
    # json_data = json.loads(json_str)  #;print(json_data[0]['m'])
    print(ALL_DSJ)
    start = time.time()
    dataAll['data'] = dataClass.feed_data(ALL_DSJ, pand=True);
    elapsed_time = float(time.time() - start)
    # separate between training and evaluation! 
    print("data read - time:{}" .format(elapsed_time ))
    # Create the excel with the new layout! 


    return dataClass
def tests_classif(filt=["", 0]):
    print("tests C4");     n_classes   = 4    
    # filt = ["<", 80] 
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType='class', labelCol = 'FP_P', 
                             dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    start = time.time()
    dataTrain,  dataEv =  dataClass.get_data( filt=filt[0], filtn=filt[1] ) 
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))
    return dataClass, dataTrain,  dataEv
def tests_classifN_100(filt=["", 0]):
    print("tests C4");     n_classes   = 100   
    filt = filt 
    print("tests C100 - filter:" + filt[0] + str(filt[1]) ) 
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType=get_datat[3], labelCol = 'FP_P', 
                             dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    # get data from type = T, E 
    print("start reading...")
    start = time.time()
    # def get_data(self, typeSep = True, pathA = "", filter = ""):
    dataTrain,  dataEv =  dataClass.get_data(pathA=ALL_DS, filt=filt[0], filtn=filt[1] ) 
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))
    return dataClass
    # recording log: ReadFile - total number - 
    # improve the batch reading - I am reading batchs (128) randomly ... I can do it sequential too...
    # test the results ...  implement the class 


#





#
def build_desc(des='C4'):
    return des + "  filt  "+str(40)+'  model  '
def main():
    
    executions = [
        { 'n':3, 'des':'C4 - FRAFLO',  'filters' :[ ["", 0]], 'n_input':1814, 'n_output':4, 
          'dT':'class', 'batch_size':128, 'lr':0.01, 'it':1000, 'path':ALL_DS }
    ]
    exec = executions[0]
    num = 3

    if num == 1: dc = tests_json()
    if num == 2: 
        #filters = [ ["", 0], ['>', 60], ['<', 93]]
        filters = [ ["", 0]]
        for i in range(len(filters)):
            dc = tests_classifN_100(filters[i])
      
    if num == 3: dc, dt, de = tests_classif()
    
    
    #dc = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType='class', labelCol = 'FP_C', dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    nc  = fpNN(n_input=1814, layers=2, hidden_nodes = [256 , 256],lr = 0.01, min_count = 10, polarity_cutoff = 0.1, output=4)
    print("network built")
    # model  FLO  0F2CV4 - C4   
    # model  FLO  0F2CV5 - C100   
    #     
    model_path  = LOGDIR + "0F2CV4/model.ckpt"      
    mlp =  fpModel(nc, dc, model_path)
    print(mlp.get_nns())
    # mlp.dummy3(); return;
    # _______DEFINITION train(self, dataClass, dataTrain, dataEv, it = 10000, desc=''):
    # mlp.train(dataTrain=dt, dataEv = de, it=1000, desc='C4 - FRAFLO')
    #_______DEFINITION def evaluate(self, dataTrain, dataEv,  desc='' )
    # mlp.evaluate(dataTrain=dt, dataEv = de, desc='C4 - FRAFLO')
    # return
    #_______DEFINITION test(self, dataClass, p_json_str=0, p_label=0, desc='')
    json_str = '''[{ "m":"8989", "c1" :0.5 }, { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
    label = [100,60]
    desc='json-desc'  
    mlp.test(COL_DS ,json_str, label, 'C100 FRAFLO - c1; c3c4')  

if __name__ == '__main__':
    main()
    # print(build_desc('test'))

