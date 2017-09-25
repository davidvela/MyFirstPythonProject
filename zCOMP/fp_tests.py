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

DSJ        = "/data_json.txt"
DSC        = "/datasc.csv"   
DC         = "/datac.csv"
DL         = "/datal.csv"

def tests_json(excel, ex):
    print("tests JSON");     n_classes   = 100  ; print(ALL_DSJ)  
    # col_df = pd.read_csv(COL_DS, index_col=0, sep=',', usecols=[0,1,2,3])  #; print(col_df)
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_P', 
                            dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    # get data from type = T, E 
    # print("start reading...")
    # start = time.time()
    # dataTrain,  dataEv =  dataClass.get_data(pathA=ALL_DS ) 
    # elapsed_time = float(time.time() - start)
    # print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))
    n_input2 = dataClass.set_columns(COL_DS, p_abs = False)
    print("input-no={}".format( n_input2))
    dataAll     = {'label' : [] , 'data' :  [] }
    # json_str = '''[{ "m":"8989", "c1" :0.5, "c3" :0.5  },
    #             { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
    # json_data = json.loads(json_str)  #;print(json_data[0]['m'])
    start = time.time()
    dataAll['data'] = dataClass.feed_data(ALL_DSJ, pand=ex['returnPandas'], d_st=ex['display_status'],  p_exp=ex['experimental']);
    elapsed_time = float(time.time() - start)
    # TO DO: separate between training and evaluation! 


    print("data read - time:{}" .format(elapsed_time ))
    if excel == True:# Create the excel with the new layout!  
        writer = pd.ExcelWriter(LOGDAT+'pandas.xlsx')
        dataAll['data'].to_excel(writer, sheet_name='Sheet1')
        writer.save()
        print("JSON downloaded into excel! ")
    return dataClass
def tests_classif(filt=["", 0]):
    print("tests C4 - filter:" + filt[0] + str(filt[1]) );   n_classes   = 4    
    # filt = ["<", 80] 
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType='class', labelCol = 'FP_P', 
                             dataCol = 4,   nC=n_classes, nRange=1, toList = False, pathFile= ex['path'] )
    start = time.time()
    dataTrain,  dataEv =  dataClass.get_data( typeSep = True, filt=filt[0], filtn=filt[1] ) 
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))
    return dataClass, dataTrain,  dataEv
def tests_classifN_100(filt=["", 0]):
    print("tests C100 - filter:" + filt[0] + str(filt[1]) );   n_classes   = 100   
    filt = filt 
    dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_P', 
                             dataCol = 4,   nC=n_classes, nRange=1, toList = False, pathFile= ex['path']  )
    # get data from type = T, E 
    print("start reading...")
    start = time.time()
    # def get_data(self, typeSep = True, pathA = "", filter = ""):
    dataTrain,  dataEv =  dataClass.get_data(pathA=ALL_DS, typeSep = True, filt=filt[0], filtn=filt[1] ) 
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={} - lenTests={} - time:{}" .format(len(dataTrain["label"]),len(dataEv["label"]),elapsed_time ))
    return dataClass, dataTrain,  dataEv
    # recording log: ReadFile - total number - 
    # improve the batch reading - I am reading batchs (128) randomly ... I can do it sequential too...
    # test the results ...  implement the class 
#
def build_desc(ex):
    return  ex['des'] + "_" +  ex['path'] + "_filt:"+  ex['filter'][0][0]+str(ex['filter'][0][1])
executions = [
    # {   'dType':'json', 'downExcel':True , 'des':'JSON Input100- execl generation FRAFLO',
    #     'path':'FRFLO', 'experimental':True , 'display_status':True, 'returnPandas':True , 
    #     'jsonFile':"/data_json2.txt"},   
    
    # _____________________C4
    # {   'dType':'class', 'des':'C4',    'path':'FRFLO', 'it':1000, 'dispIt':0.025,
    #     'filter' :[["", 0]], 'n_i':1814, 'batch_size':128,'n_o':4, 
    #     'lr':0.01, 'model': "0F2C41" },
        
    # {   'dType':'class', 'des':'C4',    'path':'FRFLO', 'it':1000, 'dispIt':0.025,
    #     'filter' :[[">", 60]], 'n_i':1814, 'batch_size':128,'n_o':4, 
    #     'lr':0.01, 'model': "0F2C41" },

    #  {  'dType':'class', 'des':'C4',    'path':'FRFLO', 'it':1000, 'dispIt':0.025,
    #     'filter' :[["<", 93]], 'n_i':1814, 'batch_size':128,'n_o':4, 
    #     'lr':0.01, 'model': "0F2C41" }, 

    # _____________________C4 
    {   'dType':'classN', 'des':'C100','path':'FRFLO' , 'it':10000, 'dispIt':0.025,
        'filter' :[["", 0]], 'n_i':1814, 'batch_size':128,'n_o':100,
        'lr':0.01, 'model': "0F2CV5" },

    {   'dType':'classN', 'des':'C100','path':'FRFLO' , 'it':10000, 'dispIt':0.025,
        'filter' :[[">", 60]], 'n_i':1814, 'batch_size':128,'n_o':100,
        'lr':0.01, 'model': "0F2CV5" },

    {   'dType':'classN', 'des':'C100','path':'FRFLO' , 'it':10000, 'dispIt':0.025,
        'filter' :[["<", 93]], 'n_i':1814, 'batch_size':128,'n_o':100,
        'lr':0.01, 'model': "0F2CV5" },
    
    # {    'dType':'classN', 
    #          }
]
ex = executions[0]

LOG        = "../../_zfp/LOG.txt"
LOGDIR     = "../../_zfp/data/my_graph/"
LOGDAT     = "../../_zfp/data/"
DESC       = "test"

LAB_DS     = LOGDAT + DESC + DL #"../../_zfp/data/FRFLO/datal.csv"
COL_DS     = LOGDAT + DESC + DC 
ALL_DSJ    = LOGDAT + DESC + DSJ 
ALL_DS     = LOGDAT + DESC + DSC 
#

def main1():
    global ex;      #ex       = executions[1]
    
    for i in range(len(executions)):
        ex        = executions[i]
        ex['des'] = build_desc(ex)
        global LAB_DS;  LAB_DS   = LOGDAT + ex['path'] + "/datal.csv"
        global COL_DS;  COL_DS   = LOGDAT + ex['path'] + "/datac.csv" 
        global ALL_DS;  ALL_DS   = LOGDAT + ex['path'] + "/datasc.csv"  
        if ex['dType'] == 'json':       #JSON
            global ALL_DSJ; ALL_DSJ  = LOGDAT + ex['path'] + ex['jsonFile']
            dc = tests_json(ex['downExcel'], ex); 
        if ex['dType'] == 'classN':     #C100
            #filters = [ ["", 0], ['>', 60], ['<', 93]]
            filters = ex['filter']
            for i in range(len(filters)):
                dc, dt, de  = tests_classifN_100(filters[i])
                main2(dc, dt, de )
        if ex['dType'] == 'class':      # C4
            dc, dt, de = tests_classif(ex['filter'][0] )
        main2(dc, dt, de )
    return

def main2(dc, dt, de ):
    #dc = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType='class', labelCol = 'FP_C', dataCol = 4,   nC=n_classes, nRange=1, toList = True )
    #dt,  de =  dataClass.get_data( filt=filt[0], filtn=filt[1] ) 
    dt_l = dc.convert_2List(dt)  # dt.as_matrix().tolist()
    de_l = dc.convert_2List(de)
    # print(len(dt["data"].columns)) #1814
    ex['n_i'] = len(dt["data"].columns)
    nc  = fpNN(n_input=ex['n_i'], layers=2, hidden_nodes = [256 , 256],lr = ex['lr'] , min_count = 10, polarity_cutoff = 0.1, output=ex['n_o'] )
    print("network built") 
    model_path  = LOGDIR + ex['model'] +"/model.ckpt"      
    mlp =  fpModel(nc, dc, model_path)
    print(mlp.get_nns())
    # mlp.dummy3(); return;
    # _______DEFINITION train(self, dataClass, dataTrain, dataEv, it = 10000, desc=''):
    mlp.train(dataTrain=dt_l, dataEv = de_l, it=ex['it'], disp=ex['dispIt'], desc=ex['des'])
    #_______DEFINITION def evaluate(self, dataTrain, dataEv,  desc='' )
    mlp.evaluate(dataTrain=dt_l, dataEv = de_l, desc=ex['des'])
    # _______DEFINITION test(self, dataClass, p_json_str=0, p_label=0, desc='')
    json_str = '''[{ "m":"8989", "c1" :0.5 }, { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
    label = [99, 60]
    desc='json-desc'  
    # mlp.test(COL_DS, True, json_str, label, ex['des'] ) #'C100 FRAFLO - c1; c3c4')  
    #_____________ test using some part of ev. 


    # push to the next level - 
    

if __name__ == '__main__':
    main1()
    # print(build_desc(ex))

