# tensorboard --logdir=.\my_graph
# tensorboard => http://localhost:6006 
# jupyter => http://localhost:8889

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
filter     = ["", 0]
type_sep   = False

DESC       = "FRFLO"
spn        = 5000  

# DESC       = "FRALL1"
# spn        = 10000  #5000 -1 = all for training 

dType      = "C4" #C1, C2, C4
MMF        = "MODX1" #2(1) OR 5 (4)
#---------------------------------------------------------------------
MODEL_DIR  = LOGDIR + DESC + '/' + DESC + dType +  MMF +"/"  
LAB_DS     = LOGDAT + DESC + DL #"../../_zfp/data/FRFLO/datal.csv"
COL_DS     = LOGDAT + DESC + DC 
ALL_DSJ    = LOGDAT + DESC + DSJ 
ALL_DS     = LOGDAT + DESC + DSC 


nout   = 100
ninp   = 0
dataT  = {'label' : [] , 'data' :  [] } #inmutables array are faster! 
dataE  = {'label' : [] , 'data' :  [] }

def des():  return DESC+'_'+dType+"_filt:"+  filter[0]+str(filter[1])
def c2(df, rv=1):
    if rv == 1:
        if( df < 60 ):                  return [1,0]  
        elif( df >= 60 ):               return [0,1]      
    elif rf==2: 
        if( df < 60 ):                  return 0
        elif( df >= 60 ):               return 1
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
    global nout
    listofzeros = [0] * nout
    dfIndex = df #//nRange
    # print('{} and {}', (df,dfIndex))
    if    0 < dfIndex < nout:   listofzeros[dfIndex] = 1
    elif  dfIndex < 0:          listofzeros[0]       = 1
    elif  dfIndex >= nout:      listofzeros[nout-1]  = 1
    
    return listofzeros 
# Maybe I can do this with hot-encoder in sckitlearn
def cc(x, rv=1):
    global nout
    if   dType == 'C4':  nout = 4;   return c4(x, rv);
    elif dType == 'C1':  nout = 102; return cN(x); 
    elif dType == 'C2':  nout = 2;   return c2(x, rv);
def dc(df, val = 1 ): 
    try:    val = df.index(val)
    except: val = 0
    return val

# def read_data2(path):
    # columns = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=0, nrows=1)
    # columns = columns.columns
    # dst = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" , skiprows=128, nrows=128)
    # # dst = pd.read_csv( tf.gfile.Open(data_path), sep=None, skipinitialspace=True,  engine="python" )
    # dst = dst.fillna(0)
    # dst.insert(2, 'FP_P', dst['FP'].map(lambda x: cc(x)))  
    

def read_data1(data_path,  typeSep = True, filt = "", filtn = 0, pand=True, shuffle = True): 
    global dataT; global dataE;
    #read excel by batchs
    dst = pd.read_csv( tf.gfile.Open(data_path), sep=None, skipinitialspace=True,  engine="python" )
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
        if shuffle: dst = dst.sample(frac=1).reset_index(drop=True) 
        dataT  = {'label' : dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, dataCol:] }
        dataE  = {'label' : dst.loc[:spn-1,'FP_P'] , 'data' :  dst.iloc[:spn, dataCol:] }
        # dataA   =  {'label' : dst.loc[:,'FP_P'], 'data' : dst.loc[:,dataCol:]  }
        
def convert_2List(dst): return {'label' : dst["label"].as_matrix().tolist(), 'data' : dst["data"].as_matrix().tolist()}

def get_batches(batch_size):
    n_batches = len(dataT["label"])//batch_size
    # x,y = dataT["data"][:n_batches*batch_size], dataT["label"][:n_batches*batch_size]
    
    for ii in range(0, len(dataT["data"][:n_batches*batch_size] ), batch_size ):
        #convert to list! 
        yield dataT["data"][ii:ii+batch_size], dataT["label"][ii:ii+batch_size]    

def mainRead(filt=["", 0]):             
    global ninp, nout, dataT, dataE; 
    print(des())
    
    start = time.time()
    read_data1(ALL_DS, typeSep = type_sep, filt=filt[0], filtn=filt[1] ) 
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={}-{} & lenEv={}-{} time:{}" .format(len(dataT["data"]), len(dataT["label"]),len(dataE["data"]),len(dataE["label"]),elapsed_time ))

    ninp = len(dataE["data"].columns)
    print("N of columns: {}" .format( str(ninp) ) )
    dataT= convert_2List(dataT)
    dataE= convert_2List(dataE)
    return ninp, nout

def mainRead2(path, part, batch_size):  # read by partitions! 
    global ninp, nout, dataT, dataE, spn;
    start = time.time()
    columns = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=0, nrows=1)
    dst = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=part*batch_size+1, nrows=batch_size, names = columns.columns)
    dst = dst.fillna(0)
    dst.insert(2, 'FP_P', dst['FP'].map(lambda x: cc( x )))  
    
    if batch_size > spn: spn = -1
    dst = dst.sample(frac=1).reset_index(drop=True) 
    dataT  = {'label' : dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, 3:] }
    dataE  = {'label' : dst.loc[:spn-1,'FP_P'] , 'data' :  dst.iloc[:spn, 3:] }

    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={}-{} & lenEv={}-{} time:{}" .format(len(dataT["data"]), len(dataT["label"]),len(dataE["data"]),len(dataE["label"]), elapsed_time ))
   
    ninp  = len(dataT["data"].columns)

    dataT= convert_2List(dataT)
    dataE= convert_2List(dataE)
    return ninp, nout

def check_perf_CN(predv, dataEv, sk_ev=False ):
    gt3 = 0; gtM = 0; 
    # predvList = predv.tolist()
    # assert(len(predv) == len(dataEv['label']))
    print("denormalization all Evaluation : {} = {}" .format(len(predv[1]), len(dataEv["label"])))
    #for i in range(100):
    for i in range(len(dataEv["label"])):
        if (i % 1000==0): print(str(i)) #, end="__") 
        try:
            # pred_v = dc( predv.tolist()[i], np.max(predv[i]))
            pred_v = predv[1][i][0]
            data_v = dataEv['label'][i] if sk_ev  else dc( dataEv['label'][i])
            if   dType == 'C4' and pred_v != data_v:  gt3=gtM=gtM+1
            elif dType == 'C2' and pred_v != data_v:  gt3=gtM=gtM+1
            elif dType == 'C1':
                num = abs(pred_v-data_v)
                if num > 3: gt3+=1
                if num > 10: gtM+=1
        except: print("error: i={}, pred={}, data={} -- ".format(i, pred_v, data_v))
    print("Total: {} GT3: {}  GTM: {}".format(len(predv[1]), gt3, gtM)) 
    return gt3, gtM 

def feed_data(dataJJ, p_abs, d_st = False, p_exp=False, pand=False, p_col = False):
    indx=[];   index_col=0 if p_abs else 2 #abs=F => 2 == 6D
 
    # col_df = pd.read_csv(COL_DS, index_col=index_col, sep=',', usecols=[0,1,2,3])    
    col_df = pd.read_csv(COL_DS, index_col=index_col, sep=',', usecols=[0,1,2,3])    
    col_df = col_df.fillna(0)
    print("input-no={}".format( len(col_df )))
    
    if p_exp:   indx.append(i for i in range(103))
    else:       indx = col_df.index
    
    if p_col: 
        dataTest_label = []
        dataJJ = "["
        for i in range(len(col_df)): 
            dataTest_label.append( cc( int(  col_df.iloc[i]["fp"]  )  )) 
            dataJJ += '{"m":"'+str(i)+'",'+'"'+str(col_df.iloc[i].name)+'"'+":1},"
        dataJJ += '{"m":"0"}]';  dataTest_label.append(cc(0))
        # dataJJ += ']'
        dataJJ = json.loads(dataJJ)

    json_df  = pd.DataFrame(columns=indx); df_entry = pd.Series(index=indx)
    df_entry = df_entry.fillna(0) 
   
    ccount = Counter()
    if(isinstance(dataJJ, list)):json_data = dataJJ
    else: json_str=open(dataJJ).read();  json_data = json.loads(json_str)
    # for i in range(20):
    for i in range(len(json_data)): # print(i)
        df_entry *= 0
        m = str(json_data[i]["m"])
        df_entry.name = m
        for key in json_data[i]:
            if key == "m": pass            
            else: 
                key_wz = key if p_abs else (int(key))  #str(int(key)) FRFLO - int // FRALL str!
                try: #filling of key - experimental or COMP 
                    ds_comp = col_df.loc[key_wz]
                    if p_exp == True:  #fp key - 0-102   
                        co = str(ds_comp['FP'])
                        if co == 'nan':  col_key = 102
                        else: 
                            col_key = int(ds_comp['FP'])
                            if col_key>101: col_key = 101
                            if col_key<0: col_key = 0
                    else: col_key = key_wz      
                    # df_entry.loc[col_key]
                    df_entry[col_key] =  np.float32(json_data[i][key])
                except: 
                    if d_st: print("m:{}-c:{} not included" .format(m, key_wz)); ccount[key_wz] +=1

        json_df = json_df.append(df_entry,ignore_index=False)
        if i % 1000 == 0: print("cycle: {}".format(i))
    print("Counter of comp. not included :"); print(ccount) # print(len(ccount))

    if p_col: return json_df.as_matrix().tolist(), dataTest_label
    else: 
        if pand:  return json_df  
        else:     return json_df.as_matrix().tolist()  

def testsJ(excel):
    print("tests JSON")    
    dataAll = {'label' : [] , 'data' :  [] }
    json_flag = True    
    if json_flag: 
        json_str = '''[{ "m":"8989", "c1" :0.5, "c3" :0.5  },
                    { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
        json_data = json.loads(json_str)  #;print(json_data[0]['m'])
    else: json_data = ALL_DSJ
  
    start = time.time()
    dataAll['data'] = feed_data(json_data, p_abs = True, pand=True, d_st=True,  p_exp=False);
    elapsed_time = float(time.time() - start)
    # TO DO: separate between training and evaluation! 
    
    print("data read - time:{}" .format(elapsed_time ))
    if excel == True:# Create the excel with the new layout!  
        writer = pd.ExcelWriter(LOGDAT+'pandas.xlsx')
        dataAll['data'].to_excel(writer, sheet_name='Sheet1')
        writer.save()
        print("JSON downloaded into excel! ")

if __name__ == '__main__':
    print("hi1")
    # mainRead()   
    ninp, nout  = mainRead2(ALL_DS, 1, 2 )  
    #testsJ(False)
