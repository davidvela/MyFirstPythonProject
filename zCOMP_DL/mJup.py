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

def des():  return DESC+'_'+dType+"_filt:"+  filter[0]+str(filter[1])
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
def cc(x, rv=1):
    global nout
    if   dType == 'C4':  return c4(x, rv);
    elif dType == 'C1':  return cN(x); 
def dc(df, val = 1 ):    return df.index(val)
#def convert_2List(dst): return {'label' : dst["label"].as_matrix().tolist(), 'data' : dst["data"].as_matrix().tolist()}
def get_batches(batch_size):
    n_batches = len(dataT["label"])//batch_size
    # x,y = dataT["data"][:n_batches*batch_size], dataT["label"][:n_batches*batch_size]
    
    for ii in range(0, len(dataT["data"][:n_batches*batch_size] ), batch_size ):
        #convert to list! 
        yield dataT["data"][ii:ii+batch_size], dataT["label"][ii:ii+batch_size]   
def normalize():     dst[:, 'FP_P'] = dst['FP'].map(lambda x: cc( x ))
def readData(all = True, shuffle = True, path, part, batch_size ):  # read by partitions!   
    global  spn, dst;
    start = time.time()
    if all:  dst = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" )
    else:     
        columns = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=0, nrows=1)
        dst = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=part*batch_size+1, nrows=batch_size, names = columns.columns)
    
    dst = dst.fillna(0)
    if shuffle: dst = dst.sample(frac=1).reset_index(drop=True) 
    dst.insert(2, 'FP_P', dst['FP'] )  
    elapsed_time = float(time.time() - start)
    print("data read - {} - time:{}" .format(len(dst), elapsed_time ))

    # #dst.insert(2, 'FP_P', dst['FP'].map(lambda x: cc( x )))  
    # if batch_size > spn: spn = -1
    # dst = dst.sample(frac=1).reset_index(drop=True) 
    # dataT  = {'label' : dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, 3:] }
    # dataE  = {'label' : dst.loc[:spn-1,'FP_P'] , 'data' :  dst.iloc[:spn, 3:] }
    #print("data read - lenTrain={}-{} & lenEv={}-{} time:{}" .format(len(dataT["data"]), len(dataT["label"]),len(dataE["data"]),len(dataE["label"]), elapsed_time ))
    # dataT= convert_2List(dataT)
    # dataE= convert_2List(dataE)
def check_perf_CN(predv, dataEv, sk_ev=False ):
    gt3 = 0; gtM = 0; 
    # predvList = predv.tolist()
    # assert(len(predv) == len(dataEv['label']))
    print("denormalization all Evaluation : {} = {}" .format(len(predv), len(dataEv["label"])))
    #for i in range(100):
    for i in range(len(predv)):
        if (i % 1000==0): print(str(i)) #, end="__") 
        try:
            pred_v = dc( predv.tolist()[i], np.max(predv[i]))
            data_v = dataEv['label'][i] if sk_ev  else dc( dataEv['label'][i])
            if   dType == 'C4' and pred_v != data_v:  gt3=gtM=gtM+1
            elif dType == 'C1':
                num = abs(pred_v-data_v)
                if num > 3: gt3+=1
                if num > 10: gtM+=1
        except: print("error: i={}, pred={}, data={} -- ".format(i, pred_v, data_v))
    print("Total: {} GT3: {}  GTM: {}".format(len(predv), gt3, gtM)) 
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
#---------------------------------------------------------------------
def get_nns(): return str(ninp)+'*'+str(h[0])+'*'+str(h[1])+'*'+str(nout)
def logr(datep = '' , time='', it=1000, nn='', typ='TR', DS='', AC=0, num=0, AC3=0, AC10=0, desc='', startTime=''):
    if desc == '': print("Log not recorded"); return 
    LOG = "../../_zfp/LOGT2.txt"
    f= open(LOG ,"a+") #w,a,
    if datep != '':   dats = datep
    else:             dats = datetime.now().strftime('%d.%m.%Y') 
    if time != '':    times = time
    else:             times = datetime.now().strftime('%H:%M:%S') 

    line =  datetime.now().strftime('%d.%m.%Y') + '\t' + times 
    line = line + '\t' + str(it) + '\t'+  get_nns() +  '\t' + str(lr)
    line = line + '\t' + typ 
    line = line + '\t' + str(DS) + '\t' + str(AC) + '\t' + str(num) + '\t' + str(AC3) + '\t' +  str(AC10) + '\t' + desc 
    line = line + '\t' + str(batch_size) + '\t' +  startTime + '\n' #new

    f.write(line);  f.close()
    print("___Log recorded")    
def restore_model(sess):   
    saver= tf.train.Saver() 
    print("Model restored from file: %s" % model_path)
    saver.restore(sess, model_path)
#---------------------------------------------------------------------
def train(it = 100, disp=50, batch_size = 128):     #dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, 3:]    total_batch  = int(len( dst.loc[spn:]  ) / batch_size)
    print("____TRAINING...")
    display_step =  disp 

    dataTest = {'label' : [] , 'data' :  [] };
    dataTest['data'], dataTest['label']  = md.feed_data("", p_abs=False , d_st=True, p_col=True)   
    md.dataT['data'].append(dataTest['data']) ;     md.dataT['label'].append(dataTest['label']) 
    print("data read - lenTrain={}-{} & lenEv={}-{}" .format(len(md.dataT["data"]), len(md.dataT["label"]),len(md.dataE["data"]),len(md.dataE["label"]) ))

    total_batch  = int(len(md.dataT['label']) / batch_size)
    
    with tf.name_scope("xent"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        tf.summary.scalar("xent", cost)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    startTime = datetime.now().strftime('%H:%M:%S')
    summ = tf.summary.merge_all()
    saver= tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore_model(sess)  #Run if I want to retrain an existing model
        start = time.time()
        for i in range(it):            
            for ii, (xtb,ytb) in enumerate(md.get_batches(batch_size) ):
                # xtb, ytb = dc.next_batch(batch_size, dataT['data'], dataT['label'])
                sess.run(optimizer, feed_dict={x: xtb, y: ytb})
                if ii % display_step ==0: #record_step == 0:
                    [train_accuracy] = sess.run([accuracy], feed_dict={x: xtb, y: ytb }) 
                    elapsed_time = float(time.time() - start)
                    reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                    rp_s = str(reviews_per_second)[0:5]
                    tr_ac = str(train_accuracy)[:5]  
                    print('Epoch: {} batch: {} / {} - %Speed(it/disp_step): {} - tr_ac {}' .format(i, ii, total_batch, rp_s, tr_ac ))
                    #writer.add_summary(s, i)
            ev_ac = str(sess.run(accuracy, feed_dict={x: md.dataE['data'], y: md.dataE['label']}))[:5] 
            print("E Ac:", ev_ac)
        
        # tr_ac = str(sess.run(accuracy, feed_dict={x: md.dataT['data'], y: md.dataT['label']}))[:5] 
        print("T Ac:", tr_ac)
        
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path) 
    print("Optimization Finished!")

    logr( it=it, typ='TR', DS=md.DESC, AC=tr_ac,num=len(md.dataT["label"]), AC3=0, AC10=0, desc=md.des(), startTime=startTime )
    logr( it=it, typ='EV', DS=md.DESC, AC=ev_ac,num=len(md.dataE["label"]), AC3=0, AC10=0, desc=md.des() )
#---------------------------------------------------------------------
def evaluate( ): 
    print("_____EVALUATION...")
    startTime = datetime.now().strftime('%H:%M:%S')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_model(sess)
        # test the model
        tr_ac = str(sess.run( accuracy, feed_dict={ x: md.dataT['data'],  y: md.dataT['label']}) )[:5]  
        ev_ac = str(sess.run( accuracy, feed_dict={ x: md.dataE['data'],  y: md.dataE['label'][:md.spn]   }))[:5] 
        print("Training   Accuracy:", tr_ac )
        print("Evaluation Accuracy:", ev_ac )
        # xtp1.append(dataTest['data'][i]);    ytp1.append(dataTest['label'][i])
        predv, softv = sess.run([prediction, softmaxT], feed_dict={x: md.dataE['data']  }) # , y: md.dataE['label'] 
        # maxa = sess.run([prediction], feed_dict={y: predv })
    print("Preview the first predictions:")
    for i in range(20):
        print("RealVal: {}  - PP value: {}".format( md.dc( md.dataE['label'][i]), 
                                                    md.dc( predv.tolist()[i], np.max(predv[i]))  ))
    gt3, gtM = md.check_perf_CN(softv, md.dataE, False) #predv
    logr(  it=0, typ='EV', AC=ev_ac,DS=md.DESC, num=len(md.dataE["label"]), AC3=gt3, AC10=gtM, desc=md.des(), startTime=startTime )
#---------------------------------------------------------------------
def tests(url_test = 'url', p_col=False):  
    print("_____TESTS...")    
    
    # Load test data 
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    if p_col: dataTest['data'], dataTest['label']  = md.feed_data("", p_abs=False , d_st=True, p_col=True)   
    else: 
        if url_test != 'url':  
            json_data = url_test + "data_json6.txt"
            tmpLab = pd.read_csv(url_test + "datal6.csv", sep=',', usecols=[0,1])    
            tmpLab = tmpLab.loc[:,'fp']
            abstcc = False
        else: 
            json_str, tmpLab = get_data_test("FRALL")
            json_data = json.loads(json_str)
            abstcc = True
            md.DESC =  'matnrList...'
        
        dataTest['data']  = md.feed_data(json_data, p_abs=abstcc , d_st=True)
        
        dataTest['label'] = []
        [dataTest['label'].append( md.cc(x) ) for x in tmpLab ]
    # Predict data 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_model(sess)
        # predv = sess.run( prediction, feed_dict={x: dataTest['data']}) 
        ts_acn = '0'
        ts_acn, predv, sf = sess.run( [accuracy, prediction, softmaxT], feed_dict={x: dataTest['data'], y: dataTest['label']}) 
        ts_ac = str(ts_acn) 
        print("test ac = {}".format(ts_ac))
    # print(dataTest['label']);     print(sf)
    range_ts = len(predv) if len(predv)<20 else 20
    for i in range( range_ts ):
        # print("RealVal: {}  - PP value: {}".format( md.dc( dataTest['label'][i]), md.dc( predv.tolist()[i], np.max(predv[i]))  ))  
        print("{} RealVal: {} - {} - PP: {} PR: {}".format( i, md.dc( dataTest['label'][i]), sf[1][i][0],  sf[1][i], sf[0][i]   ))

    # return
    gt3, gtM = md.check_perf_CN(predv, dataTest, False)
    logr( it=0, typ='TS', DS=md.DESC, AC=ts_acn ,num=len(dataTest["label"]),  AC3=gt3, AC10=gtM, desc=md.des() )  

    outfile = '../../_zfp/data/export2' 
    np.savetxt(outfile + '.csv', sf[1], delimiter=',')
    np.savetxt(outfile + 'PRO.csv', sf[0], delimiter=',')
#---------------------------------------------------------------------

spn        = 5000  #5000 -1 = all for training 
h          = [100 , 40]
# h          = [40 , 10]

DESC       = "FRFLO"
# DESC       = "FRALL1"
dType      = "C1" #C1 or C4
MMF        = "MODJJ1" #2(1) OR 5 (4)

epochs     = 120
disp       = 5
batch_size = 128
lr         = 0.0001
#---------------------------------------------------------------------
model_path = LOGDIR + DESC + '/' + DESC +  MMF +"/model.ckpt"   
LOG        = "../../_zfp/LOG.txt"
LOGDIR     = "../../_zfp/data/my_graph/"
LOGDAT     = "../../_zfp/data/"
DSJ        = "/data_json.txt"
DSC        = "/datasc.csv"   
DC         = "/datac.csv"
DL         = "/datal.csv"
LAB_DS     = LOGDAT + DESC + DL #"../../_zfp/data/FRFLO/datal.csv"
COL_DS     = LOGDAT + DESC + DC 
ALL_DSJ    = LOGDAT + DESC + DSJ 
ALL_DS     = LOGDAT + DESC + DSC 



readData()

def build_network2(is_train=False):     # Simple NN - with batch normalization (high level)
    global ninp, nout
    kp = 0.9;     nout   = 100;     ninp   = 0
    
    if   dType == 'C4':  nout = 4;   
    elif dType == 'C1':  nout = 102;
    ninp  = len(dst.columns) - 2 


    x = tf.placeholder(tf.float32,   shape=[None, ninp], name="x")
    y = tf.placeholder(tf.int16,     shape=[None, nout], name="y")

    # h0 = tf.layers.dense( x, h[0], activation=tf.nn.relu,  name )
    h0 = tf.layers.dense( x, h[0], use_bias=False, activation=None )
    h0 = tf.layers.batch_normalization(h0, training=is_train)
    h0 = tf.nn.relu(h0)
    # h0 = tf.nn.dropout(h0, kp)
    
    h1 = tf.layers.dense( h0, h[1], use_bias=False, activation=None )
    h1 = tf.layers.batch_normalization(h1, training=is_train)
    h1 = tf.nn.relu(h1)
    # h1 = tf.nn.dropout(h1, kp)
    
    out = tf.layers.dense( h1, nout, use_bias=False, activation=None )
    out = tf.layers.batch_normalization(out, training=is_train)
    out = tf.nn.relu(out)
    # out = tf.nn.dropout(h0, kp)
 
    # softmaxT = tf.nn.softmax(out)
    softmaxT = tf.nn.top_k(tf.nn.softmax(out), 4)
            
    prediction=tf.reduce_max(y,1)
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return out, accuracy, softmaxT, x,y 
print( get_nns() )
#---------------------------------------------------------------------
prediction, accuracy, softmaxT x,y = build_network2()





def mainRun(): 
    train(epochs, disp, batch_size)
    #evaluate( )
    # url_test = "../../_zfp/data/FREXP1/" ; md.DESC     = "FREXP1_6"
    # tests(url_test, p_col=False  )
    #    dataT  = {'label' : dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, dataCol:] }
    #    dataE  = {'label' : dst.loc[:spn-1,'FP_P'] , 'data' :  dst.iloc[:spn, dataCol:] }

    print("___The end!")

if __name__ == '__main__':
    mainRun()




