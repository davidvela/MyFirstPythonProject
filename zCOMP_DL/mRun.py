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
import mData as md

# READ DATA -------------------------------------------------
print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
# md.spn = 200
# ninp, nout  = md.mainRead()
# md.DESC     = "FREXP"
ninp, nout  = md.mainRead2(md.ALL_DS, 1, 2 ) # For testing I am forced to used JSON - column names and order may be different! 
print("___Data Read!")

model_path    = md.MODEL_DIR 
lr         = 0.0001
h          = [100 , 40]
# h          = [40 , 10]
epochs     = 40
disp       = 5
batch_size = 128

def get_nns(): return str(ninp)+'*'+str(h[0])+'*'+str(h[1])+'*'+str(nout)
def logr(datep = '' , time='', it=1000, nn='', typ='TR', DS='', AC=0, num=0, AC3=0, AC10=0, desc='', timeStart=''):
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
    line = line + '\t' + str(batch_size) + '\t' +  timeStart + '\n' #new

    f.write(line);  f.close()
    print("___Log recorded")    

# NETWORK-----------------------------------------------------
print( get_nns() )
x = tf.placeholder(tf.float32,   shape=[None, ninp], name="x")
y = tf.placeholder(tf.int16,     shape=[None, nout], name="y")

def build_network3():                   # RNN logic mix with NN - 
    pass
def build_network2(is_train=False):     # Simple NN - with batch normalization (high level)
    kp = 0.9

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
 
    softmaxT = tf.nn.softmax(out, )
    prediction=tf.reduce_max(y,1)
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return out, accuracy, softmaxT
def build_network1( ):                  # Simple NN - 2layers - matmul 
    biases  = { 'b1': tf.Variable(tf.random_normal( [ h[0] ]),        name="Bias_1"),
                'b2': tf.Variable(tf.random_normal( [ h[1] ]),        name="Bias_2"),
                'out': tf.Variable(tf.random_normal( [nout] ),        name="Bias_out") }
    weights = { 'h1': tf.Variable(tf.random_normal([ninp,h[0]]),      name="Weights_1"),
                'h2': tf.Variable(tf.random_normal([h[0],h[1]]),      name="Weights_2"),
                'out': tf.Variable(tf.random_normal([h[1], nout]),    name="Weights_out")}

    # tf.reset_default_graph( )
    with tf.name_scope("fc_1"):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    with tf.name_scope("fc_2"):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    with tf.name_scope("fc_output"):
        out = tf.matmul(layer_2, weights['out']) + biases['out']

    softmaxT = tf.nn.softmax(out, )
    prediction=tf.reduce_max(y,1)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)


    return out, accuracy, softmaxT, biases, weights
# prediction, accuracy, softmaxT, biases, weights = build_network1()
prediction, accuracy, softmaxT = build_network2()

def restore_model(sess):   
    saver= tf.train.Saver() 
    print("Model restored from file: %s" % model_path)
    saver.restore(sess, model_path)
print("___Network created")

def get_data_test( desc ): 
    if desc == "FRFLO": 
        json_str = '''[
            { "m":"1", "c1122" :1 },
            { "m":"2", "c884" : 1 },
            { "m":"3", "c825" : 1 },
            { "m":"4", "c1122" :0.5 , "c825" :0.5 },
            { "m":"10", "c3" :0.5 , "c4" :0.5 }] '''
        tmpLab = [121, 110, 75, 90, 80]
    elif desc == "FRALL": #most used 
        json_str =  '''[
            { "m":"1", "c1122" : 1 },
            { "m":"2", "c884"  : 1 },
            { "m":"3", "c825"  : 1 },
            { "m":"4", "c1122" : 0.5 , "c825" :0.5 },
            { "m": 5,   "c903" :1	}     ] '''
        tmpLab = [121,110, 75, 90, 80, 44]

    elif desc == "TESTS" : 
        json_str =  '''[
            { "m":"8989", "c1" :0.5 },
            { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
        tmpLab = [59,99]
    return json_str, tmpLab

# OPERATIONS-----------------------------------------------------
def train(it = 100, disp=50, batch_size = 128): 
    print("____TRAINING...")
    display_step =  disp 
    total_batch  = int(len(md.dataT['label']) / batch_size)
    
    with tf.name_scope("xent"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        tf.summary.scalar("xent", cost)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    
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
        
        tr_ac = str(sess.run(accuracy, feed_dict={x: md.dataT['data'], y: md.dataT['label']}))[:5] 
        print("T Ac:", tr_ac)
        
        print("Optimization Finished!")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path) 

        logr( it=it, typ='TR', DS=md.DESC, AC=tr_ac,num=len(md.dataT["label"]), AC3=0, AC10=0, desc=md.des() )
        logr( it=it, typ='EV', DS=md.DESC, AC=ev_ac,num=len(md.dataE["label"]), AC3=0, AC10=0, desc=md.des() )
def evaluate( ): 
    print("_____EVALUATION...")
    
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
    gt3, gtM = md.check_perf_CN(predv, md.dataE, False)
    logr(  it=0, typ='EV', AC=ev_ac,DS=md.DESC, num=len(md.dataE["label"]), AC3=gt3, AC10=gtM, desc=md.des() )
def tests(url_test = 'url'):  
    print("_____TESTS...")    
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    
    if url_test != 'url':  
        json_data = url_test + "data_json.txt"
        tmpLab = pd.read_csv(url_test + "datal.csv", sep=',', usecols=[0,1])    
        tmpLab = tmpLab.loc[:,'fp']
        abstcc = False
    else: 
        json_str, tmpLab = get_data_test("FRALL")
        json_data = json.loads(json_str)
        abstcc = True
    dataTest['data']  = md.feed_data(json_data, p_abs=abstcc , d_st=True)
    [dataTest['label'].append( md.cc(x) ) for x in tmpLab ]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_model(sess)
        # predv = sess.run( prediction, feed_dict={x: dataTest['data']}) 
        ts_acn = '0'
        ts_acn, predv = sess.run( [accuracy, prediction], feed_dict={x: dataTest['data'], y: dataTest['label']}) 
        ts_ac = str(ts_acn) 
        print("test ac = {}".format(ts_ac))
    # print(dataTest['label'])
    range_ts = len(predv) if len(predv)<20 else 20
    for i in range( range_ts ):
        print("RealVal: {}  - PP value: {}".format( md.dc( dataTest['label'][i]), md.dc( predv.tolist()[i], np.max(predv[i]))  ))  
    gt3, gtM = md.check_perf_CN(predv, dataTest, False)
    
    logr( it=0, typ='TS', DS='matnrList...', AC=ts_acn ,num=len(dataTest["label"]),  AC3=gt3, AC10=gtM, desc=md.des() )  
    
 
def mainRun(): 
    # train(epochs, disp, batch_size)
    # evaluate( )
    url_test = "../../_zfp/data/FREXP/" ; md.DESC     = "FREXP"
    tests(  )
    print("___The end!")

if __name__ == '__main__':
    mainRun()




