# tensorboard --logdir=.\_zfp\data\my_graph
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
import mData as md

# READ DATA -------------------------------------------------
print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
# md.spn = 200
ninp, nout  = md.mainRead()
# md.DESC     = "FREXP"
#ninp, nout  = md.mainRead2(md.ALL_DS, 1, 2 ) # For testing I am forced to used JSON - column names and order may be different! 
print("___Data Read!")

top_k = 2 
model_path = md.MODEL_DIR + "model.ckpt" 
lr         = 0.0001 #0.0001
# h      = [100 , 40]
# h      = [40 , 10]
h        = [200, 100, 40]
epochs     = 200
disp       = 5
batch_size = 128
def get_hpar(): return "lr_%.0E_NN%s" % (lr, get_nns())
def get_nns():  #return str(ninp)+'*'+str(h[0])+'*'+str(h[1])+'*'+str(nout)
    nns =  str(ninp)+'*' 
    for i in range(len(h)):
        nns = nns +str(h[i])+'*'
    return nns +str(nout)
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

# NETWORK-----------------------------------------------------
print( get_nns() )
x = tf.placeholder(tf.float32,   shape=[None, ninp], name="x")
y = tf.placeholder(tf.int16,     shape=[None, nout], name="y")
def fc(inp, nodes, kp, is_train):
    # h = tf.layers.dense( x, h[0], activation=tf.nn.relu,  name )
    h = tf.layers.dense( inp, nodes, use_bias=False, activation=None )
    h = tf.layers.batch_normalization(h, training=is_train)
    h = tf.nn.relu(h)
    h = tf.nn.dropout(h, kp)
    return h
def build_network2(is_train=False):     # Simple NN - with batch normalization (high level)
    kp = 0.5
    inp = x
    # h0 = fc(x,  h[0], kp, is_train)
    # h1 = fc(h0, h[1], kp, is_train)    
    for i in range(len(h)): 
        hx = fc(inp,  h[i], kp, is_train); inp = hx 
    out = tf.layers.dense( hx, nout, use_bias=False, activation=None )
    prediction=tf.reduce_max(y,1)
    
    # softmaxT = tf.nn.softmax(out)
    with tf.name_scope("accuracy"):
        softmaxT = tf.nn.top_k(tf.nn.softmax(out), top_k)         
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        tf.summary.scalar("accuracy", accuracy)

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
with tf.name_scope("xent"): #loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    tf.summary.scalar("xent", cost)
with tf.name_scope("train"): #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
summ = tf.summary.merge_all()
saver= tf.train.Saver()

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
        tmpLab = [121,110, 75, 90, 44]

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

    dataTest = {'label' : [] , 'data' :  [] };
    dataTest['data'], dataTest['label']  = md.feed_data("", p_abs=False , d_st=True, p_col=True)   
    # md.dataT['data'].append(dataTest['data']) ;     md.dataT['label'].append(dataTest['label']) 
    
    print("data read - lenTrain={}-{} & lenEv={}-{}" .format(len(md.dataT["data"]), len(md.dataT["label"]),len(md.dataE["data"]),len(md.dataE["label"]) ))
    total_batch  = int(len(md.dataT['label']) / batch_size)   
    startTime = datetime.now().strftime('%H:%M:%S')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore_model(sess)  #Run if I want to retrain an existing model  
        writer = tf.summary.FileWriter(md.MODEL_DIR + "tboard/", sess.graph ) # + get_hpar() )

        start = time.time()
        for i in range(it):            
            for ii, (xtb,ytb) in enumerate(md.get_batches(batch_size) ):
                # xtb, ytb = dc.next_batch(batch_size, dataT['data'], dataT['label'])
                sess.run(optimizer, feed_dict={x: xtb, y: ytb})
                if ii % display_step ==0: #record_step == 0:
                    #[train_accuracy] = sess.run([accuracy], feed_dict={x: xtb, y: ytb })
                    # s = sess.run(summ, feed_dict={x: xtb, y: ytb })
                    [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xtb, y: ytb }) 
                    writer.add_summary(s, i)
                     
                    elapsed_time = float(time.time() - start)
                    reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                    rp_s = str(reviews_per_second)[0:5]
                    tr_ac = str(train_accuracy)[:5]  
                    print('Epoch: {} batch: {} / {} - %Speed(it/disp_step): {} - tr_ac {}' .format(i, ii, total_batch, rp_s, tr_ac ))
                    # writer.add_summary(s, i)
            ev_ac = str(sess.run(accuracy, feed_dict={x: md.dataE['data'], y: md.dataE['label']}))[:5] 
            print("E Ac:", ev_ac)
            
            sess.run([optimizer], feed_dict={x: dataTest['data'], y: dataTest['label']})
            tr_ac = str(sess.run(accuracy, feed_dict={x: dataTest['data'], y: dataTest['label']}))[:5] 
            print("Cm Ac:", tr_ac)
        
        tr_ac = str(sess.run(accuracy, feed_dict={x: md.dataT['data'], y: md.dataT['label']}))[:5] 
        print("T Ac:", tr_ac)
        
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path) 
    print("Optimization Finished!")

    logr( it=it, typ='TR', DS=md.DESC, AC=tr_ac,num=len(md.dataT["label"]), AC3=0, AC10=0, desc=md.des(), startTime=startTime )
    logr( it=it, typ='EV', DS=md.DESC, AC=ev_ac,num=len(md.dataE["label"]), AC3=0, AC10=0, desc=md.des() )
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
def tests(url_test = 'url', p_col=False):  
    print("_____TESTS...")    
    
    # Load test data 
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    if p_col: dataTest['data'], dataTest['label']  = md.feed_data("", p_abs=False , d_st=True, p_col=True)   
    else: 
        if url_test != 'url':  
            md.DESC     = "FREXP1_X"
            json_data = url_test + "data_jsonX.txt"
            tmpLab = pd.read_csv(url_test + "datalX.csv", sep=',', usecols=[0,1])    
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
    gt3, gtM = md.check_perf_CN(sf, dataTest, False)
    logr( it=0, typ='TS', DS=md.DESC, AC=ts_acn ,num=len(dataTest["label"]),  AC3=gt3, AC10=gtM, desc=md.des() )  

    outfile = '../../_zfp/data/export2' 
    np.savetxt(outfile + '.csv', sf[1], delimiter=',')
    np.savetxt(outfile + 'PRO.csv', sf[0], delimiter=',')
 
def mainRun(): 
    # print(get_hpar() ); return 
    # epochs     = 10
    train(epochs, disp, batch_size)
    evaluate( )
    url_test = "../../_zfp/data/FREXP1/" ;
    tests(url_test, p_col=False  )
    print("___The end!")

if __name__ == '__main__':
    mainRun()




