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

from mData import *

print("___Start!___" +  str(time.time())  )
ninp, nout = mainRead()
epochs     = 10
disp       = 50
descr = des()
batch_size = 128
print("___Data Read!")

lr         = 0.01
h          = [40 , 10]

def get_nns(): return str(ninp)+'*'+str(h[0])+'*'+str(h[1])+'*'+str(nout)
def logr(datep = '' , time='', it=1000, nn='', typ='TR', DS='', AC=0, num=0, AC3=0, AC10=0, desc=''):
    if desc == '': print("Log not recorded"); return 
    LOG = "../../_zfp/LOGT2.txt"
    f= open(LOG ,"a+") #w,a,
    if datep != '':   dats = datep
    else:             dats = datetime.now().strftime('%d.%m.%Y') 
    if time != '':    times = time
    else:             times = datetime.now().strftime('%H:%M:%S') 

    line =  datetime.now().strftime('%d.%m.%Y') + '\t' + times
    line = line + '\t' + str(it) + '\t'+  get_nns() +  '\t' + str( learning_rate)
    line = line + '\t' + typ 
    line = line + '\t' + str(DS) + '\t' + str(AC) + '\t' + str(num) + '\t' + str(AC3) + '\t' +  str(AC10) + '\t' + desc + '\n'

    f.write(line);  f.close()
    print("___Log recorded")    

# cust - network 
print( get_nns() )
x = tf.placeholder(tf.float32,   shape=[None, ninp],              name="x")
y = tf.placeholder(tf.int16,     shape=[None, nout],              name="y")
biases  = { 'b1': tf.Variable(tf.random_normal( [ h[0] ]),        name="Bias_1"),
                'b2': tf.Variable(tf.random_normal( [ h[1] ]),    name="Bias_2"),
                'out': tf.Variable(tf.random_normal( [nout] ),    name="Bias_out") }
weights = { 'h1': tf.Variable(tf.random_normal([ninp,h[0]]),      name="Weights_1"),
            'h2': tf.Variable(tf.random_normal([h[0],h[1]]),      name="Weights_2"),
            'out': tf.Variable(tf.random_normal([h[1], nout]),    name="Weights_out")}

def build_network1( ):
    with tf.name_scope("fc_1"):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    with tf.name_scope("fc_2"):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    with tf.name_scope("fc_output"):
        pred = tf.matmul(layer_2, weights['out']) + biases['out']

    softmaxT = tf.nn.softmax(pred, )
    prediction=tf.reduce_max(y,1)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("xent"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("xent", cost)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    
    summ = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver= tf.train.Saver()
    return init, prediction, accuracy, cost, optimizer, saver, softmaxT
# initialize network!
init, prediction, accuracy, cost, optimizer, saver, softmaxT = build_network1()
print("___Network created")


def evaluate():
    print("EVALUATION...")
    # with tf.Session() as sess:
    #     sess.run( init)
    #     self.restore_model(sess)
    #     # test the model
    #     tr_ac = str(sess.run( accuracy, feed_dict={ x: dataTrain['data'],  y: dataTrain['label']}) )[:5]  
    #     ev_ac = str(sess.run( accuracy, feed_dict={ x: dataEv['data'],     y: dataEv['label']}))[:5] 
    #     print("Training   Accuracy:", tr_ac )
    #     print("Evaluation Accuracy:", ev_ac )
    #     # xtp1.append(dataTest['data'][i]);    ytp1.append(dataTest['label'][i])
    #     predv, softv = sess.run([prediction, softmaxT], feed_dict={x: dataEv['data']}) 
    #     print("Preview the first predictions:")
    #     for i in range(20):
    #         print("RealVal: {}  - PP value: {}".format( dc( dataEv['label'][i]), 
    #                                                     dc( predv.tolist()[i], np.max(predv[i]))  ))
    #     # maxa = sess.run([prediction], feed_dict={y: predv })
    # self.check_perf_CN(predv, dataEv , False)
    # self.logr(  it=0, typ='EV', AC=ev_ac,DS=self.dc.DSC, num=len(dataEv["label"]), AC3=self.l3, AC10=self.l15, desc=desc)
def tests():
    pass

def train(it = 100, disp=50, descr='', batch_size = 128):
    display_step =  disp 
    total_batch  = len(dataT['label']) / batch_size
    
    with tf.Session() as sess:
        sess.run(init)
        start = time.time()
        for i in range(it):            
            for ii, (xtb,ytb) in enumerate(get_batches(batch_size) ):
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
            ev_ac = str(sess.run(accuracy, feed_dict={x: dataE['data'], y: dataE['label']}))[:5] 
            print("Eval Accuracy:", ev_ac)
        
        print("Optimization Finished!")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path) 

        logr( it=it, typ='TR', DS=DESC, AC=tr_ac,num=len(dataT["label"]), AC3=0, AC10=0, desc=descr)
        logr( it=it, typ='EV', DS=DESC, AC=ev_ac,num=len(dataE["label"]), AC3=0, AC10=0, desc=descr)



def mainRun(): 
    train(epochs, disp, descr, batch_size)
    evaluate()
    tests()
    # mlp =  fpModel( MODEL_P, ni,  network , no)
    # print(mlp.nn.get_nns())
    print("___The end!")

if __name__ == '__main__':
    mainRun()