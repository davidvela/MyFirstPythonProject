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


#version 2 - network + model 
class fpNN:
    def __init__(self, n_input, layers=2, hidden_nodes = [256 , 256], 
                 lr = 0.1, min_count = 10, polarity_cutoff = 0.1, output = 100   ):
        self.init_network(n_input, layers, output, lr, hidden_nodes)
    
    def init_network(self, n_input, layers, n_classes, learning_rate,  hidden_nodes):
        self.learning_rate = learning_rate # 0.001
        self.n_input     = n_input #1814 #1221
        self.n_classes   = n_classes 
        self.n_h1 = hidden_nodes[0]
        self.n_h2 = hidden_nodes[1]
        
        # cust - network 
        self.x = tf.placeholder(tf.float32,   shape=[None, n_input],   name="x")
        self.y = tf.placeholder(tf.int16,     shape=[None, n_classes], name="cat")
        self.biases = {
            'b1': tf.Variable(tf.random_normal( [ self.n_h1 ]), name="Bias_1"),
            'b2': tf.Variable(tf.random_normal( [ self.n_h2 ]), name="Bias_2"),
            'out': tf.Variable(tf.random_normal( [n_classes] ), name="Bias_out"),
        }
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, self.n_h1 ]),    name="Weights_1"),
            'h2': tf.Variable(tf.random_normal([ self.n_h1, self.n_h2 ]), name="Weights_2"),
            'out': tf.Variable(tf.random_normal([self.n_h2, n_classes]),  name="Weights_out"),
        }
        
        #hidden_nodes = [1221,256,256,100]
        #self.weights = {} 
        #self.biases = {}
        #for (i in range(layers)):
        #self.weights["h"+string(i)] = tf.Variable(tf.random_normal([n_input, hidden_nodes[0] ]),name="Weights_1"
                
        # Hidden layer with RELU activation
        with tf.name_scope("fc_1"):
            self.layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
            self.layer_1 = tf.nn.relu(self.layer_1)
        # Hidden layer with RELU activation
        with tf.name_scope("fc_2"):
            self.layer_2 = tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.biases['b2'])
            self.layer_2 = tf.nn.relu(self.layer_2)
        # Output layer with linear activation
        with tf.name_scope("fc_output"):
            self.pred = tf.matmul(self.layer_2, self.weights['out']) + self.biases['out']

        self.softmaxT = tf.nn.softmax(self.pred, )
        self.prediction=tf.reduce_max(self.y,1)

        with tf.name_scope("accuracy"):
            self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("xent"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
            tf.summary.scalar("xent", self.cost)

        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        self.summ = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
        self.saver= tf.train.Saver()
# print("network built")
class fpModel:
    def __init__(self, nn, dataClass, model_path ):
        self.nn = nn
        self.dc = dataClass
        self.model_path = model_path 
        self.l3 = 0; self.l15 = 0; 
        # self.rp_s = '';  self.tr_ac = ''; self.ev_ac = ''; self.ts_ac = ''; self.start=0; self.elapsed_time=0;
    def ini():
        # self.rp_s = '';  self.tr_ac = ''; self.ev_ac = ''; self.ts_ac = ''; self.start=0; self.elapsed_time=0;
        pass
    def logr(self, datep = '' , time='', it=1000, nn='', typ='TR', DS='', AC=0, num=0, AC3=0, AC10=0, desc=''):
        if desc == '': print("Log not recorded"); return 
        LOG = "../../_zfp/LOGT2.txt"
        f= open(LOG ,"a+") #w,a,
        
        if datep != '':
            dats = datep
        else: 
            dats = datetime.now().strftime('%d.%m.%Y') 

        if time != '':
            times = time
        else: times = datetime.now().strftime('%H:%M:%S') 

        line =  datetime.now().strftime('%d.%m.%Y') + '\t' + times
        line = line + '\t' + str(it) + '\t'+ self.get_nns() +  '\t' + str(self.nn.learning_rate,)
        line = line + '\t' + typ 
        line = line + '\t' + str(DS) + '\t' + str(AC) + '\t' + str(num) + '\t' + str(AC3) + '\t' +  str(AC10) + '\t' + desc + '\n'

        f.write(line)
        f.close()
        print("___Log recorded")
    def get_nns(self):
        ns = "n:" + str(self.nn.n_input) + "*"
        #for(i in range(self.layers) )                
        ns = ns + str(self.nn.n_h1) + "*" + str(self.nn.n_h2) + "*"
        ns = ns + str(self.nn.n_classes)
        return ns 
    def dummy(self):
        print("hello") 
    def restore_model(self, sess):
        print("Model restored from file: %s" % self.model_path)
        self.nn.saver.restore(sess, self.model_path)
    def check_perf(self, lA, lB):
        assert(len(lA) == len(lB))
        gt3  = 0
        gtM = 0
        num = 0
        if self.dc.dType == 'class': 
            for i in range(len(lA)):
                # print( "comp {} and {}" .format(lA[i], lB[i]))
                if lA[i] != lB[i]: gt3+=1; gtM+=1
        else:
            for i in range(len(lA)):
                num = abs(lA[i]-lB[i])
                if num > 3: gt3+=1
                if num > 10: gtM+=1
        return gt3, gtM    
    def check_perf_CN(self, predv, dataEv, sk_ev=False ):
        pred_val = []; data_val = []; self.l3 = 0; self.l15 = 0; 
        predvList = predv.tolist()
        print("denormalization all Evaluation : {} = {}" .format(len(predv), len(dataEv["label"])))
        #for i in range(100):
        for i in range(len(predv)):
            if (i % 500==0): print(str(i)) #print('i='+str(i), end="")
            pred_vali = 0; data_vali = 0;
            try:
                pred_vali = self.dc.deClassifN( predv.tolist()[i], np.max(predv[i]))
                if sk_ev == True:
                    data_vali = dataEv['label'][i]
                else: data_vali = self.dc.deClassifN( dataEv['label'][i])
                # print("realVal: {} -- PP value: {}".format(data_vali,pred_vali))
                pred_val.append(pred_vali)
                data_val.append(data_vali)
            except:
                print("error: i={}, pred={}, data={} -- ".format(i, pred_vali, data_vali))

        self.l3, self.l15 = self.check_perf(pred_val, data_val)  
        print("Total: {} GT3: {}  GTM: {}".format(len(pred_val), self.l3, self.l15)) 

    def train(self, dataTrain, dataEv, it = 10000, disp=0.1, desc=''):
        it = it # = 10000 #200000
        batch_size = 128
        display_step =  it*disp #10%
        record_step  =  it*(disp/2) 
        with tf.Session() as sess:
            sess.run(self.nn.init)
            start = time.time()
            for i in range(it):  
                xtb, ytb = self.dc.next_batch(batch_size, dataTrain['data'], dataTrain['label']) 
                elapsed_time = float(time.time() - start)
                reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                if i % record_step == 0:
                    [train_accuracy] = sess.run([self.nn.accuracy], feed_dict={self.nn.x: xtb, self.nn.y: ytb }) 
                    #writer.add_summary(s, i)
                if i % display_step == 0:
                    #print("step %d, training accracy %g " %(i, train_accuracy))
                    elapsed_time = float(time.time() - start)
                    reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                    rp_s = str(reviews_per_second)[0:5]
                    tr_ac = str(train_accuracy)[:5]  
                    print('step {} - %Speed(it/disp_step): {} - tr_ac {}' .format(i, rp_s, tr_ac ))
                    ev_ac = str(sess.run(self.nn.accuracy, feed_dict={self.nn.x: dataEv['data'],    self.nn.y: dataEv['label']}))[:5] 
                    print("Eval Accuracy:", ev_ac)
                sess.run(self.nn.optimizer, feed_dict={self.nn.x: xtb, self.nn.y: ytb})
            print("Optimization Finished!")
            save_path = self.nn.saver.save(sess, self.model_path)
            print("Model saved in file: %s" % save_path) 
            # ev_ac = str(sess.run(self.nn.accuracy, feed_dict={self.nn.x: dataEv['data'], self.nn.y: dataEv['label']}))[:5] 
            # print("Eval Accuracy:", ev_ac)
            # def logr(self, datep = '' , time='', it=1000, nn='', lr=0.01, typ='TR', DS='', AC=0, num=0, AC3=0, AC10=0, desc=''):
            self.logr( it=it, typ='TR', DS=self.dc.DSC, AC=tr_ac,num=len(dataTrain["label"]), AC3=0, AC10=0, desc=desc)
            self.logr( it=it, typ='EV', DS=self.dc.DSC, AC=ev_ac,num=len(dataEv["label"]),    AC3=0, AC10=0, desc=desc)
    def evaluate(self, dataTrain, dataEv,  desc='' ):
        print("EVALUATION...")
        with tf.Session() as sess:
            sess.run(self.nn.init)
            self.restore_model(sess)
            # test the model
            tr_ac = str(sess.run(self.nn.accuracy, feed_dict={self.nn.x: dataTrain['data'], self.nn.y: dataTrain['label']}) )[:5]  
            ev_ac = str(sess.run(self.nn.accuracy, feed_dict={self.nn.x: dataEv['data'],    self.nn.y: dataEv['label']}))[:5] 
            print("Training   Accuracy:", tr_ac )
            print("Evaluation Accuracy:", ev_ac )
            # xtp1.append(dataTest['data'][i]);    ytp1.append(dataTest['label'][i])
            predv, softv = sess.run([self.nn.pred, self.nn.softmaxT], feed_dict={self.nn.x: dataEv['data']}) 
            print("Preview the first predictions:")
            for i in range(20):
                print("RealVal: {}  - PP value: {}".format( self.dc.deClassifN( dataEv['label'][i]), 
                                                            self.dc.deClassifN( predv.tolist()[i], np.max(predv[i]))  ))
            # maxa = sess.run([prediction], feed_dict={y: predv })
        self.check_perf_CN(predv, dataEv , False)
        self.logr(  it=0, typ='EV', AC=ev_ac,DS=self.dc.DSC, num=len(dataEv["label"]), AC3=self.l3, AC10=self.l15, desc=desc)
    def test(self, col, p_abs, p_json_str=0, p_label=0, desc=''):
        print("TESTS...")    
        dataTest = {'label' : [] , 'data' :  [] }
        pred_val = []
        print("input-no={}".format(self.dc.set_columns(col) ))
        
        if p_json_str != 0: json_str = p_json_str
        else: 
            json_str = '''[{ "m":"8989", "c1" :0.5 },
            { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''

        if p_label != 0:  tmpLab= p_label
        else: tmpLab = [59,99]
        
        json_data = json.loads(json_str)
        dataTest['data']  = self.dc.feed_data(json_data, abstract = p_abs) 
        # dataTest['label'] = tmpLab.map( lambda x: self.dc.classify(x)) 
        for x in tmpLab:
            dataTest['label'].append(self.dc.classify(x))

        with tf.Session() as sess:
            sess.run(self.nn.init)
            self.restore_model(sess)
            predv = sess.run( self.nn.pred, feed_dict={self.nn.x: dataTest['data']}) 
            #ts_acn= sess.run( [self.nn.pred], feed_dict={self.nn.x: dataTest['data'], self.nn.y: dataTest['label']}) 
            #ts_ac = str(ts_acn)[:5]  
            #print("test ac = {}".format(ts_ac))
        # print(dataTest['label'])
        for i in range(len(predv)):
            print("RealVal: {}  - PP value: {}".format( self.dc.deClassifN( dataTest['label'][i] ), #self.dctmpLab[i], 
                                                        self.dc.deClassifN( predv.tolist()[i], np.max(predv[i]))  ))  
        self.check_perf_CN(predv, dataTest, False )
        self.logr( it=0, typ='TS', 
                         DS='matnrList...', AC='0',num=len(dataTest["label"]),  AC3=self.l3, AC10=self.l15, desc=desc)      
    def dummy3(self): 
        self.logr( it=1000, nn='200*200', typ='TR', DS='FRALL', AC=0.99, num=400, AC3=4, AC10=3, desc='test fclass')
    def train2(self, dataTrain, dataEv, it = 100, disp=50, desc='', batch_size = 128):
        display_step =  disp
        total_batch  = len(dataTrain['label']) / batch_size
        
        with tf.Session() as sess:
            sess.run(self.nn.init)
            start = time.time()
            for i in range(it):            
                for ii, (xtb,ytb) in enumerate(self.dc.get_batches(dataTrain['data'], dataTrain['label'],batch_size)):
                    # xtb, ytb = self.dc.next_batch(batch_size, dataTrain['data'], dataTrain['label']) 
                    sess.run(self.nn.optimizer, feed_dict={self.nn.x: xtb, self.nn.y: ytb})
                    if ii % display_step ==0: 
                        [train_accuracy] = sess.run([self.nn.accuracy], feed_dict={self.nn.x: xtb, self.nn.y: ytb }) 
                        elapsed_time = float(time.time() - start)
                        reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                        rp_s = str(reviews_per_second)[0:5]
                        tr_ac = str(train_accuracy)[:5]  
                        print('Epoch: {} batch: {} / {} - %Speed(it/disp_step): {} - tr_ac {}' .format(i, ii, total_batch, rp_s, tr_ac ))
                        #writer.add_summary(s, i)
                ev_ac = str(sess.run(self.nn.accuracy, feed_dict={self.nn.x: dataEv['data'],    self.nn.y: dataEv['label']}))[:5] 
                print("Eval Accuracy:", ev_ac)

            print("Optimization Finished!")
            save_path = self.nn.saver.save(sess, self.model_path)
            print("Model saved in file: %s" % save_path) 
            # ev_ac = str(sess.run(self.nn.accuracy, feed_dict={self.nn.x: dataEv['data'], self.nn.y: dataEv['label']}))[:5] 
            # print("Eval Accuracy:", ev_ac)
            # def logr(self, datep = '' , time='', it=1000, nn='', lr=0.01, typ='TR', DS='', AC=0, num=0, AC3=0, AC10=0, desc=''):
            self.logr( it=it, typ='TR', DS=self.dc.DSC, AC=tr_ac,num=len(dataTrain["label"]), AC3=0, AC10=0, desc=desc)
            self.logr( it=it, typ='EV', DS=self.dc.DSC, AC=ev_ac,num=len(dataEv["label"]),    AC3=0, AC10=0, desc=desc)
# print("model class build")

# test:  
# LOGDIR     = "../../_zfp/data/my_graph/"        
# nc = fpNN(ncol=1814, layers=2, hidden_nodes = [256 , 256],lr = 0.01, min_count = 10, polarity_cutoff = 0.1, output=4)
# print("network built")
# model_path  = LOGDIR + "0F2CV4/model.ckpt"      
# mlp =  fpModel(nc, model_path)
# mlp.logr(desc = 'testVC - class')

