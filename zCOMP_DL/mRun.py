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

# from mData import *

lr         = 0.01
batch_size = 128
epochs     = 100
network    = [40 , 10]

def build_network1():
    pass




def evaluate():
    pass
def tests():
    pass



def train():
    pass
    def train2(self, dataTrain, dataEv, it = 100, disp=50, desc='', batch_size = 128):
        display_step =  disp 
        total_batch  = len(dataTrain['label']) / batch_size
        
        with tf.Session() as sess:
            sess.run(self.nn.init)
            start = time.time()
            for i in range(it):            
                for ii, (xtb,ytb) in enumerate(get_batches(dataTrain['data'], dataTrain['label'],batch_size)):
                    # xtb, ytb = self.dc.next_batch(batch_size, dataTrain['data'], dataTrain['label']) 
                    sess.run(self.nn.optimizer, feed_dict={self.nn.x: xtb, self.nn.y: ytb})
                    if ii % display_step ==0: #record_step == 0:
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



def main(): 
    build_network1()
    train()
    evaluate()
    tests()

if __name__ == '__main__':
    print("hi")
    main1()