import pandas as pd 
import tensorflow as tf 
import numpy as np 

import requests
import json
import sys
import os

dv = 0 # tests 
if len(sys.argv) > 1:
    dv = int(sys.argv[1])

LOGDIR      = "./my_graph/0FP2_1/"
ALL_DS      = "../../knime-workspace/Data/FP/TFFRFL_ALSN.csv"
MODELP      = LOGDIR + "model.ckpt"

# Datasets  
tra_it = 10000   #200000
# Parameters dict
p = {
    "lr" : 0.001,
    "bs" : 128,
   
    "dis_i" : tra_it*0.01,   #10%
    "rec_i" : tra_it*0.005, 
    # Network Parameters
    "n_h1"  : 256,   # 1st layer number of features
    "n_h2"  : 256,   # 2nd layer number of features
    "n_ou"  : 100,     # total classes 
}





# read data - data.txt, columns.txt, labels.txt
# create pandas with columns and create data - pandas little by little. 
    #indeces idea? => my own matmul 
# init network - data prop -> len(columns); labels -> depend on model! 

init = tf.global_variables_initializer()
summ = tf.summary.merge_all()
saver = tf.train.Saver()

def train_model(hparam): 
    # Running first session
    print("Starting 1st session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        writer = tf.summary.FileWriter(LOGDIR + hparam)
        writer.add_graph(sess.graph)

        for i in range(training_iters): 
            xtb, ytb = dataClass.next_batch(batch_size, dataTrain['data'], dataTrain['label']) 
            if i % record_step == 0:
                [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xtb, y: ytb }) 
                writer.add_summary(s, i)
            if i % display_step == 0:
                print("step %d, training accracy %g" %(i, train_accuracy))
            sess.run(optimizer, feed_dict={x: xtb, y: ytb})
        print("Optimization Finished!")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: dataTest['data'], y: dataTest['label']}))


def main(dv):
    hparam = make_hparam_string(p["lr"], 3)
    
    # if dv == 0:         train_model(hparam)
    # elif dv == 1:              evaluate_model()   
    # elif dv == 2:              test_model()   
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

def make_hparam_string(lr, no_fc):
    return "lr_%.0E,fc=%d" % (lr, no_fc) 

if __name__ == "__main__":
    main(dv)  
    print(p["lr"])  
                                                                                                                                                                                                             