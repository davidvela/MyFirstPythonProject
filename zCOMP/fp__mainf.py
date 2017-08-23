import pandas as pd 
import tensorflow as tf 
import numpy as np 
import requests
import json
import sys
import os

from fp_drc     import fpDataModel
from fp_modelf  import *

dv = 0 # tests 
if len(sys.argv) > 1:
    dv = int(sys.argv[1])

LOGDIR      = "./my_graph/0F2CR2/"
ALL_DS      = "../_zfp/data/TFFRFLO_ALSN.csv"
COL_DS     = "../_zfp/data/TFFRFLO_COL.csv"

MODELP      = LOGDIR + "model.ckpt"
json_path   = "../_zfp/data/json_fflo_ex.txt"
JLA_DS      = "../_zfp/data/TFFRAL_LAB.csv"

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
n_classes   = 100 
# read data - data.txt, columns.txt, labels.txt
# create pandas with columns and create data - pandas little by little. 
    #indeces idea? => my own matmul 
# init network - data prop -> len(columns); labels -> depend on model! 

dataClass   = fpDataModel( path= ALL_DS, norm = '', batch_size = p["bs"], dType="classN", labelCol = 'FP_C', dataCol = 4,   nC=n_classes, nRange=1, toList = True )
n_input     = dataClass.set_columns(COL_DS)
x           =  nn_input(n_input)
y           =  nn_output(n_classes) 
keep_prob   = nn_prob()


# n_hidden_1  = 256   # 1st layer number of features
# n_hidden_2  = 256   # 2nd layer number of features
# weights = {
#     'h1': tf.Variable(tf.random_normal([ x.get_shape().as_list()[1], n_hidden_1]),    name="Weights_1"),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1,  n_hidden_2]), name="Weights_2"),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="Weights_out"),
# }

# biases = {
#     'b1': tf.Variable(tf.zeros([n_hidden_1]), name="Bias_1"),
#     'b2': tf.Variable(tf.zeros([n_hidden_2]), name="Bias_2"),
#     'out': tf.Variable(tf.zeros([n_classes]), name="Bias_out"),
# }
pred        = nn(x, keep_prob, n_classes)

softmaxT    = tf.nn.softmax(pred)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax( pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("xent"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar("xent", cost)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=p["lr"]).minimize(cost)
 
init = tf.global_variables_initializer()
summ = tf.summary.merge_all()
saver = tf.train.Saver()

def train_model(hparam): 
    print("Training")
    dataTrain,  dataTest =  dataClass.get_data( ) 
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(LOGDIR + hparam)
        writer.add_graph(sess.graph)

        for i in range(tra_it): 
            xtb, ytb = dataClass.next_batch(p["bs"], dataTrain['data'], dataTrain['label']) 
            sess.run(optimizer, feed_dict={x: xtb, y: ytb})

            if i % p["rec_i"] == 0:
                [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xtb, y: ytb }) 
                writer.add_summary(s, i)
            if i % p["dis_i"] == 0:
                print("step %d, training accracy %g" %(i, train_accuracy))
        print("Optimization Finished!")

        save_path = saver.save(sess, MODELP)
        print("Model saved in file: %s" % save_path)
        
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: dataTest['data'], y: dataTest['label']}))

def evaluate_model( ): 
    dataTrain,  dataTest =  dataClass.get_data( ) 
    print("Evaluation")
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, MODELP)
        print("Model restored from file: %s" % MODELP)
        print("Testing  Accuracy:", sess.run(accuracy, feed_dict={x: dataTest['data'],  y: dataTest['label']}))
        print("Training Accuracy:", sess.run(accuracy, feed_dict={x: dataTrain['data'], y: dataTrain['label']}))
        predv, softv = sess.run([pred, softmaxT], feed_dict={x: dataTest['data']}) 
        # print("Real value: {}", dataClass.deClassifN( ytp1[i])  )
        for i in range(20):
            print("RealVal: {}  - PP value: {}".format( dataClass.deClassifN( dataTest['label'][i]), dataClass.deClassifN( predv.tolist()[i], np.max(predv[i]))  ))
            # maxa = sess.run([prediction], feed_dict={y: predv })
        pred_val = [];      data_val = []
        # return
        for i in range(len(predv)):
        #for i in range(100):
            if (i % 100==0): print(i)
            pred_vali = dataClass.deClassifN( predv.tolist()[i], np.max(predv[i]))
            data_vali = dataClass.deClassifN( dataTest['label'][i])
            # print("realVal: {} -- PP value: {}".format(data_vali,pred_vali))
            pred_val.append(pred_vali)
            data_val.append(data_vali)
        l3, l15 = dataClass.check_perf(pred_val, data_val)  
        print("Total: {} GT3: {}  GTM: {}".format(len(pred_val), l3, l15))    
def test_model( ): 
    dataTest = {'label' : [] , 'data' :  [] }
    pred_val = []
    # dataTest['data'] = dataClass.feed_data(json_path) 
    json_str="""[{"m":"000","100109":1}  ]"""
    json_data = json.loads(json_str)
    dataTest['data'] = dataClass.feed_data(json_data, d_st =True) 
    print(len(dataTest['data'][0]))
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, MODELP)
        print("Model restored from file: %s" % MODELP)
        predv = sess.run(pred, feed_dict={x: dataTest['data']})
    for i in range(len(predv)):
        # if (i % 10==0): print(i)
        pred_vali = dataClass.deClassifN( predv.tolist()[i], np.max(predv[i]))
        # data_vali = dataClass.deClassifN( dataTest['label'][i])
        print("{} realVal: {}".format(i,pred_vali))
        pred_val.append(pred_vali)
        # data_val.append(data_vali)
    # np.savetxt(LOGDIR + 'test_FF0_R.csv', pred_val, delimiter=',')   # X is an array
def print_stats(pb,rv):
    pass
        
def main(dv):
    hparam = make_hparam_string(p["lr"], 3)
    if   dv == 1:       train_model(hparam)
    elif dv == 0:       evaluate_model()   
    elif dv == 2:       test_model()   
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

def make_hparam_string(lr, no_fc):
    return "lr_%.0E,fc=%d" % (lr, no_fc) 

if __name__ == "__main__":
    main(dv)                                                                                                                                                                                                               