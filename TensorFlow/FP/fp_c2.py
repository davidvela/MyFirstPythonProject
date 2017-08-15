# Classification Model [4885 rows x 1221 columns] - reuse the model! 
# and use the data reader class! - normalization and different categories / Type S
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fp_drc     import fpDataModel
import sys
import os
import urllib.request as urllib7m   
import pandas as pd
import numpy as np
import tensorflow as tf

dv = 0 # tests 
if len(sys.argv) > 1:
    dv = int(sys.argv[1])

# Datasets  
xtp1        = []  
ytp1        = []

LOGDIR      = "./my_graph/0FCR2/"
ALL_DS      = "../../knime-workspace/Data/FP/TFFRFL_ALSN.csv"
model_path  = LOGDIR + "model.ckpt"

# Parameters
learning_rate = 0.001
batch_size = 128
training_iters = 10000 #200000
display_step = training_iters*0.01 #10%
record_step  = training_iters*0.005 
# Network Parameters
n_hidden_1  = 256   # 1st layer number of features
n_hidden_2  = 256   # 2nd layer number of features
# n_input     = 969   # data input: FRFL
n_input     = 969   # data input: FRAL
n_classes   = 100     # total classes 

dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = 128, dType="classN", labelCol = 'FP_C', dataCol = 4,   nC=n_classes, nRange=3, toList = True )
dataTrain,  dataTest =  dataClass.get_data( ) 


# Model variables
x = tf.placeholder(tf.float32,   shape=[None, n_input],   name="x")
y = tf.placeholder(tf.int16,     shape=[None, n_classes], name="cat")
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),    name="Weights_1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="Weights_2"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="Weights_out"),
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="Bias_1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="Bias_2"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="Bias_out"),
}

# String for the logs
def make_hparam_string(learning_rate, no_fc):
    return "lr_%.0E,fc=%d" % (learning_rate, no_fc) 

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    with tf.name_scope("fc_1"):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    with tf.name_scope("fc_2"):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    with tf.name_scope("fc_output"):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# - declaration of model and global attributes 
pred = multilayer_perceptron(x, weights, biases)
softmaxT = tf.nn.softmax(pred)
prediction=tf.reduce_max(y,1)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
with tf.name_scope("xent"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar("xent", cost)
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
summ = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#
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

#
def test_model():
    # Running a new session
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
        # test the model
        print("Testing  Accuracy:", sess.run(accuracy, feed_dict={x: dataTest['data'],  y: dataTest['label']}))
        print("Training Accuracy:", sess.run(accuracy, feed_dict={x: dataTrain['data'], y: dataTrain['label']}))

        # xtp1.append(dataTest['data'][i]);    ytp1.append(dataTest['label'][i])
        predv, softv = sess.run([pred, softmaxT], feed_dict={x: dataTest['data']}) 
        # print("Real value: {}", dataClass.deClassifN( ytp1[i])  )
        for i in range(20):
            print("RealVal: {}  - PP value: {}".format( dataClass.deClassifN( dataTest['label'][i]), dataClass.deClassifN( predv.tolist()[i], np.max(predv[i]))  ))
            # maxa = sess.run([prediction], feed_dict={y: predv })

        pred_val = []
        data_val = []
        for i in range(len(predv)):
        #for i in range(100):
            if (i % 10==0): print(i)
            pred_vali = dataClass.deClassifN( predv.tolist()[i], np.max(predv[i]))
            data_vali = dataClass.deClassifN( dataTest['label'][i])
           
            pred_val.append(pred_vali)
            data_val.append(data_vali)

        l3, l15 = dataClass.check_perf(pred_val, data_val)  
        print("Total: {} GT3: {}  GTM: {}".format(len(pred_val), l3, l15))    
#
#
def main(dv):
    # Construct model
    hparam = make_hparam_string(learning_rate, 3)

    if dv == 0:         train_model(hparam)
    else :              test_model()
        
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == "__main__":
    main(dv)    

#end 