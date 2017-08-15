# Regresion Reuse model - FRFL T and E - new datareader class 
# tensorboard --logdir=.\my_graph\0FR\
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
xtp1        = []  
ytp1        = []
LOGDIR      = "./my_graph/0FR2/"
model_path  = LOGDIR + "model.ckpt"
        # Parameters
batch_size = 128
ALL_DS      = "../../knime-workspace/Data/FP/TFFRFL_ALSN.csv"
#ALL_DS      = "../../knime-workspace/Data/FP/TFFRAL_ALSNN.csv"
dataClass = fpDataModel( path= ALL_DS, norm = '', batch_size = batch_size, dType="reg", labelCol = 'FP_R', dataCol = 4,   nC=100, nRange=1, toList = True )



learning_rate = 0.01
training_iters = 5000 #200000
display_step = training_iters*0.1 #10%
record_step  = training_iters*0.005
        # Network Parameters
n_hidden_1  = 256   # 1st layer number of features
n_hidden_2  = 256   # 2nd layer number of features
n_input     = 969   # data input FL
#n_input     = 1801  # data input AL
n_classes   = 1     # total classes

        # Model variables
x = tf.placeholder(tf.float32,   shape=[None, n_input],   name="x")
y = tf.placeholder(tf.float32,   shape=[None, n_classes], name="cat")

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
        #normalization - min_max => not working! probably because of the random values of Wegights 
# xt, yt      = get_data(TRAI_DS, 1)
# xtt, ytt    = get_data(TEST_DS, 1)
# n_samples   = xt.shape[0]
n_samples   = 4724
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

with tf.name_scope("R2"):
    total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, pred)))
    R_squared = tf.subtract(tf.to_float(1), tf.div(total_error, unexplained_error))
    tf.summary.scalar("R2", R_squared)
with tf.name_scope("xent"):
    cost = tf.reduce_mean(tf.square(pred-y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*batch_size)
    # cost = tf.reduce_mean(tf.pow(pred - y, 2)) / 2  
    tf.summary.scalar("xent", cost)
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
summ = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# accuracy

#
def train_model(hparam): 
    # Running first session
    dataTrain,  dataTest =  dataClass.get_data( ) 
    print("Starting 1st session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        writer = tf.summary.FileWriter(LOGDIR + hparam)
        writer.add_graph(sess.graph)
        for i in range(training_iters): 
            # for p in  range(100):
            xtb, ytb = dataClass.next_batch(batch_size, dataTrain['data'], dataTrain['label'])
            sess.run(optimizer, feed_dict={x: xtb, y: ytb})
          
            if i % record_step == 0:
                [training_ac, s] = sess.run([cost, summ], feed_dict={x: dataTrain['data'], y: dataTrain['label'] }) 
                writer.add_summary(s, i)
            if i % display_step == 0:
                print("step %d, training_cost: %g" %(i, training_ac))

        print("Optimization Finished!")
        print("R2 Training: %g", sess.run([R_squared], feed_dict={x: dataTrain['data'], y: dataTrain['label'] })  )
        print("R2 Test:     %g", sess.run([R_squared], feed_dict={x: dataTest['data'], y: dataTest['label'] })  )

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
   
#
def evaluate_model():
    # Running a new session
    dataTrain,  dataTest =  dataClass.get_data( ) 
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
        # test the model
        print("Training R_squared:", sess.run(R_squared, feed_dict={x: dataTrain['data'], y: dataTrain['label']}))
        print("Testing  R_squared:", sess.run(R_squared, feed_dict={x: dataTest['data'],  y: dataTest['label']}))\
        

        pred_val = sess.run(pred, feed_dict={x: dataTest['data'], y: dataTest['label']})
        # pred_dval = dataClass.denormalize(pred_val)
   
        for i in range(20):
            # print("RealVal: {}  - PP value: {}".format( dataClass.deClassifN( dataTest['label'][i]), dataClass.deClassifN( predv.tolist()[i], np.max(predv[i]))  ))
            print("RealVal: {}  - PP value: {}".format( dataTest['label'][i], pred_val.tolist()[i] ))

        l3, l15 = dataClass.check_perf(pred_val, dataTest['label'])  
        print("Total: {} GT3: {}  GTM: {}".format(len(pred_val), l3, l15)) 

        #print("Testing Accuracy: \n", pred_val)
        #np.savetxt(LOGDIR + 'test_FF0_R.csv', pred_val, delimiter=',')   # X is an array
def test_model():
    pass      
#
def main(dv):
    # Construct model
    # xtp1.append(xtt[2]);    ytp1.append(ytt[2])
    hparam = make_hparam_string(learning_rate, 3)
    if dv == 0:         train_model(hparam)
    else :              evaluate_model()


    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == "__main__":
    main(dv)
