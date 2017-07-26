# Classification Model [4885 rows x 1221 columns]
# with a loop of elements -> tensorboard; compare learning rates and network models

# tensorboard --logdir=.\my_graph\0F\
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request as urllib7m   

import pandas as pd
import numpy as np
import tensorflow as tf

#Directories
LOGDIR      = "./my_graph/0F/"
TRAI_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNT.csv"
TEST_DS     = "../../knime-workspace/Data/FP/TFFRFL_ALSNE.csv"

# Datasets 
xt          = [] 
yt          = []
xtt         = [] 
ytt         = []
xtp1        = []  
ytp1        = []
batch_size = 128

# Model variables
x = tf.placeholder(tf.float32,   shape=[None, 1221], name="x")
y = tf.placeholder(tf.int16,     shape=[None, 4], name="cat")

# Run Parameters 
dv = 2   

training_iters = 1000 #200000

display_step = training_iters*0.1 #10%
record_step  = training_iters*0.005

#----------------------------------------------------
# DATA HANDLING... 
#----------------------------------------------------
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def classif(df):
    if( df < 40 ): return [0,0,0,1] 
    elif( df >= 40 and df < 60 ): return [0,0,1,0]
    elif( df >= 60 and df < 90 ): return [0,1,0,0] 
    elif( df >= 90 ): return [1,0,0,0]

def get_data(path):
    dst =  pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python")
    dst = dst.fillna(0)

    dst.insert(2, 'FP_C', dst['FP'].map(lambda x: classif(x)))
    cat  = dst.loc[:,'FP_C']
    dat  = dst.iloc[:, 3:]
    # catL = list(cat.values.flatten())
    catL = cat.as_matrix().tolist()
    datL = dat.as_matrix().tolist()
        
    return datL, catL

#----------------------------------------------------
#   RUN 
#----------------------------------------------------
def run_simple_model(learning_rate, use_two_fc, use_two_conv, hparam):
    xt, yt = get_data()

    y = tf.constant(yt)
    x = tf.constant(xt)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1221)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    m = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=4)   
    print("Fit model")
    m.fit(input_fn=lambda:(x,y), steps=learning_rate) 
    return
    print("Evaluate Accuray")
    #accuracy_score = m.evaluate(input_fn=lambda: input_fn(test), steps=1)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    
def run_simple_model2(learning_rate, use_two_fc, use_two_conv, hparam):
    xt, yt = get_data(TRAI_DS)
    

    x = tf.placeholder(tf.float32,   shape=[None, 1221],    name="x")
    y_ = tf.placeholder(tf.float32,  shape=[None, 4],       name="cat")
    # W = tf.Variable(tf.zeros([1221, 4]), name="Weights" )
    # b = tf.Variable(tf.zeros([4]), name="Bias")

    W = tf.Variable(tf.zeros([1221, 4]), name="Weights" )
    b = tf.Variable(tf.zeros([4]), name="Bias")

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = tf.square(Y - Y_predicted, name='loss')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy) 
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    #print("start training")
    # for _ in range(steps): mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: xt, y_: yt})
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: xt,  y_: yt }))
    return
# Architecture: Convolutional -> pooling -> convolutional -> pooling -> Fully Connected -> Fully Connected
# String for the logs
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    #conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s" % (learning_rate, fc_param) 
# Convolutional Layer
def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Fuyll Connected 
def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        # act = tf.nn.relu(act)
        # act = tf.nn.dropout(act, dropout) # Apply Dropout 
        return act  

# Run Productive model 
def run_pmodel():
    sess = tf.Session()
    saver = tf.train.import_meta_graph(LOGDIR + 'model.ckpt-4500.meta')
    rest  = saver.restore(sess, LOGDIR + 'model.ckpt-4500')
    print("print variables!", rest)

    all_vars = tf.get_collection( saver )
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)
    return
    print("Real value: %d", ytp1  )
    print("Predicted value:", sess.run(logits, feed_dict={x: xtp1}) ) 
    sess.close()
# Run  model 
def run_model(learning_rate, use_two_fc, use_two_conv, hparam, save_model):
    tf.reset_default_graph()
    sess = tf.Session()

    if use_two_fc: 
        fc1 = fc_layer(x, 1221, 500,    "fc1")
        relu = tf.nn.relu(fc1)
        logits = fc_layer(fc1, 500, 4,  "fc2")
    else:
        logits = fc_layer(x, 1221, 4,   "fc")

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=y), 
            name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # merge summary
    summ = tf.summary.merge_all()

    # Execution!
    if dv == 2: 
        saver = tf.train.import_meta_graph(LOGDIR + 'model.ckpt-4500.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        print("Printing Predictions:", \
             sess.run(xent, feed_dict={x: xtp1}))
        print("Real value: %d", ytp1[0]  )
        return
    else: # training! 
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOGDIR + hparam)
        writer.add_graph(sess.graph)
        for i in range(training_iters):  
            xtb, ytb = next_batch(batch_size, xt, yt)
            if i % record_step == 0:
                #[train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xt, y: yt }) 
                [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xtb, y: ytb }) 
                writer.add_summary(s, i)
            if i % display_step == 0:
                print("step %d, training accracy %g" %(i, train_accuracy))
                if save_model == True:
                    saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
            sess.run(train_step, feed_dict={x: xtb, y: ytb})
        print("Optimization Finished!")

        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: xtt, y: ytt}))
        
        # predict = tf.argmax(y ,1)
        print("Real Value:", ytp1 )
        print("Predicted value:", sess.run(logits, feed_dict={x: xtp1}) ) # sess.run(predict, feed_dict={y: yp }))
    sess.close()

# MAIN!
def main():
    save_model = True           
    learning_rate = 1E-3
    use_two_fc = True
    use_two_conv = True
    hparam  = ""

    if dv == 1:
        hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
        print('Starting run for %s' % hparam)
        run_model(learning_rate, use_two_fc, use_two_conv, hparam, save_model)
    elif dv == 0:
        for learning_rate in [0.05, 1E-3]: # , 1E-4]:
            for use_two_fc in [False, True]:
                for use_two_conv in [False]: #, True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
                    hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)        
                    print('Starting run for %s' % hparam)
                    run_model(learning_rate, use_two_fc, use_two_conv, hparam, False)
                    # run_simple_model2(learning_rate, False, False, hparam)

        print('Done training!')
    elif dv== 2: 
        print("Real Test!")
        hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
        #run_model(learning_rate, use_two_fc, use_two_conv, hparam, save_model)
        run_pmodel()
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)


if __name__ == "__main__":
    xtt, ytt    = get_data(TEST_DS)
    xt, yt      = get_data(TRAI_DS)
    xtp1.append(xtt[0]);    ytp1.append(ytt[0])
    main()
    # for i in range(2):  
    #     xtb, ytb = next_batch(10, xt, yt)
    #     print(ytb)
    #     print("--new--")
    # return
    xtp={
        'M' : 123456,
        'FP'       : 24,
        '111222'   : 0.1, 
        '113322'   : 0.9    }