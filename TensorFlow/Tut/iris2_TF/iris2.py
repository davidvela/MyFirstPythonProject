#help(plt.hist)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf

model_types= ['wide', 'deep', 'wide_n_deep']
file_name = "irisData.csv"
COLUMNS = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class" ]
CONTINUOUS_COLUMNS = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width" ]
LABEL_COLUMN = "Class"

ClassElements = {'virginica':0,  'versicolor':1, 'setosa':2 }


def printTF(data):
    sess = tf.Session() #sess.close()
    with tf.Session() as sess:
        result = sess.run(data)
        print(result) 
#*********************************************************************
# input + model  
#********************************************************************* 
def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    '''categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}'''
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    #feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    #printTF(feature_cols)
    #printTF(label)
    return feature_cols, label

def build_estimator(model_dir, model_type):
#     # Specify that all features have real-value data
#     print("feature_columns")
#     feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]  
#     # Build 3 layer DNN with 10, 20, 10 units respectively.
#     print("classifier")
#     classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                               hidden_units=[10, 20, 10],
#                                               n_classes=3,
#                                               model_dir="./iris_model")
#                                            #   model_dir="/tmp/iris_model")

    # Continuous base columns. ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width" ]
    SepalL = tf.contrib.layers.real_valued_column("Sepal_Length")
    SepalW = tf.contrib.layers.real_valued_column("Sepal_Width")
    PetalL = tf.contrib.layers.real_valued_column("Petal_Length")
    PetalW = tf.contrib.layers.real_valued_column("Petal_Width")
    
    wide_columns = []
    deep_columns = [SepalL, SepalW, PetalL,PetalW]

    if model_type == "wide":
            m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                            feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier( model_dir=model_dir,
                                            feature_columns=deep_columns,
                                            # hidden_units=[100, 50])
                                            hidden_units=[10, 20, 10],
                                            n_classes=3        )
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)
    return m

 
#*********************************************************************
# paint 
#*********************************************************************  
def paint_plot(data,  x_label, y_label,clase,c,m,label):
    print("paint plot: " + label)
    colors = list()
    palette = {0: "red", 1: "green", 2: "blue"}
    x = data[ data['Class'] == clase ][x_label]
    y = data[ data['Class'] == clase ][y_label]
    plt.scatter(x, y ,color=c, edgecolors='k',s=50, alpha=0.9, marker=m,label=label)
    
def paint(df_iris):
    print(df_iris)
    paint_plot(df_iris, COLUMNS[0], COLUMNS[1],'virginica',  'g','o','virginica')
    paint_plot(df_iris, COLUMNS[0], COLUMNS[1],'versicolor', 'b','o','versicolor')
    paint_plot(df_iris, COLUMNS[0], COLUMNS[1],'setosa',     'r','o','setosa')
    plt.show()
    #df_iris.plot(kind='scatter',x='Sepal Length',y='Sepal Width')
def read_irisData():
    # Read my file
    if not os.path.exists(file_name): 
        print("Iris file not found")
        return
    #print( tf.gfile.Open(file_name))
    df_iris = pd.read_csv(
            tf.gfile.Open(file_name),
            #names=COLUMNS,
            skipinitialspace=True,
            engine="python")
            
    #paint(df_iris)
    train =df_iris.sample(frac=0.2,random_state=200)
    test  =df_iris.drop(train.index)
    # train[LABEL_COLUMN] = (  train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    train[LABEL_COLUMN] = ( train[LABEL_COLUMN].apply(lambda x: ClassElements[x] )   ).astype(int)
    test[LABEL_COLUMN] = ( test[LABEL_COLUMN].apply(lambda x: ClassElements[x] )   ).astype(int)
    return train, test

def read_irisDataT():
    train = pd.read_csv(
            tf.gfile.Open("irisDataTR.csv"),
            #names=COLUMNS,
            skipinitialspace=True,
            engine="python")
            
    test = pd.read_csv(
            tf.gfile.Open("irisDataTS.csv"),
            #names=COLUMNS,
            skipinitialspace=True,
            engine="python")
    return train, test 


#*********************************************************************
#Main function
#*********************************************************************
def main(_):
    print("Start main")

    train, test = read_irisData()

    model_dir  = "./iris_model"
    model_type = model_types[1]
    m = build_estimator(model_dir, model_type)

    # Fit model.
    print("Fit model")
    m.fit(input_fn=lambda: input_fn(train), steps=2000)
    # Evaluate accuracy.
    print("Evaluate Accuray")
    accuracy_score = m.evaluate(input_fn=lambda: input_fn(test), steps=1)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    print("END")


    # Classify two new flower samples.
    '''def new_samples():
        return np.array(
        [[6.4, 3.2, 4.5, 1.5],
        [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
   
    predictions = list(m.predict(input_fn=new_samples)) 
    print(
        "New Samples, Class Predictions:    {}\n"
        .format(predictions)) '''

#*********************************************************************
#*********************************************************************
#* MAIN 
#*********************************************************************
#*********************************************************************
if __name__ == "__main__":
  #main("")
  tf.app.run(main=main, argv=[sys.argv[0]])