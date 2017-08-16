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
xtp1        = []  
ytp1        = []

# Parameters dict
p = {
    "lr" : 0.001,
    "bs" : 128,
    "tra_i" : 10000,        #200000
    "dis_i" : 10000*0.01,   #10%
    "rec_i" : 10000*0.005, 
    # Network Parameters
    "n_h1"  : 256,   # 1st layer number of features
    "n_h2"  : 256,   # 2nd layer number of features
    "n_ou"  : 100,     # total classes 
}


# read data - data.txt, columns.txt, labels.txt
# create pandas with columns and create data - pandas little by little. 
    #indeces idea? => my own matmul 

# init network - data prop -> len(columns); labels -> depend on model! 


def main(dv):
    hparam = make_hparam_string(p["lr"], 3)
    # if dv == 0:         train_model(hparam)
    # else :              test_model()   
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

def make_hparam_string(lr, no_fc):
    return "lr_%.0E,fc=%d" % (lr, no_fc) 

if __name__ == "__main__":
    main(dv)  
    print(p["lr"])  
