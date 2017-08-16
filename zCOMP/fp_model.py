# DATA HANDLING... 
import pandas as pd 
import tensorflow as tf 
import numpy as np 

import requests
import json

class fpModel:
   
    def __init__(self, hidden_nodes = 256, learning_rate = 0.1):
        np.random.seed(1)
        #self.pre_process_data(reviews, labels)
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)


    def check_perf(self, lA, lB):
    pass

