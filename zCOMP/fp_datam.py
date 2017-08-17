import pandas as pd 
import tensorflow as tf 
import numpy as np 
import requests
import json

class fpDataModel:
    def __init__(self, batch_size=128, ):
        self.path           = path
        self.norm           = norm
        self.batch_size     = batch_size
        self.dType          = dType
        self.nC             = nC
        self.nRange         = nRange
        self.toList         = toList 
    
    # Ordered batch... 
    def next_obatch(self, num, data, labels):
        return True
    # Random batch... 
    def next_batch(self, num, data, labels):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    #
    def deClassifN(self, df, val = 1 ):
        return df.index(val)
    #Get Data
    def get_data(self, typeSep = True, pathA = "", filter = ""):
        #read 3 files: data, columns and (maybe) labels 
        pass
    #
    def check_perf(self, lA, lB):       
        assert(len(lA) == len(lB))
        less3  = 0
        less15 = 0
        num = 0
        for i in range(len(lA)):
            num = abs(lA[i]-lB[i])
            if num > 3: less3+=1
            if num > 15: less15+=1
        return less3, less15

            
    # WS - Conversion
    def feed_data(self, url):
       
#  main 
def main():
    dataClass = fpDataModel()

if __name__ == '__main__':
    main()


