import pandas as pd 
import tensorflow as tf 
import numpy as np 
import requests
import json

class fpDataModel:
    def __init__(self, path='', norm= '', batch_size=128, dType='reg', nC=100, nRange=1 , toList = True):
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
        '''
        Return a total of `num` random samples and labels. 
        '''
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
        url_oData_people = "http://services.odata.org/TripPinRESTierService/(S(pk4yy1pao5a2nngmm2ecx0hy))/People"
        # response = requests.get( url_oData_people )
        # people   = response.json();   # print(people)
        # CONVERT JOSN into object -> Pandas or dictionary array.7
        movie_json = """
        {
        "Title":"Johnny 5",
        "Year":"2001",
        "Runtime":"119 min",
        "Country":"USA"
        }
        """
        movie_data = json.loads(movie_json) # <class 'dict'>
        print("The title is {}".format(movie_data.get('Title')))    
        # add new elements to the dataset

#  main 
def main():
    dataClass = fpDataModel( )
    test1 = [ 4, 5, 10, 20]
    test2 = [ 1, 8, 20, 50] 
    l3, l15 = dataClass.check_perf(test1, test2)
    print(l3, l15)
    
def read_json():    
    file_directory    = "../../knime-workspace/Data/FP2/JSON.txt"
    json_data=open(file_directory).read()
    data = json.loads(json_data)
    #print(json_data)
    print(data[1]['m'])

if __name__ == '__main__':
    main()
    read_json()


