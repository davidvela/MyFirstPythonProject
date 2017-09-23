import pandas as pd 
import tensorflow as tf 
import numpy as np 
import requests
import json
from collections import Counter
import csv

class fpDataModel:
    def __init__(self, batch_size=128, ):
        self.batch_size     = batch_size
    
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
    def classif(self, df):
        if( df < 40 ): return [0,0,0,1] 
        elif( df >= 40 and df < 60 ): return [0,0,1,0]
        elif( df >= 60 and df < 90 ): return [0,1,0,0] 
        elif( df >= 90 ): return [1,0,0,0] 
    #
    def declassif(self, df): 
        if  ( df == [0,0,0,1] ):   return 1 
        elif( df == [0,0,1,0] ):   return 2
        elif( df == [0,1,0,0] ):   return 3  
        elif( df == [1,0,0,0] ):   return 4    
    #
    def feed_data(self, url , type="", d_st = False):
        json_df = pd.DataFrame(columns=self.col_df.index) 
        df_entry = pd.Series(index=self.col_df.index)

        df_entry = df_entry.fillna(0) 
        comp_out_count = Counter()
        
        if(isinstance( url, list)):json_data = url
        else:   
            json_str=open(url).read()
            json_data = json.loads(json_str)

        for i in range(len(json_data)):
            df_entry *= 0
            m = str(json_data[i]["m"])
            df_entry.name = m
            for key in json_data[i]:
                if key == "m":  
                    pass            
                else: 
                    key_wz = int(key)#str(int(key)
                    try:
                        ds_comp = self.comp_df.loc[key_wz]
                        # fp key - 0-102 
                        print(ds_comp['fp'])   
                        if ds_comp['fp'] == NaN:col_key = 102
                        else:                             
                            col_key = int(ds_comp['fp'])
                            if col_key>101: col_key=101
                            if col_key<0: col_key=0

                        #df_entry.loc[key_wz]
                        # col_key = key_wz
                        df_entry[col_key] =  np.float32(json_data[i][key]) #amount
                    except: 
                        if d_st == True: 
                            print("m: {} -c: {} not included" .format(m, key_wz))
                        # comp_out_count[key_wz] +=1
            json_df = json_df.append(df_entry,ignore_index=False)
        
        # print("Counter of comp. not included : {}".format(len(comp_out_count)))
        # return json_df  
        return json_df.as_matrix()#.tolist()         
    def read_json(url_col, url_comp, url_lab, url_json=""):
        self.comp_df = pd.read_csv(url_comp, index_col=0, sep=',', usecols=[0,1,2,3])
        print("columns: " + str(len(self.col_df)))
        self.col_df  = pd.read_csv(url_col,  index_col=0, sep=',', usecols=[0])
        print("columns: " + str(len(self.col_df)))
        self.lab_df = pd.read_csv(url_lab, index_col=0, sep=',', usecols=[0])
        print("labels: " + str(len(self.col_df)))
        #hot encode labels: 
        filter = 0
        if filter != 0:
            self.lab_df =  self.lab_df[self.lab_df["FP"]>filter]
        
        class_f = self.classif(x)
        class_f = self.classifN(x)

        self.lab_df.insert(2, 'FP_C', self.lab_df['FP'].map(lambda x: self.classifN(x))) 

        if url_json != "": self.npData = self.feed_data(url_json)
        return

    def write_data( ): #codes, labels): 
        # write codes to file
        with open('codes', 'w') as f:
            codes.tofile(f)
        # write labels to file
        with open('labels', 'w') as f:
            writer = csv.writer(f, delimiter='\n')
            writer.writerow(labels)
        return True

    def read_data( ):
        with open('labels') as f:
            reader = csv.reader(f, delimiter='\n')
            labels = np.array([each for each in reader if len(each) > 0]).squeeze()
        with open('codes') as f:
            data = np.fromfile(f, dtype=np.float32)
            data = codes.reshape((len(labels), -1))
        return codes, labels
#  main 
def main():
    COM_DS     = "../_zfp/data/TFFRFLO_COM.csv"
    COL_DS     = "../_zfp/data/TFFRFLO_COL.csv"
    dataTest = {'label' : [] , 'data' :  [] }

    dataClassTest = fpDataModel()
    



    n_inputcm = dataClassTest.set_comp(COM_DS)
    n_inputcl = dataClassTest.set_col(COL_DS)
    print("comp={}, col={}" .format(n_inputcm, n_inputcl))

    json_str="""[{"m":"000","100028":9}
    ,{ "m":"000000000000636978", "100000000000100028" :0.009 , "000000000000660567" :0.008 , "000000000000131503" :0.008 }
    ]"""

    json_data = json.loads(json_str)
    dataTest['data'] = dataClassTest.feed_data(json_data, d_st =True) 

    print(  dataTest['data'][1]  )

if __name__ == '__main__':
    main()


