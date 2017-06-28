from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


#COLUMNS = ["age", "workclass", "fnlwgt", ...]
LABEL_COLUMN = "Class"

# paint 

#Main function
def main(_):
    print("Start main")
    file_name = "iris_iris.csv"

    # Read my file
    if not os.path.exists(file_name): 
        print("Iris file not found")
    else: 
        df_train = pd.read_csv(
            tf.gfile.Open(file_name),
            #names=COLUMNS,
            skipinitialspace=True,
            engine="python")
        print(df_train)
        
if __name__ == "__main__":
  tf.app.run(main=main, argv=[sys.argv[0]])