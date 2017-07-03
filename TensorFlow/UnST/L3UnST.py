# Linear Regr. - lesson 3 TF Standfor University
# ph1 - assemble a graph
# ph2 - use session to execute operations in a graph 
#. 
# Feed values into placeholders or variables 
# .
# 03_linear_regr. fire_thef.xls x = fires y = theft.
# want: Predict thefts from fires 
# model = W*x + b.   (Y-YP)^2 
# . 
#----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'

# Phase 1: Assemble the graph
#------------------------------------------------
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

print(data)