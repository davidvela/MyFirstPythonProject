# *************************************************
#  INTRO TO PYTHON FOR DATA SCIENCE 
# *************************************************

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath] ]

# Print out house
print(type(house)); print( house ); print(house[0]);
print(str(house[1])+"="+str(house[-4]))
# array manipulation - slice inclusive:exclusive - del(array[2]) 

# funcitons: max, len, sorted, complex,  // help(max) - documentation
# methods: capitalize, replace, bit_length, conjugate, index, count, append, reverse, remove
HallwayIndex = house.index(["hallway",hall])

# Numpy:        For efficient work in arrays
# Matplotlib:   Data visualization
# Scikit-learn: Machine Learning  

# install package: pip3 install // pip.readthedocs.org ! import as // from .. import .. 
# List - slow , operations/calculations for all data collecitons 
# Numpy - fast and possible  = numeric python - arrays of single types!

# *************************************************
#   NUMPY - way faster 
# *************************************************
import numpy as np 
np_house = np.array(house)
# np_house_2 = np_house * 2
# print(np_house_2)

# np_boolean = np_array < 21 : print(np_array[np_boolean])
# 2D numpy array: np.array( [1,2],[3,4]).shape = (2,2)  2Daray[:2, 1]
# FUNCTIONS: np.mean, np.median, corrcoef, std - standard deviation, sum(), sort(), np.round, np.random.normal 
