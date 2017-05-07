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

# funcitons: max, len, sorted, complex, 
# methods: capitalize, replace, bit_length, conjugate, index, count, append, reverse, remove
HallwayIndex = house.index("hallway")

# Numpy:        For efficient work in arrays
# Matplotlib:   Data visualization
# Scikit-learn: Machine Learning  

# install package: pip3 install // pip.readthedocs.org ! 
