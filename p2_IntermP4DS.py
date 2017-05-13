
# *************************************************
# INTERMEDIATE PYTHON FOR DATA SCIENCE 
# *************************************************

# 1 NATPLOTLIB! 
# 2 DICTIONARIES AND PANDAS.  
# 3 LOGIC, CONTORL FLOW  AND FILTER.
# 4 LOOPS. 
# 5 CASE STUDY: HACKER STATISTICS

import matplotlib.pyplot as plt 
import numpy as np 

year = [2010,2011,2012] # x
plot = [10,20,30] # y

# Print the last item from year and pop
print(year[-1]); print(plot[-1])
# plt.plot(year,plot); 
plt.scatter(year, plot)
# Put the x-axis on a logarithmic scale - plt.xscale('log')

plt.xlabel('Year')
plt.ylabel('Popuplation')
plt.title('World Population Projections')
plt.yticks([0, 10, 20, 30], ['0B', '10B', '20B', '30B'], ) # Billions

plt.text(2010, 10, 'Hola')
plt.grid(True)
plt.show()

# documentation - help(plt.hist) // plt.plot
