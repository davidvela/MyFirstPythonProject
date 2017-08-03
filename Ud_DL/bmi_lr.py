# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv') 

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[["BMI"]], bmi_life_data[["Life expectancy"]] )

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([21.07931])


# results
# /usr/local/lib/python3.4/dist-packages/sklearn/utils/validation.py:386: DeprecationWarning: 
# Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19. 
# Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) 
# if it contains a single sample.  DeprecationWarning)
# .
# 1. The data was loaded correctly!
# 2. Well done, you fitted the model!
# 3. Well done, your prediction of a life expectancy 60.315647164 is correct!