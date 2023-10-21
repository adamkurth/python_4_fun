import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression

# Importing the dataset
data = pd.read_csv('data.csv')

distance = np.array(data.iloc[:, 0].values)
time = np.array(data.iloc[:, 1].values) 

# Fitting the data to the model
pred = LinearRegression().fit(time.reshape(-1, 1), distance.reshape(-1, 1))

x_pred = np.linspace(0, 60, 100)
y_pred = pred.predict(x_pred.reshape(-1, 1))

# Plotting the model
plt.plot(x_pred, y_pred, color='red')
plt.scatter(time, distance)

plt.show()

input()