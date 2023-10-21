import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Importing the dataset
data = pd.read_csv('data.csv')

distance = np.array(data.iloc[:, 0].values)
time = np.array(data.iloc[:, 1].values) 

# Plotting the data
plt.scatter(distance, time)
plt.xlabel('Distance')
plt.ylabel('Time')
plt.title('Time vs Distance')
plt.show()


# Fitting the data to the model
pred = LinearRegression().fit(distance.reshape(-1, 1), time.reshape(-1, 1))

x_pred = np.linspace(0, 60, 100)
y_pred = pred.predict(x_pred.reshape(-1, 1))

# Plotting the model
plt.plot(x_pred, y_pred, color='red')
plt.scatter(distance, time)

plt.show()

input()