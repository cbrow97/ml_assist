import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = load_boston()
X = pd.DataFrame(data.data)
Y = pd.DataFrame(data.target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


model = linear_model.LinearRegression()
model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#Mean Absolute Error
#Measures the average absolute difference between a prediction and the ground truth, without taking into account the direction of the error
MAE = mean_absolute_error(Y_test, Y_pred)

#Root Mean Squared Error
#A quadratic metric that also measures the average magnitude of error betweem the ground truth and the prediction
RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))

print(MAE)
print(RMSE)