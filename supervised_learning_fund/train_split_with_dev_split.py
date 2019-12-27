import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X = pd.DataFrame(iris_data.data)
Y = pd.DataFrame(iris_data.target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

test_proportion = len(X_test) / len(X_train)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size = 0.25)

print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape, X_test.shape, Y_test.shape)