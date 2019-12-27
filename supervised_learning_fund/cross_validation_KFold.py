import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold


#iris_data = load_iris()
#X = pd.DataFrame(iris_data.data)
#Y = pd.DataFrame(iris_data.target)
#
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#
#kf = KFold(n_splits=10)
#splits = kf.split(X)
#
#for train_index, dev_index in splits:
#    X_train, X_dev = X.iloc[train_index], X.iloc[dev_index]
#    Y_train, Y_dev = Y.iloc[train_index], Y.iloc[dev_index]
#
#print(len(X_train), len(X_dev))

digits_data = load_digits()
X = pd.DataFrame(digits_data.data)
Y = pd.DataFrame(digits_data.target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

dev_proportion = round(len(X_test) / len(X_train), 2)

#X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=dev_proportion)


kf = KFold(n_splits=10)
splits = kf.split(X_train)

for train_index, dev_index in splits:
    X_train, X_dev = X.iloc[train_index], X.iloc[dev_index]
    Y_train, Y_dev = Y.iloc[train_index], Y.iloc[dev_index]
    
print(splits)