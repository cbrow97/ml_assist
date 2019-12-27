import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

data = load_digits()

X = pd.DataFrame(data.data)
Y = pd.DataFrame(data.target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0) 

model = tree.DecisionTreeClassifier(random_state=0)
model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

confusion_matrix = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

#Precision and Recall can only be preformed on binary values. To test the accuracy of only the number 6, we need to break the values into
# a one-hot vector. A one hot vector is a vector that contains zeros and ones. 0 will represent the number 6 and 0 will represent any other number
Y_test_2 = Y_test[:]
Y_test_2[Y_test_2 != 6] = 1
Y_test_2[Y_test_2 == 6] = 0

Y_pred_2 = Y_pred[:]
Y_pred_2[Y_pred_2 != 6] = 1
Y_pred_2[Y_pred_2 == 6] = 0

precision = precision_score(Y_test_2, Y_pred_2)
recall = recall_score(Y_test_2, Y_pred_2)

print(accuracy, precision, recall)