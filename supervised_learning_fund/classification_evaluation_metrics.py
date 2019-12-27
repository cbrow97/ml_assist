import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

data = load_breast_cancer()
X = pd.DataFrame(data.data)
Y = pd.DataFrame(data.target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

model = tree.DecisionTreeClassifier(random_state=0)
model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

confusion_matrix = confusion_matrix(Y_test, Y_pred)

#Accuracy measures the model's ability to correctly classify all instances.
#Formula: (True Positive + True Negatives) / (Total Number of Instances)
accuracy = accuracy_score(Y_test, Y_pred)

#Precision meansures the model's ability to correctly classify positive labels (the label that represents the occurence of the event)
#by comparing it to the total number of instances predicted as positive
#Formula: (True Positives) / (True Positives + False Positives)
precision = precision_score(Y_test, Y_pred)

#Recall measures the number of correctly predicted positive labels against all positive labels
#Formula: (True Positives) / (True Positives + False Negatives)
recall = recall_score(Y_test, Y_pred)

print(confusion_matrix)
print(accuracy)
print(precision)
print(recall)
