import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import recall_score

data = load_breast_cancer()
X = pd.DataFrame(data.data)
Y = pd.DataFrame(data.target)

X_new, X_test, Y_new, Y_test = train_test_split(X, Y, test_size=.1, random_state=101)

dev_proportion = round((len(X_test)/len(X_new)), 2)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_new, Y_new, test_size=0.11, random_state=101)

np.random.seed(101)

indicies_train = np.random.randint(0, len(X_train), 25)
indicies_dev = np.random.randint(0, len(X_dev), 25) 

X_train_dev = pd.concat([X_train.iloc[indicies_train,:],
                X_dev.iloc[indicies_dev,:]])

Y_train_dev = pd.concat([Y_train.iloc[indicies_train,:],
                Y_dev.iloc[indicies_dev,:]])


model = tree.DecisionTreeClassifier(random_state=101)
model = model.fit(X_train, Y_train)

X_sets = [X_train, X_train_dev, X_dev, X_test]
Y_sets = [Y_train, Y_train_dev, Y_dev, Y_test]

scores = []
for i in range(0, len(X_sets)):
    pred = model.predict(X_sets[i])
    score = recall_score(Y_sets[i], pred)
    scores.append(score)

for i in scores:
    print(1 - i)