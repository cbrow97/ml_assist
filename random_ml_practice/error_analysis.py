import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

data = load_digits()
X = pd.DataFrame(data.data)
Y = pd.DataFrame(data.target)

X_new, X_test, Y_new, Y_test = train_test_split(X, Y, test_size=.1, random_state=101)

dev_proportion = round((len(X_test)/len(X_new)), 2)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_new, Y_new, test_size=dev_proportion, random_state=101)

#print(X_new.shape, X_test.shape, Y_new.shape, Y_test.shape)
#print(X_train.shape, X_dev.shape, Y_train.shape, Y_dev.shape)

np.random.seed(101)

indicies_train = np.random.randint(0, len(X_train), 89)
indicies_dev = np.random.randint(0, len(X_dev), 89)

X_train_dev = pd.concat([X_train.iloc[indicies_train,:],
                        X_dev.iloc[indicies_dev,:]])

Y_train_dev = pd.concat([Y_train.iloc[indicies_train,:],
                        Y_dev.iloc[indicies_dev,:]])

X_sets = [X_train, X_train_dev, X_dev, X_test]
Y_sets = [Y_train, Y_train_dev, Y_dev, Y_test]

model = tree.DecisionTreeClassifier(random_state=101)
model = model.fit(X_train, Y_train)

scores = []
for i in range(0, len(X_sets)):
    pred = model.predict(X_sets[i])
    accuracy = accuracy_score(Y_sets[i], pred)
    scores.append(accuracy)

scores_updated = [1 - score for score in scores]
print(scores_updated)
