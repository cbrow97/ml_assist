import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN 
import DataPrep

#n_samples = 1500
#data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)[0]
#est_dbscan = DBSCAN(eps=.1)
#pred_dbscan = est_dbscan.fit_predict(data)
#
#plt.scatter(data[:,0], data[:,1], c=pred_dbscan)
#plt.show()

data = pd.read_csv(r'C:\Users\cb049c\Documents\Python Projects\ml_assist\Wholesale customers data.csv')

dr = DataPrep.DataRescaling()
data_standardized = dr.rescale_standarization(data)

est_dbscan = DBSCAN(eps=0.8)
pred_dbscan = est_dbscan.fit_predict(data_standardized)

plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16,8))
plt.scatter(data.iloc[:,5], data.iloc[:,3], c=pred_dbscan, s=20)
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.xlabel('Frozen')
plt.subplot(1, 2, 1)
plt.scatter(data.iloc[:,4], data.iloc[:,3], c=pred_dbscan, s=20)
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.xlabel('Grocery')
plt.ylabel('Milk')
#plt.show()