import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans 
import DataPrep
from sklearn.metrics import silhouette_score, calinski_harabaz_score

##Find the breaking point of the dataset. By looking at the plot we see that the breaking point is 5
##5 clusters (k) will be used
#n_samples = 1500
#data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)[0]
#ideal_k = []
#for i in range(1,21):
#    est_kmeans = KMeans(n_clusters=i)
#    est_kmeans.fit(data)
#
#    ideal_k.append([i, est_kmeans.inertia_])
#
#ideal_k = np.array(ideal_k)
#
#plt.plot(ideal_k[:,0], ideal_k[:,1])
#plt.show()

##Train the model with K=5
#est_kmeans = KMeans(n_clusters=5)
#est_kmeans.fit(data)
#pred_kmeans = est_kmeans.predict(data)
#
#plt.scatter(data[:,0], data[:,1], c=pred_kmeans)
#plt.show()


data = pd.read_csv(r'C:\Users\cb049c\Documents\Python Projects\ml_assist\Wholesale customers data.csv')
#dp = DataPrep.DataPreprocessing()
#data = dp.remove_numerical_outliers(data)

dr = DataPrep.DataRescaling()
#data_standardized = 
data_standardized = dr.rescale_standarization(data)

ideal_k = []
for i in range(1,21):
    est_kmeans = KMeans(n_clusters=i)
    est_kmeans.fit(data_standardized)

    ideal_k.append([i, est_kmeans.inertia_])

ideal_k = np.array(ideal_k)

#plt.plot(ideal_k[:,0], ideal_k[:,1])
#plt.show()

est_kmeans = KMeans(n_clusters=6)
est_kmeans.fit(data_standardized)
pred_kmeans = est_kmeans.predict(data_standardized)

plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16,8))
plt.scatter(data.iloc[:,5], data.iloc[:,3], c=pred_kmeans, s=20)
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.xlabel('Frozen')
plt.subplot(1, 2, 1)
plt.scatter(data.iloc[:,4], data.iloc[:,3], c=pred_kmeans, s=20)
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.xlabel('Grocery')
plt.ylabel('Milk')
#plt.show()
