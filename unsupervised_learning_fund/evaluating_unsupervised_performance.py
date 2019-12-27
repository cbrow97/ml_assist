from sklearn.metrics import silhouette_score, calinski_harabasz_score
import DBSCAN_algorithm as dbscan
import kmeans_algorithm as km
import meanshift_algorithm as ms
import pandas as pd
import DataPrep

data = pd.read_csv(r'C:\Users\cb049c\Documents\Python Projects\ml_assist\Wholesale customers data.csv')
dr = DataPrep.DataRescaling()
data_standarized = dr.rescale_standarization(data)


kmeans_score = silhouette_score(data_standarized, km.pred_kmeans, metric='euclidean')
meanshift_score = silhouette_score(data_standarized, ms.pred_meanshift, metric='euclidean')
dbscan_score = silhouette_score(data_standarized, dbscan.pred_dbscan, metric='euclidean')

print(kmeans_score, meanshift_score, dbscan_score)

kmeans_score = calinski_harabasz_score(data_standarized, km.pred_kmeans)
meanshift_score = calinski_harabasz_score(data_standarized, ms.pred_meanshift)
dbscan_score = calinski_harabasz_score(data_standarized, dbscan.pred_dbscan)

print(kmeans_score, meanshift_score, dbscan_score)