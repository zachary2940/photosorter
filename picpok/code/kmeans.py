from sklearn.clustering import MiniBatchKMeans
import numpy as np

class KMeans_Clustering:
    def __init__(self):
        self.kmeans=None
    
    def calculate_centroids(self,clusters,n_features):
        n_clusters = len(clusters)
        centroid_arr = np.zeros((n_clusters,n_features))
        for i,label in enumerate(clusters):
            kmeans = MiniBatchKMeans(n_clusters=1).fit(clusters[label])
            centroid_arr[i]=kmeans.cluster_centers_ #broadcasting
        return centroid_arr
    
    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_
    
    def get_kmeans_model(self,centroid_arr):
        self.kmeans =  MiniBatchKMeans(init=centroid_arr)
    
    #after checking for outlier
    def update_kmeans_model(self,point):
        self.kmeans = self.kmeans.partial_fit(point)

    def predict(self,kmeans,point):
        return self.kmeans.predict(point)