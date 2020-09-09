from sklearn.cluster import DBSCAN
import numpy as np

class DB_Clustering:
    """
    min_samples depends on dataset_size
    """
    def __init__(self,epsilon,min_samples):
        # epsilon is distance between points to be considered a neighbor, if it is too large then clusters may be merged
        # min_samples is number of samples in a cluster to be considered a cluster
        self.dbscan = DBSCAN(epsilon, min_samples)
        self.points = None
        self.outliers = 0

    # training set is coordinates of each face encoding in its vector space
    def add_training_set(self,points):
        self.points=points

    def train(self):
        self.dbscan.fit(self.points)
        return self.dbscan.labels_
    
    def match_picture_to_character(self, picture_encoding_dict):
        picture_character_dict={}
        for picture in picture_encoding_dict:
            characters_for_picture=[]
            for encoding in picture_encoding_dict[picture]:
                for i, x_core in enumerate(self.dbscan.components_): 
                    if not np.all(x_core - encoding):#all elements of the numpy array x_core match the encoding
                        characters_for_picture.append(self.dbscan.labels_[self.dbscan.core_sample_indices_[i]]) 
                        #core sample indices map core points to the correct label because some labels are for non core points
                    else:
                        print("x_core not in picture")
            picture_character_dict[picture] = characters_for_picture
        return picture_character_dict

    # sklearn has not implemented this functionality, the memory consumption may be large
    # furthermore by updating the training set say by the closest core point which is epsilon distance from the new point
    # It may work for the first point but it may form a line in the worst case which connects clusters
    # for prediction it may be ok as you are not adding new points but for my application I need to add new points if the 
    # training data is not enough
    # next best will be a Kmeans on dbscan clusters (consider SGD classifier later)
            # for i, x_core in enumerate(dbscan_model.components_):  [x,y,z]
            # if metric(x_new, x_core) < dbscan_model.eps:
            #     # Assign label of x_core to x_new
            #     y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]] [2,3,1]
            #     break

        # from labels we can get clusters of core points
        # if cluster is big enough (10?), we do not need to update the cluster
        # assign clusters a name label
        # get mean from cluster
        # use kmeans update
    def get_clusters(self):# numpy (n_clusters,n_features)
        clusters={} # keys: labels of clusters values: core points
        for i,label in enumerate(self.dbscan.labels_):
            i_components = self.dbscan.core_sample_indices_.index(i)
            if label not in clusters:
                clusters[label] = [self.dbscan.components_[i_components]]
            else:
                clusters[label].append[self.dbscan.components_[i_components]]
        return clusters
    
    # Determine when to cluster again
    # first option in batches of 10 photos
    # when a cluster has x photos do not cluster again for that cluster

    def check_core_point(self,point):
        # outlier is defined as closer than eps for not more than min_samples
        # if this passes the outlier check then update the point in kmeans
        # else maintain an outlier count
        # if outlier count exceeds a certain amount call dbscan again
        neighbors=0
        for i, x_core in enumerate(self.dbscan.components_):  
            if np.linalg.norm(x_core - point,ord=2) < self.dbscan.eps:
                neighbors+=1
            if neighbors>self.dbscan.min_samples:
                return True
            else: 
                self.outliers+=1
                return False

    def get_outliers(self):
        self.outliers += np.count_nonzero(self.dbscan.labels_ == -1)