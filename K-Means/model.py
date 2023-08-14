from random import randint
import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, max_iter= 300):
        self.__k = n_clusters
        self.__max_iter = max_iter
        self.__centroids = None
        self.cluster_centers_ = None


    def __distance(self, X, centroid):
        return np.power(np.sum(np.power(np.subtract(X, centroid), 2)), 1/2)
    

    def __closest_centroid(self, X, centroids):
        distances = [self.__distance(X, cen) for cen in centroids]
        
        return distances.index(min(distances))
    

    def __create_clusters(self, X):
        p_clustered = [[] for _ in range(self.__k)]

        for p in X:
            for cen in self.__centroids:
                if p is cen:
                    continue

            index = self.__closest_centroid(p, self.__centroids)

            p_clustered[index].append(p)

        return p_clustered
    

    def __new_centroids(self, clusters):
        return [np.average(c, axis= 0) for c in clusters]
    

    def __is_converged(self, centroids, new_centroids):
        check = False
        cen_distances = [self.__distance(new_centroids[i], centroids[i]) for i in range(self.__k)]
        
        if sum(cen_distances) <= 0.003:
            check = True

        return check


    def fit(self, X):
        n_samples = X.shape[0]
        self.__centroids = [X[index] for index in np.random.choice(n_samples, self.__k, replace=False)]

        for _ in range(self.__max_iter):
            clusters = self.__create_clusters(X)
            new_centroids = self.__new_centroids(clusters)

            if self.__is_converged(self.__centroids, new_centroids):
                break

            self.__centroids = new_centroids

        self.cluster_centers_ = self.__centroids
            

    def predict(self, X):
        labels = []

        for p in X:
            labels.append(self.__closest_centroid(p, self.__centroids))

        return labels


    def accuracy_score(self, X, y):
        prediction = self.predict(X)
        correct = 0
        
        for i in range(len(prediction)):
            if prediction[i] == y[i]:
                correct += 1

        return correct/len(prediction)