# Classification
class KNN_Classifier:
    def __init__(self, n_neighbors=5, p=2):
        self.k = n_neighbors
        self.p = p

    def __distance(self, data, X):
        dim = len(X)
        s = 0

        for i in range(dim):
            s = s + pow(data[i] - X[i], self.p)

        return pow(s, 1/self.p)
    
    def __kneighbors(self, X, n_neighbors):
        num_samples = len(self.X_train)
        samples_dist = []

        for i in range(num_samples):
            dist = self.__distance(X, self.X_train[i])
            samples_dist.append((dist, i))

        samples_dist = sorted(samples_dist, key= lambda x : x[0])

        return samples_dist[:n_neighbors]

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        labels = []
        num_samples = len(X)

        for i in range(num_samples):
            k_neighbors = self.__kneighbors(X[i], self.k)
            checking = {}

            for neighbor in k_neighbors:
                index = neighbor[1]

                if self.y_train[index] in checking.keys():
                    checking[self.y_train[index]] += 1
                else:
                    checking[self.y_train[index]] = 1

            checking = dict(sorted(checking.items(), key=lambda item: item[1]))
            labels.append(list(checking.keys()).pop())

        return labels


    def score(self, X, y):
        y_predicted = self.predict(X)
        count = 0

        for i in range(len(X)):
            if y_predicted[i] == y[i]:
                count += 1

        return count/len(X)
    


# Regression
class KNN_Regression:
    pass