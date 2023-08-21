import numpy as np



class LinearRegression:
    def __init__(self):
        self.weight = None
        self.bias = None


    def fit(self, X, y):
        one = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((one, X), axis= 1)
        
        w = np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, y))
        self.weight = np.array([w[1:]]).T
        self.bias = w[0]


    def predict(self, X):
        n_samples = X.shape[0]
        prediction = []

        for i in range(n_samples):
            y_pred = np.dot(X[i], self.weight) + self.bias
            prediction.append(y_pred[0])

        return prediction



    def score(self, X, y):
        y_predicted = self.predict(X)
        y_mean = 0
        RSS = 0
        TSS = 0

        for l in y:
            y_mean += l

        y_mean = y_mean/len(y)

        for i in range(len(y)):
            RSS += pow(y[i] - y_predicted[i], 2)
            TSS += pow(y[i] - y_mean, 2)

        return (1 - (RSS/TSS))