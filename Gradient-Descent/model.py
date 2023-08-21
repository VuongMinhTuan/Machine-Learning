import numpy as np



class LinearRegression:
    def __init__(self, lr= 0.001, max_iters= 10000):
        self.__w = None
        self.__b = None
        self.__lr = lr
        self.__n_iters = max_iters
        pass


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.__w = np.zeros(n_features)
        self.__b = 0
        it = 0

        for _ in range(self.__n_iters):
            it += 1

            dw = (1/n_samples) * np.dot(X.T, np.dot(X, self.__w) + self.__b - y)
            db = (1/n_samples) * np.sum(np.dot(X, self.__w) + self.__b - y)

            w_new = self.__w - np.dot(self.__lr, dw)
            b_new = self.__b - self.__b * db

            if np.sum(w_new - self.__w) <= 0.001 * n_features and b_new - self.__b <= 0.001:
                break

            self.__w = w_new
            self.__b = b_new


    def predict(self, X):
        return np.dot(X, self.__w) + self.__b


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