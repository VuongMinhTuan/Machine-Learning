import numpy as np




class LogisticRegression:
    def __init__(self, lr= 0.01, epoches= 1000):
        self.__w = None
        self.__b = None
        self.__lr = lr
        self.__epoches = epoches


    def __sigmoid(self, S):
        return 1 / (1 + np.exp(-S))


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.__w = np.zeros(n_features)
        self.__b = 0

        for _ in range(self.__epoches):
            dw = (1 / n_samples) * np.dot((self.__sigmoid(np.dot(X, self.__w) + self.__b) - y), 2 * X)
            db = (1 / n_samples) * np.sum(self.__sigmoid(np.dot(X, self.__w) + self.__b) - y)

            self.__w  -= self.__lr * dw
            self.__b -= self.__lr * db


    def predict(self, X):
        prediction = self.__sigmoid(np.dot(X, self.__w) + self.__b)
        return [1 if p > 0.5 else 0 for p in prediction]


    def score(self, X, y):
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        check = [True if y_pred[i] == y[i] else False for i in range(n_samples)]
        
        return check.count(True) / n_samples