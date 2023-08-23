import numpy as np



class Perceptron:
    def __init__(self, lr= 1, n_iters = 1000):
        self.w = None
        self.b = None
        self.lr = lr
        self.n_iters = n_iters


    def __activation_func(self, x):
        return np.where(x > 0, 1, 0)


    def fit(self, X, y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)

            delta_w = np.dot(self.lr, np.dot(X.T, y - y_pred))
            delta_b = np.dot(self.lr, np.sum(y - y_pred))

            self.w = self.w + delta_w
            self.b = self.b + delta_b

            if np.sum(np.subtract(self.predict(X), y)) == 0:
                break


    def predict(self, X):
        prediction = np.dot(X, self.w) + self.b
        return self.__activation_func(prediction)


    def score(self, X, y):
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        correct = 0

        for i in range(n_samples):
            if y_pred[i] == y[i]:
                correct += 1

        return correct/n_samples