import numpy as np



class GaussianNB:
    def __init__(self):
        self.__mean = dict()
        self.__var = dict()
        self.__priors = dict()
        self.__classes = []
    

    def __separate_classes(self, X, y):
        n_samples = X.shape[0]
        X_c = dict().fromkeys(self.__classes)
        self.__priors = self.__priors.fromkeys(self.__classes, 0)

        for i in range(n_samples):
            if X_c[y[i]] is None:
                X_c[y[i]] = [X[i]]
            else:
                X_c[y[i]].append(X[i])

            self.__priors[y[i]] += 1/n_samples

        return X_c


    def __pdf(self, X, mean, var):
        return np.exp(-np.power(X - mean, 2)/(np.dot(2, var))) / np.sqrt(np.dot(2 * np.pi, var))


    def fit(self, X, y):
        self.__classes = np.unique(y)
        X_c = self.__separate_classes(X, y)

        for k, v in X_c.items():
            self.__mean[k] = np.mean(v, axis= 0)
            self.__var[k] = np.var(v, axis= 0)


    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = []

        for i in range(n_samples):
            posteriors = []

            for c in self.__classes:
                prior = np.log(self.__priors[c])
                posterior = np.sum(np.log(self.__pdf(X[i], self.__mean[c], self.__var[c])))
                posteriors.append(posterior + prior)
                
            y_pred.append(self.__classes[np.argmax(posteriors)])

        return y_pred

        
    def score(self, X, y):
        y_pred = self.predict(X)
        correct = 0

        for i in range(len(y)):
            if y[i] == y_pred[i]:
                correct += 1

        return correct/len(y)