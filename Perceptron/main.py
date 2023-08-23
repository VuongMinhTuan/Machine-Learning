from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from model import Perceptron


X, y = datasets.make_blobs(n_samples= 150, n_features= 2, centers= 2, cluster_std= 1.05, random_state= 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 123)


model = Perceptron()
model.fit(X_train, y_train)

print("The accuracy of my model: {}%".format(model.score(X_test, y_test) *100))


# Using Sklearn libraries
model = linear_model.Perceptron()
model.fit(X_train, y_train)

print("The accuracy of Sklearn's model: {}%".format(model.score(X_test, y_test) *100))