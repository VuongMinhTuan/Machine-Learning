from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import naive_bayes
from model import GaussianNB

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = GaussianNB()
model.fit(X_train, y_train)

print("The accuracy of my model: {}%".format(model.score(X_test, y_test) *100))


# Using Sklearn libraries
model = naive_bayes.GaussianNB()
model.fit(X_train, y_train)

print("The accuracy of Sklearn's model: {}%".format(model.score(X_test, y_test) *100))