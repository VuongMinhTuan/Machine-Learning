from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from model import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = LinearRegression()
model.fit(X_train, y_train)

print("The accuracy of my model: {}%".format(model.score(X_test, y_test) *100))


# Using Sklearn libraries
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

print("The accuracy of Sklearn's model: {}%".format(model.score(X_test, y_test) *100))