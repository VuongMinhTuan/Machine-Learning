from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from model import LogisticRegression


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

print("The accuracy of my model: {}%".format(model.score(X_test, y_test) *100))


# Using Sklearn libraries
model = linear_model.LogisticRegression(max_iter= 10000)
model.fit(X_train, y_train)

print("The accuracy of Sklearn's model: {}%".format(model.score(X_test, y_test) *100))