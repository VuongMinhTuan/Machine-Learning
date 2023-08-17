from sklearn import datasets
from sklearn.model_selection import train_test_split
from model import KNN_Classifier, KNN_Regression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# # K-Nearest Neighbors Classification
# model = KNN_Classifier()
# model.fit(X_train, y_train)
# y_predicted = model.predict(X_test)


# print("The accuracy of my model: {}%".format(model.score(X_test, y_test) *100))


# K-Nearest Neighbors Regression
model = KNN_Regression()
model.fit(X_train, y_train)

print("The accuracy of my model: {}%".format(model.score(X_test, y_test) *100))

#----------------------------------------------------------#

# # Using sklearn libraries
# # K-Nearest Neighbors Classification
# model = KNeighborsClassifier()
# model.fit(X_train, y_train)

# print("The accuracy of Sklearn's model: {}%".format(model.score(X_test, y_test) *100))


# K-Nearest Neighbors Regression
model = KNeighborsRegressor()
model.fit(X_train, y_train)

print("The accuracy of Sklearn's model: {}%".format(model.score(X_test, y_test) *100))