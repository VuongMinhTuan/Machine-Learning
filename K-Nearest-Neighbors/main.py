from sklearn import datasets
from sklearn.model_selection import train_test_split
from model import KNN_Classifier

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = KNN_Classifier()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print(y_predicted)
print(model.score(X_test, y_test))