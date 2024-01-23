import numpy as np

class KNN:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        dists = np.sqrt(np.sum((self.X_train - X_test[:, np.newaxis])**2, axis=2))
        closest_y = self.y_train[np.argsort(dists, axis=1)[:, :self.k]]
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=closest_y)
        return y_pred

X_train = np.array([[11, 1], [3, 24], [1, 4]])
y_train = np.array([2, 1, 4])
X_test = np.array([[15, 26], [7, 18]])
print("X_train = ", X_train)
print("y_train = ", y_train)
print("X_test = ", X_test)

k = 3

knn = KNN(k = k)
knn.fit(X_train = X_train, y_train = y_train)
y_pred = knn.predict(X_test = X_test)

print("y_pred = ", y_pred)

