import numpy as np

class LogisticRegression:
    def __init__(self):
        self.theta = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_function(self, X, y):
        m = len(y)
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        J = -1 / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        return J

    def gradient_descent(self, X, y, learning_rate=0.1, iterations=10):
        m = len(y)
        J_1 = np.zeros((iterations, 1))
        for i in range(iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            self.theta = self.theta - (learning_rate / m) * np.dot(X.T, (h - y))
            J_1[i] = self.loss_function(X, y)
        print("z = ", z)
        print("h = ", h)
        print("J_1 = ", J_1)
        return self.theta

    def fit(self, X, y, learning_rate = 0.1, iterations = 10):
        self.theta = np.zeros((X.shape[1], 1))
        self.theta = self.gradient_descent(X, y, learning_rate, iterations)

    def predict(self, X):
        z = np.dot(X, self.theta)
        return self.sigmoid(z)

X = np.array([[1, 4], [1, 2], [5, 3]])
y = np.array([2, 1, 0]).reshape(-1, 1)
print("X = ", X)
print("y = ", y)

lr = LogisticRegression()
lr.fit(X, y)
print("Final value of theta = ", lr.theta)

s = lr.predict(X)
print("Predicted sigmoid values = ", s)