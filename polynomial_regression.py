import numpy as np

class PolynomialRegression:
    def __init__(self, degree = 2, learning_rate = 0.1, num_iterations = 10):
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        self.X_poly = np.ones((X.shape[0], 1))
        for i in range(1, self.degree+1):
            self.X_poly = np.concatenate((self.X_poly, np.power(X, i)), axis=1)
            print("X_poly = ", self.X_poly)

            self.weights = np.zeros(self.X_poly.shape[1])
            self.bias = 0

        for i in range(self.num_iterations):
            y_pred = np.dot(self.X_poly, self.weights) + self.bias
            d_w = (1 / self.X_poly.shape[0]) * np.dot(self.X_poly.T, (y_pred - y))
            d_b = (1 / self.X_poly.shape[0]) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * d_w
            self.bias -= self.learning_rate * d_b

    def predict(self, X):
        X_poly = np.ones((X.shape[0], 1))
        for i in range(1, self.degree + 1):
            X_poly = np.concatenate((X_poly, np.power(X, i)), axis=1)
        return np.dot(X_poly, self.weights) + self.bias


X = np.array([[10, 2, 3], [4, 15, 61], [27, 8, 9]])
y = np.array([11, 22, 33])
print("X = ", X)
print("y = ", y)

learning_rate = PolynomialRegression(degree = 2, learning_rate = 0.1, num_iterations = 10)
learning_rate.fit(X, y)

print("Weights = ", learning_rate.weights)
print("Bias = ", learning_rate.bias)
