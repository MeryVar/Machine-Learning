import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = 0
        self.bias = 0

    def fit(self, X, y, iteration = 10, learn_rate = 0.1, regularization = '0'):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for i in range(iteration):
            y_pred = np.dot(X, self.weights) + self.bias
            d_w = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            d_b = (1 / n_samples) * np.sum(y_pred - y)

            if regularization == 'L1':
                d_w += learn_rate * np.abs(self.weights)
            elif regularization == 'L2':
                d_w += learn_rate * np.sqr(self.weights)

            self.weights -= learn_rate * d_w
            self.bias -= learn_rate * d_b
        print("d_w = ", d_w)

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

X = np.array([[10, 2, 3], [4, 15, 61], [27, 8, 9]])
y = np.array([11, 22, 33])
print("X = ", X)
print("y = ", y)

learn_rate = LinearRegression()
learn_rate.fit(X, y, iteration = 10,  learn_rate = 0.1, regularization = '0')

print("Weights = ", learn_rate.weights)
print("Bias = ", learn_rate.bias)

y_pred = learn_rate.predict(X)
print("Predictions = ", y_pred)


