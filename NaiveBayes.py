import numpy as np

# posterior probs = P(c|x)
# likelihood = P(x|c)
# class priors = P(c)
# prior prob = P(x)
# P(c|x) = (P(x|c)P(c))/P(x) naive bayes
# y = argmax log(P(x_1|c)...log(P(x_n|c) + logP(y))
# f(x, myu, sigma) = exp(-0.5*sqr((x-myu)/sigma)/sqrt(2pi * sigma**2)
# var = sigma**2    myu = mean

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        #mean, var, prior
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros((n_classes))

        for i, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[i, :] = X_c.mean(axis = 0)
            self.var[i, :] = X_c.var(axis = 0)
            self.prior[i] = X_c.shape[0] / n_samples

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        y_pred = np.array(y_pred)
        return y_pred

    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.prior[i])
            likelihood = np.sum(np.log(self.pdf(i,x)))
            posterior = likelihood + prior
            posteriors.append(posterior)
            y =self.classes[np.argmax(posteriors)]
        return y

    def pdf(self, j, x):
        mean = self.mean[j]
        var = self.var[j]
        f = np.exp(-((x - mean) ** 2) / ( 2 * var)) / np.sqrt(2 * np.pi * var)
        return f

X = np.array([[40, 35, 45],
              [25, 15, 30],
              [10, 20, 50]])

y = np.array(['A', 'B', 'C'])
print("X = ", X)
print("y = ", y)

nb = NaiveBayes()
nb.fit(X, y)

y_pred = nb.predict(X)
print("Prediction =  ", y_pred)
