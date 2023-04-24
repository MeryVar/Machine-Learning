import numpy as np

# Algorithm KMeans
# Let  X = {x1,x2,x3,……..,xn} be the set of data points
# 1) Randomly select ‘c’ cluster centers.
# 2) Calculate the distance between each data point and cluster centers.
# euclidean distance = sqrt(sum((x_i - y_i)^2) for i in range(d))
# 3) Assign the data point to the cluster center whose distance from the cluster center is minimum of all the cluster centers.
# 4) Recalculate the new cluster center

class KMeans:
    def __init__(self, k = 2, n = 10):
        self.k = k
        self.n = n

    def fit(self, X):
        self.c = X[np.random.choice(range(len(X)), self.k)]

        for i in range(self.n):
            euclidean_distance = np.sqrt(((X - self.c[:, np.newaxis])**2).sum(axis=2))
            self.indices_clusters = np.argmin(euclidean_distance, axis=0)

            for j in range(self.k):
                self.c[j] = X[self.indices_clusters == j].mean(axis=0)

    def predict(self, X):
        euclidean_distance = np.sqrt(((X - self.c[:, np.newaxis])**2).sum(axis=2))
        indices_clusters = np.argmin(euclidean_distance, axis=0)
        print("Euclidean Distance = ", euclidean_distance)
        return indices_clusters

X = np.random.rand(10, 5)
print("X = " ,X)
kmeans = KMeans(k = 2, n = 10)
kmeans.fit(X)
indices_clusters = kmeans.predict(X)
print("Indices Clusters = ", indices_clusters)