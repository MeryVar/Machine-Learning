import numpy as np
from sklearn.datasets import load_iris

"""
t-Distributed Stochastic Neighbor Embedding algorithm

Similarity measure between two data points Xi and Xj
p(i, j) = exp(-||Xi - Xj||^2 / 2σ^2) / ∑∑ exp(-||Xk - Xl||^2 / 2σ^2)
||.|| -> Euclidean distance between two vectors
σ -> controls the variance of the Gaussian distribution

Joint probability distribution over pairs of high-dimensional data points
P = (p(i,j) + p(j,i)) / 2n   
n -> number of data points

Probability distribution over pairs of low-dimensional objects Y1, Y2, ..., Yn
q(i, j) = (1 + ||Yi - Yj||^2)^-1 / ∑∑ (1 + ||Yk - Yl||^2)^-1
||.|| -> Euclidean distance between two vectors

KL divergence between P and Q:
KL(P||Q) = ∑∑ p(i,j) log(p(i,j) / q(i,j))

Gradient of the KL divergence with respect to the low-dimensional embeddings Y
∂KL(P||Q) / ∂Yi = 4 ∑j (p(i,j) - q(i,j)) (Yi - Yj) (1 + ||Yi - Yj||^2)^-1

Update the low-dimensional embeddings Y using gradient descent
Yi := Yi - η ∂KL(P||Q) / ∂Yi
η -> learning rate
"""

def similarity_measure(X_i, X_j, sigma):
    p_i_j = np.exp(-np.linalg.norm(X_i - X_j)**2 / (2 * sigma**2))
    return p_i_j

def prob_dist_high_dimensional(X, sigma):
    n = X.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            X_i = X[i]
            X_j = X[j]
            p_i_j = similarity_measure(X_i, X_j, sigma)
            P[i, j] += p_i_j
    P = P / np.sum(P)
    P = (P + P.T) / 2 * n
    return P

def pairwise_probability(Y_i, Y_j):
    q_i_j = (1 + np.linalg.norm(Y_i - Y_j)**2)**-1
    return q_i_j

def prob_dist_low_dimensional(Y):
    n = Y.shape[0]
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Y_i = Y[i]
            Y_j = Y[j]
            q_i_j = pairwise_probability(Y_i, Y_j)
            Q[i, j] += q_i_j
    Q = Q / np.sum(Q)
    Q = (Q + Q.T) / 2 * n
    return Q

def KL_divergence(P, Q):
    if P.shape == Q.shape:
        kl_divergence = np.sum(P * np.log(P / Q))
    else:
        print("P and Q must have the same shape")
        kl_divergence = 0
    return kl_divergence

def Gradient(P, Q, Y):
    if P.shape == Q.shape and P.shape[0] == Y.shape[0]:
        n = P.shape[0]
        gradient = np.zeros_like(Y)
        for i in range(n):
            for j in range(n):
                q_i_j = Q[i, j]
                p_i_j = P[i, j]
                gradient[i] += 4 * (p_i_j - q_i_j) * Y[i] - Y[j] / 1 + np.linalg.norm(Y - Y[i])**2 + np.linalg.norm(Y - Y[j])**2
    else:
        print("P and Q must have the same shape and P and Y must have the same number of rows")
    return gradient

def update_low_dimensional(P, Q, Y, eta):
    Y = Y.astype(float)
    gradient = Gradient(P, Q, Y)
    Y -= eta * gradient
    return Y

iris = load_iris()
X = iris.data
Y = iris.target
sigma = 1
eta = 0.001

n = 3
for i in range(n):
    P = prob_dist_high_dimensional(X, sigma)
    Q = prob_dist_low_dimensional(Y)
    kl_divergence = KL_divergence(P, Q)
    gradient = Gradient(P, Q, Y)
    Y = update_low_dimensional(P, Q, Y, eta)

    print("KL divergence = ", kl_divergence)

print("P = ", P)
print("Q  = ", Q)
print("Gradient = ", gradient)
print("Y = ", Y)
