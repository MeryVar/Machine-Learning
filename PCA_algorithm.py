import numpy as np

"""
X matrix of size nxm 

Data standardization 
μ = (1/n) * ∑(x_i)     mu
σ = sqrt((1/n-1) * (∑x_i^2 - n*μ^2))   sigma
Z = (X - μ) / σ     size nxm

Covariance matrix
C = Z^T * Z / (n - 1)    size mxm

Calculate the eigenvalues and eigenvectors of the covariance matrix
λ is eigenvalues
V is eigenvectors

Sort the eigenvectors in descending order of their corresponding eigenvalues

Choose the first k eigenvectors and form a projection matrix W
W = [v_1, v_2, ..., v_k]    size mxn

Project the standardized data matrix Z onto the reduced space by multiplying it by the projection matrix W
Y = Z * W    size nxk

X_new is the reconstructed data matrix
X_new = Y * W^T

the percentage of information saved in PCA
% of information saved = (sum of eigenvalues of top k components / sum of all eigenvalues) * 100%
"""

def mean_sigma (X,x):
    n = X.shape[0]
    col = X[:, x]
    mu = np.sum(col) / n
    sigma = np.sqrt(np.sum((col - mu) ** 2) / (n - 1))
    return mu, sigma

def standardize(X):
    mu, sigma = mean_sigma(X,0)
    Z = (X - mu) / sigma
    return Z

def covariance_matrix(Z):
    n = Z.shape[0]
    C = np.dot(Z.T, Z) / (n - 1)
    return C

def eigenvalues_eigenvectors(C):
    eigenvalues, eigenvectors = np.linalg.eig(C)
    return eigenvalues, eigenvectors

def sort_eigenvalues_descending(eigenvalues, eigenvectors):
    i = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[i]
    eigenvalues_descending = eigenvalues[i]
    eigenvectors_descending = eigenvectors.T[i].T
    return eigenvalues_descending, eigenvectors_descending

def pca(X, k):
    Z = standardize(X)
    C = covariance_matrix(Z)
    eigenvalues, eigenvectors = np.linalg.eig(C)
    eigenvalues_descending, eigenvectors_descending = sort_eigenvalues_descending(eigenvalues, eigenvectors)
    W = eigenvectors_descending[:, :k]
    Y = np.dot(Z, W)
    X_new = np.dot(Y, W.T)
    return Y, X_new, eigenvalues_descending, eigenvectors_descending

def percent_info_saved(eigenvalues, k):
    top_components = np.sum(eigenvalues[:k])
    total_eigenvalues = np.sum(eigenvalues)
    percent_saved = (top_components  / total_eigenvalues ) * 100
    return percent_saved

n = int(input("Observations n = "))
m = int(input("Variables m = "))

X = np.random.rand(n, m)
print("X = ", X)

mu = mean_sigma(X,0)
print("μ = ", mu)

sigma = mean_sigma(X,0)
print("σ = ", sigma)

Z = standardize(X)
print("Z = ", Z)

C = covariance_matrix(Z)
print("C = ", C)

eigenvalues, eigenvectors = eigenvalues_eigenvectors(C)
print("λ = ", eigenvalues)
print("V = ", eigenvectors)

k = int(input("k = "))

Y, X_new, eigenvalues_descending, eigenvectors_descending = pca(X, k)
print("Y = ", Y)
print("X = ", X_new)
print("Sorted λ = ", eigenvalues_descending)
print("Sorted V = ", eigenvectors_descending)

percent_saved = percent_info_saved(eigenvalues_descending, k)
print("Percent of information saved = {:.2f}%".format(percent_saved))




