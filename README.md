# Linear Regression with L1/L2 Regularization

This Python script implements a simple linear regression model with optional L1 or L2 regularization. Linear regression is a supervised learning algorithm used for predicting a continuous outcome variable based on one or more predictor features.

## Usage

1. Ensure you have NumPy installed. If not, you can install it using:

   ```bash
   pip install numpy
   ```

2. Run the script `Linear_regression_L1_L2_regularization.py`:

   ```bash
   python Linear_regression_L1_L2_regularization.py
   ```

## Notes

- The script includes an example with sample data (`X` and `y`).
- The regularization term can be applied by setting the `regularization` parameter to 'L1' or 'L2'.
- Adjust the `iteration` and `learn_rate` parameters based on your dataset and convergence requirements.
- Make sure your dataset is appropriately scaled for better convergence.
  

# Logistic Regression with Gradient Descent

This Python script implements logistic regression using gradient descent. Logistic regression is a supervised learning algorithm used for binary classification tasks.

## Usage

1. Ensure you have NumPy installed. If not, you can install it using:

   ```bash
   pip install numpy
   ```

2. Run the script `logistic_regression.py`:

   ```bash
   python logistic_regression.py
   ```

## Notes

- The script includes an example with sample data (`X` and `y`).
- Adjust the `learning_rate` and `iterations` parameters based on your dataset and convergence requirements.
- The model is designed for binary classification tasks where the target variable `y` is binary (0 or 1).
- Make sure your dataset is appropriately scaled for better convergence.

# Polynomial Regression

This Python script implements polynomial regression using gradient descent. Polynomial regression is a type of regression analysis where the relationship between the independent variable `X` and the dependent variable `y` is modeled as an nth-degree polynomial.

## Usage

1. Ensure you have NumPy installed. If not, you can install it using:

   ```bash
   pip install numpy
   ```

2. Run the script `polynomial_regression.py`:

   ```bash
   python polynomial_regression.py
   ```

## Notes

- The script includes an example with sample data (`X` and `y`).
- Adjust the `degree`, `learning_rate`, and `num_iterations` parameters based on your dataset and convergence requirements.
- The model fits a polynomial of the specified degree to the input features.
- Make sure your dataset is appropriately scaled for better convergence.

# KMeans Clustering

This Python script implements the KMeans clustering algorithm. KMeans is an unsupervised machine learning algorithm used for partitioning data into 'k' distinct, non-overlapping subsets (clusters). The algorithm iteratively assigns data points to clusters and updates cluster centers until convergence.

## Usage

1. Ensure you have NumPy installed. If not, you can install it using:

   ```bash
   pip install numpy
   ```

2. Run the script `kmeans_clustering.py`:

   ```bash
   python kmeans_clustering.py
   ```

## Notes

- The script includes an example with randomly generated data (`X`).
- Adjust the number of clusters (`k`) and the number of iterations (`n`) in the `KMeans` class constructor based on your requirements.
- The `fit` method fits the KMeans model to the data points.
- The `predict` method assigns data points to the nearest cluster based on Euclidean distance.
- The script prints the Euclidean distance for each data point and the final indices of the assigned clusters.

# K-Nearest Neighbors (KNN) Algorithm

This Python script implements the K-Nearest Neighbors (KNN) algorithm, a simple and effective classification algorithm based on the principle of finding the 'k' nearest neighbors to a given data point in the feature space.

## Usage

1. Ensure you have NumPy installed. If not, you can install it using:

   ```bash
   pip install numpy
   ```

2. Run the script `knn_algorithm.py`:

   ```bash
   python knn_algorithm.py
   ```

## Notes

- The script includes an example with training data (`X_train` and `y_train`) and test data (`X_test`).
- The `KNN` class allows you to set the number of neighbors (`k`) in the constructor.
- The `fit` method is used to provide the training data to the KNN model.
- The `predict` method predicts the labels for the test data based on the k-nearest neighbors.
- The example prints the training and test data, the specified value of `k`, and the predicted labels for the test data.

# Naive Bayes Classifier

This Python script implements a Naive Bayes classifier, a probabilistic algorithm based on Bayes' theorem that assumes independence between features. The classifier calculates the posterior probabilities of each class given the input features and predicts the class with the highest probability.

## Usage

1. Ensure you have NumPy installed. If not, you can install it using:

   ```bash
   pip install numpy
   ```

2. Run the script `naive_bayes_classifier.py`:

   ```bash
   python naive_bayes_classifier.py
   ```

## Notes

- The script includes an example with feature data (`X`) and corresponding class labels (`y`).
- The `NaiveBayes` class provides methods for fitting the model (`fit`) and making predictions (`predict`).
- The `fit` method calculates the mean, variance, and class priors based on the training data.
- The `predict` method predicts the class labels for input data using the calculated probabilities.
- The example prints the feature data, class labels, and the predicted class labels based on the Naive Bayes classifier.

# Principal Component Analysis (PCA)

This Python script performs Principal Component Analysis (PCA), a dimensionality reduction technique that transforms a dataset into a lower-dimensional space by preserving the most important information. The steps include data standardization, calculating the covariance matrix, obtaining eigenvalues and eigenvectors, sorting them, and projecting the data into the reduced space.

## Usage

1. Ensure you have NumPy installed. If not, you can install it using:

   ```bash
   pip install numpy
   ```

2. Run the script `pca_analysis.py`:

   ```bash
   python pca_analysis.py
   ```

3. Input the number of observations (`n`) and variables (`m`) when prompted.

4. Input the value of `k`, the number of principal components to retain.

## Notes

- The script includes an example with randomly generated data (`X`).
- The PCA process involves standardizing the data, calculating the covariance matrix, obtaining eigenvalues and eigenvectors, sorting them in descending order, and projecting the data into a reduced space.
- The percentage of information saved in PCA is also calculated and printed.

# t-Distributed Stochastic Neighbor Embedding (t-SNE) Algorithm

This Python script implements the t-Distributed Stochastic Neighbor Embedding (t-SNE) algorithm, a dimensionality reduction technique that is particularly effective for visualizing high-dimensional data in lower-dimensional space. The algorithm measures similarities between data points in both high and low-dimensional spaces and minimizes the divergence between their probability distributions.

## Usage

1. Ensure you have NumPy and scikit-learn installed. If not, you can install them using:

   ```bash
   pip install numpy scikit-learn
   ```

2. Run the script `t_sne_algorithm.py`:

   ```bash
   python t_sne_algorithm.py
   ```

## Notes

- The script uses the t-SNE algorithm to reduce the dimensionality of the Iris dataset.
- It calculates the probability distributions `P` and `Q`, the Kullback-Leibler (KL) divergence between them, and updates the low-dimensional embeddings `Y` using gradient descent.
- The script prints the KL divergence and other relevant information after each iteration.

# Decision Tree for Classification

This Python script implements a simple decision tree for binary classification. The script includes functions to calculate entropy, information gain, and build a decision tree based on the provided dataset. The decision tree is constructed recursively using the ID3 algorithm, and stopping criteria such as maximum depth, minimum samples per leaf, and minimum impurity decrease are considered.

## Usage

1. Ensure you have NumPy and Pandas installed. If not, you can install them using:

   ```bash
   pip install numpy pandas
   ```

2. Run the script `decision_tree_classification.py`:

   ```bash
   python decision_tree_classification.py
   ```

## Notes

- The script uses a simple dataset with features 'age' and 'income' to predict whether a person will buy a computer ('buys computer' column).
- The decision tree is built using the ID3 algorithm with stopping criteria for maximum depth, minimum samples per leaf, and minimum impurity decrease.
- The script prints the entropy of the target variable, the information gain for each feature, and the nodes of the decision tree as it is being constructed.

# Ensemble Learning with Random Forest and Boosting Models

This Python script demonstrates ensemble learning using Random Forest, Gradient Boosting, and XGBoost models for both classification and regression tasks. The script splits the dataset into training and testing sets, trains Random Forest Classifier and Regressor models, as well as Gradient Boosting and XGBoost models, and evaluates their performance.

## Usage

1. Ensure you have NumPy, Pandas, scikit-learn, and XGBoost installed. If not, you can install them using:

   ```bash
   pip install numpy pandas scikit-learn xgboost
   ```

2. Run the script `ensemble_learning.py`:

   ```bash
   python ensemble_learning.py
   ```

## Notes

- The script uses a simple dataset with features 'age' and 'income' to predict whether a person will buy a computer ('buys computer' column).
- It splits the dataset into training and testing sets and trains Random Forest Classifier and Regressor models, Gradient Boosting models, and XGBoost models.
- The script prints the majority vote and mean of predictions for classification and regression tasks, respectively.
- Accuracy for classification tasks and Mean Squared Error (MSE) for regression tasks are calculated for each model.

# Decision Tree and Random Forest

This Python script demonstrates the implementation of a Decision Tree and a Random Forest for both classification tasks. The dataset used contains information about individuals, including their age, income, and whether they buy a computer.

## Decision Tree

1. The script defines functions for entropy calculation, information gain calculation, and decision tree building.
2. The Decision Tree is built using a recursive approach, considering stopping criteria such as maximum depth, minimum number of samples per leaf, and minimum impurity decrease.
3. A sample dataset is created, and the Decision Tree is trained and printed.

## Random Forest

1. The script defines a function for building a Random Forest, which creates multiple decision trees using bootstrap samples.
2. Each tree is trained on a subset of the dataset, and the resulting forest is returned.
3. A sample dataset is created, and the Random Forest is trained and printed.

## Usage

1. Ensure you have NumPy and Pandas installed. If not, you can install them using:

   ```bash
   pip install numpy pandas
   ```

2. Run the script `decision_tree_random_forest.py`:

   ```bash
   python decision_tree_random_forest.py
   ```

## Notes

- The Decision Tree and Random Forest are implemented for classification tasks in this script.
- The functions can be adapted for regression tasks as well.
- The script prints the Decision Tree structure and the Random Forest, providing insights into the individual trees that make up the forest.