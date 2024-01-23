import pandas as pd
import numpy as np

"""
p(X) = x / n
0 < Entropy < 1 = - sum(p(X) * log_2 (p(X)))
information_gain = Entropy(parent) - [weighted average] * Entropy(children)s

build tree

stopping criteria - max depth, min number of simples, min impurity decrease

check the stopping criteria

find the best split

create child nodes
"""

# create a dataset
data = {
    'age': [25, 35, 45, 20, 30, 40, 50, 25, 35, 45, 20, 30, 40, 50],
    'income': [40, 60, 80, 20, 40, 60, 80, 30, 50, 70, 30, 50, 70, 90],
    'buys computer': ['no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes']
}

df = pd.DataFrame(data)
print(df)

def entropy(y):
    unique, counts = np.unique(y, return_counts = True)
    p = counts / len(y)
    Entropy = -np.sum(p * np.log2(p))
    return Entropy

def information_gain(X, y, split_feature):
    split_col = X[:, split_feature]
    split_values = np.unique(split_col)
    parent_entropy = entropy(y)
    children_entropy = 0

    for value in split_values:
        child_index = np.where(split_col == value)
        child_y = y[child_index]
        children_entropy += (len(child_y) / len(y)) * entropy(child_y)
    information_gain = parent_entropy - children_entropy
    return information_gain

# Max depth - This is being used as a stopping criterion in the build_tree function
max_depth = 2
print("Max Depth: ", max_depth)

def build_tree(X, y, depth, max_depth, min_samples_leaf, min_impurity_decrease):
    num_samples, num_features = X.shape
    # check stopping criteria
    if depth >= max_depth or num_samples < min_samples_leaf:
        max_depth = np.bincount(y).argmax()
        return max_depth

    # find the best split
    best_feature, best_split_value, best_information_gain = 0, 0, -np.inf

    for feature in range(num_features):
        information_gain_feature = information_gain(X, y, feature)
        if information_gain_feature > best_information_gain:
            best_feature = feature
            best_split_value = np.median(X[:, feature])
            best_information_gain = information_gain_feature

    if best_information_gain < min_impurity_decrease:
        max_depth = np.bincount(y).argmax()
        return  max_depth

    # create child nodes
    left_index = X[:, best_feature] < best_split_value
    right_index = X[:, best_feature] >= best_split_value
    left_child = build_tree(X[left_index], y[left_index], depth + 1, max_depth, min_samples_leaf, min_impurity_decrease)
    right_child = build_tree(X[right_index], y[right_index], depth + 1, max_depth, min_samples_leaf, min_impurity_decrease)
    create_node = (best_feature, best_split_value, left_child, right_child)
    print("Create Node = ", create_node)
    return create_node

X = df[['age', 'income']].values
print("X = ", X)

y = df['buys computer'].map({'no': 0, 'yes': 1}).values
print("y = ", y)

Entropy = entropy(y)
print("Entropy = ", Entropy)

Tree = build_tree(X, y, depth = 0, max_depth = 2, min_samples_leaf = 1, min_impurity_decrease = 0)
print("Tree = ", Tree)















