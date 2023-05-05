import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb

"""
RANDOM FOREST
Split data into training and testing set
Train Random Forest Classifier
Train Random Forest Regressor
Get predictions from each tree
Trains Gradient Boosting Classifier, Gradient Boosting Regressor, XGBoost Classifier, and XGBoost Regressor models on the training set
Gets predictions from each model on the testing set and calculates the accuracy and mean squared error of each model

training:
get a subset of the dataset
create a decision tree
repeat of as many times as the number of trees

testing:
get the predictions from each tree
Classification: hold a majority vote
Regression: get the mean of the predictions
Build a decision tree on the bootstrap sample

MSE = mean squared error
"""

data = {
    'age': [25, 35, 45, 20, 30, 40, 50, 25, 35, 45, 20, 30, 40, 50],
    'income': [40, 60, 80, 20, 40, 60, 80, 30, 50, 70, 30, 50, 70, 90],
    'buys computer': [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

X_train, X_test, y_train, y_test = train_test_split(df[['age', 'income']], df['buys computer'])
print("X_train : ", '\n', X_train)
print("X_test : ", '\n', X_test)
print("y_train : ", '\n', y_train)
print("y_test : ", '\n', y_test)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

reg = RandomForestRegressor()
reg.fit(X_train, y_train)

clf_predictions = clf.predict(X_test)
reg_predictions = reg.predict(X_test)

clf_majority_vote = clf_predictions.round().mean()
print("Classifier majority vote = ", clf_majority_vote)

reg_mean = reg_predictions.mean()
print("The mean of the predictions = ", reg_mean)

gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)

gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train, y_train)

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(X_train, y_train)

gb_clf_predictions = gb_clf.predict(X_test)
gb_reg_predictions = gb_reg.predict(X_test)
xgb_clf_predictions = xgb_clf.predict(X_test)
xgb_reg_predictions = xgb_reg.predict(X_test)

print("Gradient Boosting Classifier predictions = ", gb_clf_predictions)
print("Gradient Boosting Regressor predictions = ", gb_clf_predictions)
print("XGBoost Classifier predictions = ", gb_clf_predictions)
print("XGBoost Regressor predictions = ", gb_clf_predictions)

print("Gradient Boosting Classifier Accuracy = ", accuracy_score(y_test, gb_clf_predictions))
print("Gradient Boosting Regressor MSE = ", mean_squared_error(y_test, gb_reg_predictions))
print("XGBoost Classifier Accuracy = ", accuracy_score(y_test, xgb_clf_predictions))
print("XGBoost Regressor MSE = ", mean_squared_error(y_test, xgb_reg_predictions))
