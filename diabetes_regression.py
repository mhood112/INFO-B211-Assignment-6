import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
except Exception as e:
    print("Error loading dataset:", str(e))
    exit(1)

try:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    print("Error splitting data:", str(e))
    exit(1)

try:
    # Initialize the models
    linear_reg = LinearRegression()
    decision_tree = DecisionTreeRegressor(random_state=42)
    random_forest = RandomForestRegressor(random_state=42)
except Exception as e:
    print("Error initializing models:", str(e))
    exit(1)

try:
    # Train the models
    linear_reg.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
except Exception as e:
    print("Error training models:", str(e))
    exit(1)

try:
    # Make predictions
    y_pred_linear = linear_reg.predict(X_test)
    y_pred_tree = decision_tree.predict(X_test)
    y_pred_forest = random_forest.predict(X_test)
except Exception as e:
    print("Error making predictions:", str(e))
    exit(1)

try:
    # Evaluate the models
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    mse_tree = mean_squared_error(y_test, y_pred_tree)
    r2_tree = r2_score(y_test, y_pred_tree)

    mse_forest = mean_squared_error(y_test, y_pred_forest)
    r2_forest = r2_score(y_test, y_pred_forest)

    # Print the results
    print("Linear Regression - MSE:", mse_linear, "R2:", r2_linear)
    print("Decision Tree Regression - MSE:", mse_tree, "R2:", r2_tree)
    print("Random Forest Regression - MSE:", mse_forest, "R2:", r2_forest)
except Exception as e:
    print("Error evaluating models:", str(e))
    exit(1)
