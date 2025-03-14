# INFO-B211-Assignment-6


## Purpose
The purpose of this project is to predict diabetes progression using various regression models. The project utilizes the diabetes dataset from the `sklearn` library and implements three different regression models: Linear Regression, Decision Tree Regression, and Random Forest Regression. The performance of these models is evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics.

## Class Design and Implementation
This project does not use custom classes but relies on the regression models provided by the `sklearn` library. Below is an explanation of the key components and methods used in the project:

### Attributes
- `X`: Features from the diabetes dataset.
- `y`: Target values from the diabetes dataset.
- `X_train`, `X_test`: Training and testing sets for features.
- `y_train`, `y_test`: Training and testing sets for target values.
- `linear_reg`: Instance of `LinearRegression`.
- `decision_tree`: Instance of `DecisionTreeRegressor`.
- `random_forest`: Instance of `RandomForestRegressor`.

### Methods
- **Data Loading**: Loads the diabetes dataset using `load_diabetes()`.
- **Data Splitting**: Splits the dataset into training and testing sets using `train_test_split()`.
- **Model Initialization**: Initializes the regression models.
- **Model Training**: Trains the models using the training data.
- **Model Prediction**: Makes predictions using the testing data.
- **Model Evaluation**: Evaluates the models using MSE and R2 metrics.

### Limitations
- The project does not include hyperparameter tuning for the models.
- The project assumes that the dataset is clean and does not perform any data preprocessing other than splitting.

## Performance Evaluation
After evaluating the models, the following results were obtained:

- **Linear Regression**:
  - MSE: 2900.1936284934823
  - R2: 0.45260276297191926

- **Decision Tree Regression**:
  - MSE: 4976.797752808989
  - R2: 0.060653981041140725

- **Random Forest Regression**:
  - MSE: 2952.0105887640448
  - R2: 0.4428225673999313

Based on these metrics, the Linear Regression model gives the best performance with the lowest MSE of 2900.1936284934823 and a relatively high R2 score of 0.45260276297191926. This indicates that the Linear Regression model is better at capturing the variance in the data and making accurate predictions compared to the Decision Tree and Random Forest models. The Decision Tree model has the lowest performance, likely due to overfitting on the training data, while the Random Forest model performs reasonably well but not as good as the Linear Regression model.

## Conclusion
The Linear Regression model is the best performer among the three models evaluated in this project. It provides a good balance of accuracy and robustness, making it a suitable choice for predicting diabetes progression.
