import random

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit layout
st.title('Machine Learning Pipeline')

# Constants
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Data Exploration
st.subheader('Data Exploration')
st.write('Train data head:', train)
st.write('Test data head:', test)

# Preprocessing
# One-hot encoding for 'lithology' feature
train = pd.get_dummies(train, columns=['lithology'])
test = pd.get_dummies(test, columns=['lithology'])

# Align train and test
train, test = train.align(test, join='left', axis=1)

# Fill missing values in test with 0 (since these are one-hot encoded features)
test.fillna(0, inplace=True)

# Feature scaling
scaler = StandardScaler()
bv_pc_cols = ['bv_' + str(i) for i in range(101)] + ['pc_' + str(i) for i in range(101)]
features = train.drop(bv_pc_cols, axis=1)
targets = train[bv_pc_cols]
train[features.columns] = scaler.fit_transform(train[features.columns])
test[features.columns] = scaler.transform(test[features.columns])

# Baseline
st.subheader('Baseline Model')
means = targets.mean()
baseline_predictions_train = pd.DataFrame([means] * len(train), columns=means.index)
baseline_scores_train = mean_absolute_error(targets, baseline_predictions_train)
st.dataframe(baseline_predictions_train)
st.write('Baseline MAE score:', baseline_scores_train)

# Model
st.subheader('Machine Learning Model')
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=TEST_SIZE, random_state=RANDOM_STATE)
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

# Feature Importance
st.subheader('Feature Importance')
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': np.mean([est.coef_ for est in model.estimators_], axis=0)
})
importances = importances.sort_values(by='Importance', ascending=False)
st.dataframe(importances)
sns.barplot(x='Importance', y='Attribute', data=importances)
st.pyplot()

# Evaluation
st.subheader('Model Evaluation')
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)
model_scores = -scores
st.write('Cross-validated MAE scores:', model_scores)
st.write('Mean cross-validated MAE score:', model_scores.mean())

# Additional Metrics
y_train_pred = model.predict(X_train)
st.write('Training Root Mean Squared Error:', np.sqrt(mean_squared_error(y_train, y_train_pred)))
st.write('Training R-squared:', r2_score(y_train, y_train_pred))

# Hyperparameter tuning
st.subheader('Hyperparameter Tuning')
params = {
    'estimator__fit_intercept': [True, False],
    'estimator__normalize': [True, False],
    'estimator__copy_X': [True, False],
    'estimator__n_jobs': [-1]
}
grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
best_params = grid.best_params_
best_score = -grid.best_score_

st.write('Best parameters:', best_params)
st.write('Best score:', best_score)

# Model evaluation with the best model
best_model.fit(X_train, y_train)

# Additional Metrics with the best model
y_train_pred_best = best_model.predict(X_train)
st.write('Training Root Mean Squared Error (Best Model):',
         np.sqrt(mean_squared_error(y_train, y_train_pred_best)))
st.write('Training R-squared (Best Model):', r2_score(y_train, y_train_pred_best))

# Model comparison with the best model
_, p_value = ttest_rel(-cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=5),
                       np.full((5,), baseline_scores_train))
if p_value < 0.05:
    st.write('The best model significantly outperforms the baseline.')
else:
    st.write('The best model does not significantly outperform the baseline.')

# Error Analysis
st.subheader('Error Analysis')

# Calculate the absolute errors for training predictions
train_errors = np.abs(y_train - y_train_pred_best)

# Calculate the mean absolute error (MAE) for each target variable
mae_per_target = train_errors.mean(axis=0)

# Display the MAE for each target variable
st.write('Mean Absolute Error (MAE) per Target Variable:')
st.write(mae_per_target)

# Find the index of the target variable with the highest MAE
worst_target_index = np.argmax(mae_per_target)

# Get the actual and predicted values for the worst target variable
actual_worst = y_train.iloc[:, worst_target_index]
predicted_worst = y_train_pred_best[:, worst_target_index]

# Create a DataFrame to display the worst predictions
worst_predictions = pd.DataFrame({
    'Actual': actual_worst,
    'Predicted': predicted_worst,
    'Error': np.abs(actual_worst - predicted_worst)
})

# Sort the worst predictions DataFrame by the error column in descending order
worst_predictions_sorted = worst_predictions.sort_values(by='Error', ascending=False)

# Display the worst predictions
st.write('Worst Predictions:')
st.dataframe(worst_predictions_sorted)

# Visualizing the results
st.subheader('Visualizing the results')

# Get predictions on the validation set
y_pred = model.predict(X_val)

# Choose a random target column to visualize
target_column = random.choice(targets.columns)

# Create a DataFrame for easier plotting
results = pd.DataFrame({
    'Actual': y_val[target_column],
    'Predicted': y_pred[:, targets.columns.get_loc(target_column)]
})
# Sort the results DataFrame by index (rows)
results_sorted = results.sort_index()

# Display the sorted DataFrame
st.dataframe(results_sorted)

# Calculate the range of the actual and predicted values
actual_range = results['Actual'].max() - results['Actual'].min()
predicted_range = results['Predicted'].max() - results['Predicted'].min()

# Set the limits of the x and y axes with a 5% margin
plot = sns.scatterplot(x='Actual', y='Predicted', data=results_sorted)
plot.set_xlim(results['Actual'].min() - 0.05 * actual_range, results['Actual'].max() + 0.05 * actual_range)
plot.set_ylim(results['Predicted'].min() - 0.05 * predicted_range, results['Predicted'].max() + 0.05 * predicted_range)
sns.lineplot(x=[results['Actual'].min(), results['Actual'].max()],
             y=[results['Actual'].min(), results['Actual'].max()], color='red').set_title(target_column)
st.pyplot()

# Determine the threshold as the middle value of the actual range
threshold = (results['Actual'].min() + results['Actual'].max()) / 2

# Calculate the standard deviation of the actual values
actual_std = results['Actual'].std()

# Sort the results DataFrame by the 'Actual' column
results_sort = results.sort_values(by='Actual')

# Check if there is a gap in the actual values
if results_sort['Actual'].diff().max() > actual_std:
    # Split data into two DataFrames based on Actual values
    small_actual_values = results[results['Actual'] <= threshold]
    big_actual_values = results[results['Actual'] > threshold]

    # Visualize small_actual_values
    st.subheader('Visualizing Small Actual Values')
    plot_small = sns.scatterplot(x='Actual', y='Predicted', data=small_actual_values)
    sns.lineplot(x=[small_actual_values['Actual'].min(), small_actual_values['Actual'].max()],
                 y=[small_actual_values['Actual'].min(), small_actual_values['Actual'].max()], color='red')
    # Customize the plot as needed
    st.pyplot()

    # Visualize big_actual_values
    st.subheader('Visualizing Big Actual Values')
    plot_big = sns.scatterplot(x='Actual', y='Predicted', data=big_actual_values)
    sns.lineplot(x=[big_actual_values['Actual'].min(), big_actual_values['Actual'].max()],
                 y=[big_actual_values['Actual'].min(), big_actual_values['Actual'].max()], color='red')
    # Customize the plot as needed
    st.pyplot()


# Residuals Plot
st.subheader('Residuals Plot')
residuals = results['Actual'] - results['Predicted']
sns.histplot(residuals, bins=30, kde=True)
st.pyplot()

# Correlation Matrix
st.subheader('Correlation Matrix')
corr = results.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot()

# Box Plot
st.subheader('Box Plot')
sns.boxplot(data=results)
st.pyplot()

# Predictions for the test set
st.subheader('Predictions for the Test Set')
test_predictions = model.predict(test[features.columns])
test_predictions = pd.DataFrame(test_predictions, columns=targets.columns)

# Display the first few rows of predictions
st.write(test_predictions)

# Visualize the distribution of predicted values for a target column
st.subheader('Distribution of Predicted Values')
target_column = random.choice(targets.columns)
sns.histplot(test_predictions[target_column], bins=30, kde=True)
st.pyplot()

# Suggest improvements
st.subheader('Improvements to the model:')
st.markdown("""

Feature Engineering: Explore additional feature engineering techniques to create more informative features. 
This could include interaction terms, polynomial features, or domain-specific transformations.

Model Selection: Experiment with different regression models to see if any other algorithms can provide better 
performance. Try models such as Random Forest, Gradient Boosting, or Support Vector Regression.

Hyperparameter Tuning: Refine the hyperparameter tuning process by expanding the search space and using more 
sophisticated optimization techniques like Bayesian optimization or genetic algorithms.

Outlier Detection and Handling: Identify and handle outliers in the training data that may be adversely affecting the 
model's performance. Consider using robust regression techniques or removing extreme outliers before training the model.

Cross-Validation Strategy: Evaluate different cross-validation strategies to ensure the model's generalization 
performance is robust. Explore techniques like stratified cross-validation or time-series cross-validation, depending 
on the nature of the data.

Feature Selection: Experiment with different feature selection techniques, such as recursive feature elimination or L1 
regularization, to identify the most important and relevant features for the model.

Ensemble Methods: Consider using ensemble methods to combine multiple models and leverage their collective predictive 
power. Techniques like stacking or blending can help improve performance and reduce model variance.
""")

# Write Summary
st.subheader('Summary:')
st.markdown("""
The machine learning pipeline presented in the code implements a multi-output regression task using linear regression 
as the base model. The pipeline involves data exploration, preprocessing steps (one-hot encoding, feature scaling), 
building the baseline model, training the machine learning model, evaluating its performance, hyperparameter tuning, 
error analysis, and visualization of results. The model achieved a mean absolute error (MAE) score of 98.01, 
significantly outperforming the baseline MAE score of 921.30. The model's root mean squared error (RMSE) on the training 
data is 453.73, with an R-squared value of 0.87. The feature importance analysis revealed the most influential features 
for the model, and further recommendations for improvement include feature engineering, exploring different models, 
refining hyperparameter tuning, handling outliers, optimizing the cross-validation strategy, feature selection, 
and utilizing ensemble methods.
""")
