import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import logging

# Configuring logging
logging.basicConfig(filename='etl_ml_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Implement machine learning
# Load transformed data from Snowflake to pandas DataFrame
# Replace 'your_snowflake_connection_string' with your actual Snowflake connection string
snowflake_connection_string = 'your_snowflake_connection_string'
query = 'SELECT feature1, feature2, target_column FROM ml_training_data_table'
df = pd.read_sql_query(query, snowflake_connection_string)
logging.info("Data extraction completed.")

# Separate features and target variable
X = df[['feature1', 'feature2']]
y = df['target_column']

# Create polynomial features up to degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
logging.info("Data transformation completed.")


# Using GridSearchCV to find the best hyperparameters
param_grid = {'fit_intercept': [True, False]}
model = LinearRegression()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get best model after tuning
best_model = grid_search.best_estimator_

# Train the best model on the full training data
best_model.fit(X_train, y_train)

# Analyze and visualize results
# Predict target values on test set
y_pred = best_model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the best model's coefficients and intercept
print("Best Model Coefficients:", best_model.coef_)
print("Best Model Intercept:", best_model.intercept_)
print("Best Model Mean Squared Error:", mse)
logging.info("Machine learning model trained.")

