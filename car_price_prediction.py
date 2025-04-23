
"""
Author: Madhav Sapkota

This script reads a dataset of vehicle commercials from an RData file and prepares it for analysis.
We begin by looking through the data, then filter it to just include specific automobile models, fuel types, and body kinds.
Next, we reduce down to the top five car colors, retaining only a few key columns (3 numerical and 3 categories).
We clean the data by changing types and deleting missing values, and then verify that we have exactly 2527 items.
Finally, we save the cleaned data to a CSV file and show some quick statistics and insights about it.

"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap



# ───────────────────────────────────────────────
# 1️⃣ PREPARE THE DATA FOR ANALYSIS
# ───────────────────────────────────────────────



# Load the cleansed car data.
car_data = pd.read_csv('cleaned_car_data.csv')

# Let's briefly check the data types to ensure everything is in the correct format.
print("Checking data types before processing:")
print(car_data.dtypes)

# Check if there are any missing values in the dataset.
print("\nChecking for missing values:")
print(car_data.isnull().sum())


# We must convert the category input into numbers so that the machine learning model can understand it.
# This phase converts categories like 'Car Model', 'Fuel Type', and 'Color' to binary columns (0 or 1).
car_model_dummies = pd.get_dummies(car_data['Genmodel'], prefix='CarModel', drop_first=True)
fuel_type_dummies = pd.get_dummies(car_data['Fuel_type'], prefix='FuelType', drop_first=True)
color_dummies = pd.get_dummies(car_data['Color'], prefix='Color', drop_first=True)


# Now, let's choose only the numeric columns we'll utilize to anticipate automobile costs.
numeric_columns = car_data[['Adv_year', 'Adv_month']]


# Combine all of the features (numeric and categorical) into one large table of data.
# We will use this to predict car costs.
features = pd.concat([numeric_columns, car_model_dummies, fuel_type_dummies, color_dummies], axis=1)
features = features.astype(float)  


# To anticipate the car's price
target_price = car_data['Price'].astype(float)



# ───────────────────────────────────────────────
# 2️⃣ ANALYZE THE DATA
# ───────────────────────────────────────────────


# Divide the data into two parts: one for training the model (80%), and one for testing (20%).
# This manner, we can test how well the model performs with data it hasn't seen before.
X_train, X_test, y_train, y_test = train_test_split(features, target_price, test_size=0.2, random_state=42)


# We'll use a Random Forest model to predict car prices.
# Random Forest is a common algorithm that works well with a variety of data.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model with the training data.
rf_model.fit(X_train, y_train)

# Now, let's utilize the trained model to estimate automobile prices using the test data.
y_pred = rf_model.predict(X_test)



# Evaluate the model by calculating how closely the forecasts match the real automobile costs.
# We employ Mean Squared Error (MSE) and R-squared (R2) for evaluation.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model's performance
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")



# ───────────────────────────────────────────────
# 3️⃣ INTERPRET THE RESULTS
# ───────────────────────────────────────────────

# Let's see which qualities were most important in predicting car costs.
feature_importances = rf_model.feature_importances_
feature_names = features.columns

# Make a table that displays each feature and how important it was in forecasting price.
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Print out the top 10 most important features
print("\nMost Important Features for Predicting Car Prices:")
print(importance_df.head(10))


# Let's create a plot of the top ten elements that influenced car price estimations the most.
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Features Influencing Car Price Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()


plt.savefig('random_forest_feature_importance.png')
print("\nFeature importance plot saved as 'random_forest_feature_importance.png'")



#Refereces:
"""
McKinney, W. (2010). Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference (pp. 51–56).
https://pandas.pydata.org

Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585, 357–362.
https://numpy.org


Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95.
https://matplotlib.org

Waskom, M. L. (2021). Seaborn: Statistical data visualization. Journal of Open Source Software, 6(60), 3021.
https://seaborn.pydata.org


Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
https://scikit-learn.org

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems (pp. 4765–4774).
https://github.com/slundberg/shap

"""