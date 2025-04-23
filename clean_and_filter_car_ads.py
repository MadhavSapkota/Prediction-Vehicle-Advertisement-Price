"""
Author: Madhav Sapkota

This script loads a dataset of car advertisements from an RData file and prepares it for analysis.
We start by exploring the data, then filter it to only include specific car models, fuel types, and body types.
Next, we narrow down to the top 5 car colors and keep only a few important columns (3 numeric and 3 categorical).
We clean the data by converting types and removing missing values, then ensure we have exactly 2527 entries.
Finally, we save the cleaned data to a CSV and display some quick stats and insights about it.
"""

# Importing all the necessary tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pyreadr 



# ----------------------------------------------
# DATA PREPARATION
# ----------------------------------------------

#Step 1: Load RData file.  The data is saved in an R-compatible format, so we're using pyreadr to read it.
read_result = pyreadr.read_r('car_ads_fp.RData')
car_ads_df = list(read_result.values())[0]  

# Step 2: First exploration
print(f"Original dataset has {car_ads_df.shape[0]} rows and {car_ads_df.shape[1]} columns.")
print(f"Here's what the columns are called: {car_ads_df.columns.tolist()}")
print("\nAnd here’s the first few rows so we know what we’re dealing with:")
print(car_ads_df.head())

# Step 3: Filter the data by applicable models, body kinds, and fuel types.
selected_models = ["C3", "DS3", "Grande Punto", "Panda"]
car_ads_df = car_ads_df[car_ads_df['Genmodel'].isin(selected_models)]
car_ads_df = car_ads_df[car_ads_df['Bodytype'] == 'Hatchback']
car_ads_df = car_ads_df[car_ads_df['Fuel_type'].isin(['Petrol', 'Diesel'])]

# Step 4: Limit to the top 5 car colors.
top_5_colors = car_ads_df['Color'].value_counts().nlargest(5).index.tolist()
print(f"\nThe 5 most popular car colors are: {top_5_colors}")
car_ads_df = car_ads_df[car_ads_df['Color'].isin(top_5_colors)]

# Step 5: Choose features for analysis (3 numeric, 3 categorical).
numeric_features = ['Adv_year', 'Adv_month', 'Price']
categorical_features = ['Genmodel', 'Fuel_type', 'Color']
df_filtered = car_ads_df[numeric_features + categorical_features]

# ----------------------------------------------
# DATA CLEANING
# ----------------------------------------------

# Step 6: Convert types and address missing values
for column in numeric_features:
    df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce')
for column in categorical_features:
    df_filtered[column] = df_filtered[column].astype('object')
df_filtered = df_filtered.dropna()

# Step 7: Ensure the required row count.
current_count = len(df_filtered)
print(f"\nAfter cleaning, we have {current_count} rows left.")
if current_count > 2527:
    df_filtered = df_filtered.sample(n=2527, random_state=42)
elif current_count < 2527:
    print("⚠️ Not enough rows — consider relaxing your filters.")

# Step 8: Check the dataset structure.
print("\nHere’s what our final dataset looks like:")
print(f"Shape: {df_filtered.shape}")
print(df_filtered.dtypes)
num_numeric = sum(df_filtered.dtypes == 'float64') + sum(df_filtered.dtypes == 'int64')
num_categorical = sum(df_filtered.dtypes == 'object')
print(f"\n✅ Numeric columns: {num_numeric}")
print(f"✅ Categorical columns: {num_categorical}")
if num_numeric == 3 and num_categorical == 3 and df_filtered.shape[0] == 2527:
    print("\n🎉 Dataset is ready for analysis.")
else:
    print("\n⚠️ Please double-check data selection or formatting.")

# ----------------------------------------------
# PRELIMINARY ANALYSIS
# ----------------------------------------------

# Step 9: Save as CSV for future usage.
df_filtered.to_csv('cleaned_car_data.csv', index=False)
print("\nSaved the cleaned dataset to 'cleaned_car_data.csv'.")

#Step 10: Quick metrics and frequency checks.
print("\nSummary statistics for numeric features:")
print(df_filtered[numeric_features].describe())

print("\nFrequency of values in categorical features:")
for column in categorical_features:
    print(f"\n{column} value counts:")
    print(df_filtered[column].value_counts())


#References:

"""
https://pandas.pydata.org

https://numpy.org

https://matplotlib.org

https://www.statsmodels.org

https://github.com/ofajardo/pyreadr

"""