import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# STEP 1: Loading and Inspecting the Data
# ==========================================
print("--- Loading Data ---")
try:
    df = pd.read_csv("car data.csv")
    print("Dataset successfully loaded!\n")
except FileNotFoundError:
    print("Error: 'car data.csv' file not found. Please check if the file is in the correct folder.")
    exit()

# Inspecting basic details of the data
print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Basic information of the dataset:")
print(df.info(), "\n")

# ==========================================
# STEP 2: Data Preprocessing
# ==========================================
# Dropping 'Car_Name' as it won't significantly help in pricing prediction
df.drop(['Car_Name'], axis=1, inplace=True)

# Calculating the age of the car (Assuming current year is 2026)
df['Current_Year'] = 2026
df['Car_Age'] = df['Current_Year'] - df['Year']
df.drop(['Year', 'Current_Year'], axis=1, inplace=True)

# Converting categorical text columns to numbers using One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

print("Dataset columns after preprocessing:")
print(df.columns, "\n")

# ==========================================
# STEP 3: Features and Target Split
# ==========================================
# X = input features, y = target variable (Selling_Price)
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Splitting data into Train and Test sets (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: Model Training
# ==========================================
print("--- Training the Model ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed successfully!\n")

# ==========================================
# STEP 5: Model Evaluation
# ==========================================
y_pred = model.predict(X_test)

# Checking model performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model R-squared Score (Accuracy): {r2:.4f} (Approximately {r2*100:.2f}%)")
print(f"Mean Squared Error: {mse:.4f}\n")

# Checking feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importances:")
print(importances.sort_values(ascending=False))

# ==========================================
# STEP 6: Visualization
# ==========================================
# Plotting Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Selling Prices")
plt.show()