import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("--- Loading Data ---")
try:
    df = pd.read_csv("car data.csv")
    print("Dataset successfully loaded!\n")
except FileNotFoundError:
    print("Error: 'car data.csv' file not found. Please check if the file is in the correct folder.")
    exit()
a
print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Basic information of the dataset:")
print(df.info(), "\n")

df.drop(['Car_Name'], axis=1, inplace=True)

df['Current_Year'] = 2026
df['Car_Age'] = df['Current_Year'] - df['Year']
df.drop(['Year', 'Current_Year'], axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

print("Dataset columns after preprocessing:")
print(df.columns, "\n")

X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Training the Model ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed successfully!\n")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model R-squared Score (Accuracy): {r2:.4f} (Approximately {r2*100:.2f}%)")
print(f"Mean Squared Error: {mse:.4f}\n")

importances = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importances:")
print(importances.sort_values(ascending=False))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Selling Prices")
plt.show()
