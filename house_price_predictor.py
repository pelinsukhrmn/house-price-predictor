# House Price Predictor using Linear Regression
# After doing the Iris classifier last week I wanted to try a regression problem
# This time I'm predicting house prices based on features like size, rooms, etc.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- Generate a simple dataset ---
# I couldn't find a small clean CSV so I'm generating fake data for now
# (plan to use a real dataset later)
np.random.seed(42)
n_samples = 200

size_sqft = np.random.randint(500, 4000, n_samples)
num_bedrooms = np.random.randint(1, 6, n_samples)
num_bathrooms = np.random.randint(1, 4, n_samples)
house_age = np.random.randint(1, 50, n_samples)

# price formula: roughly based on size + rooms, with some noise
price = (
    size_sqft * 150
    + num_bedrooms * 8000
    + num_bathrooms * 5000
    - house_age * 1000
    + np.random.randint(-20000, 20000, n_samples)
)

data = pd.DataFrame({
    "size_sqft": size_sqft,
    "num_bedrooms": num_bedrooms,
    "num_bathrooms": num_bathrooms,
    "house_age": house_age,
    "price": price,
})

print("=== House Price Predictor ===\n")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nBasic statistics:")
print(data.describe())

# --- Prepare features and target ---
X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (I read this helps linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train the model ---
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Results ===")
print(f"RMSE : ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# feature importance (coefficients)
print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:<20} {coef:>10.2f}")

# --- Predict a sample house ---
sample = pd.DataFrame([{
    "size_sqft": 1800,
    "num_bedrooms": 3,
    "num_bathrooms": 2,
    "house_age": 10,
}])
sample_scaled = scaler.transform(sample)
predicted_price = model.predict(sample_scaled)[0]

print(f"\nSample House Prediction:")
print(f"  Size: 1800 sqft | 3 bed | 2 bath | 10 years old")
print(f"  Predicted Price: ${predicted_price:,.2f}")
