import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("house_price.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Remove duplicates
df = df.drop_duplicates()

# Remove rows where target is missing
df = df.dropna(subset=["totalprice"])

# Features and target
X = df[
    [
        "bhk",
        "propertytype",
        "location",
        "sqft",
        "pricepersqft"
    ]
]

y = df["totalprice"]


# Numerical features
numeric_features = [
    "bhk",
    "sqft",
    "pricepersqft"
]

# Categorical features
categorical_features = [
    "propertytype",
    "location"
]


# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# Linear Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train model
print("\nTraining model...")
model.fit(X_train, y_train)

print("Model trained successfully!")


# Predictions
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# Display results
print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

print("==============================")


# Save model
joblib.dump(model, "house_price_model.pkl")

print("\nModel saved as house_price_model.pkl")


# Save metrics
with open("metrics.txt", "w") as file:
    file.write("HOUSE PRICE PREDICTION MODEL\n")
    file.write("============================\n")
    file.write(f"MAE  : {mae:.2f}\n")
    file.write(f"MSE  : {mse:.2f}\n")
    file.write(f"RMSE : {rmse:.2f}\n")
    file.write(f"R2   : {r2:.4f}\n")

print("Metrics saved as metrics.txt")