import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load dataset
df = pd.read_csv("data/housing.csv")

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Define categorical column(s)
categorical_features = ["location"]
categorical_transformer = OneHotEncoder()

# Build preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_features)
], remainder="passthrough")

# Define pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# Try predicting a new sample
sample = pd.DataFrame({
    "area_sqft": [1500],
    "bedrooms": [3],
    "bathrooms": [2],
    "location": ["City"]
})
predicted_price = model.predict(sample)[0]
print(f"Predicted price for {sample.to_dict(orient='records')[0]}: ₹{int(predicted_price)}")
