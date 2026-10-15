# 1. IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 2. LOAD DATASET
df = pd.read_csv("data/StudentPerformanceFactors.csv")
print("Dataset loaded successfully!")
print("Shape:", df.shape)

# 3. SELECT FEATURE AND TARGET
# Feature
X = df[["Hours_Studied"]]
# Target
y = df["Exam_Score"]

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. CREATE MODEL
model = LinearRegression()

# 6. TRAIN MODEL
model.fit(X_train, y_train)

# 7. MAKE PREDICTIONS
y_pred = model.predict(X_test)

# 8. MODEL COEFFICIENT AND INTERCEPT
coefficient = model.coef_[0]
intercept = model.intercept_
print("\nModel Equation:")
print(
    f"Exam Score = {intercept:.2f} + "
    f"({coefficient:.2f} × Hours Studied)"
)

# 9. EVALUATE MODEL
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# 10. SAVE MODEL
import os
import joblib
# Create the model folder if it doesn't exist
os.makedirs("model", exist_ok=True)
# Save the trained model
joblib.dump(model, "model/linear_regression_model.pkl")
print("Model saved successfully!")