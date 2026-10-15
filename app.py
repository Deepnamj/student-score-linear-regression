# 1. IMPORT LIBRARIES
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)


# 3. FILE PATHS
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "linear_regression_model.pkl"
DATA_PATH = BASE_DIR / "data" / "StudentPerformanceFactors.csv"


# 4. LOAD TRAINED MODEL
model = joblib.load(MODEL_PATH)


# 5. LOAD DATASET
df = pd.read_csv(DATA_PATH)

# Keep only the columns used in this project
df = df[["Hours_Studied", "Exam_Score"]].dropna()


# 6. TITLE
st.title("🎓 Student Exam Score Predictor")

st.write(
    "Predict a student's exam score based on "
    "the number of hours they studied."
)


# 7. USER INPUT
hours_studied = st.number_input(
    "Enter Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)


# 8. PREDICTION BUTTON
if st.button("Predict Exam Score"):

    # Convert user input into DataFrame
    input_data = pd.DataFrame({
        "Hours_Studied": [hours_studied]
    })

    # Make prediction
    prediction = model.predict(input_data)

    predicted_score = prediction[0]

    # Display prediction
    st.success(
        f"Predicted Exam Score: {predicted_score:.2f}"
    )


# 9. MODEL INFORMATION
st.divider()

st.subheader("📊 Model Information")

st.write("**Algorithm:** Simple Linear Regression")
st.write("**Feature:** Hours Studied")
st.write("**Target:** Exam Score")


# 10. REGRESSION EQUATION
st.subheader("📐 Regression Equation")

coefficient = model.coef_[0]
intercept = model.intercept_

st.write(
    f"Exam Score = {intercept:.2f} + "
    f"({coefficient:.2f} × Hours Studied)"
)


# 11. INTERPRETATION
st.subheader("💡 Interpretation")

st.write(
    f"For every additional hour studied, the model predicts "
    f"an average increase of approximately "
    f"{coefficient:.2f} marks in the exam score."
)


# 12. PREPARE DATA FOR EVALUATION
X = df[["Hours_Studied"]]
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Generate predictions for test data
y_pred = model.predict(X_test)


# 13. DATASET INSIGHTS
st.divider()

st.subheader("📋 Dataset Insights")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Student Records",
    f"{len(df):,}"
)

col2.metric(
    "Average Exam Score",
    f"{df['Exam_Score'].mean():.2f}"
)

correlation = df["Hours_Studied"].corr(df["Exam_Score"])

col3.metric(
    "Correlation",
    f"{correlation:.3f}"
)


# 14. MODEL EVALUATION
st.divider()

st.subheader("📈 Model Evaluation")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R²", f"{r2:.3f}")


# ============================================================
# GRAPH 1: ACTUAL DATA + REGRESSION LINE
# ============================================================

st.divider()

st.subheader("📊 1. Hours Studied vs Exam Score")

st.write(
    "This graph shows the actual student data points and "
    "the regression line learned by the model."
)

st.info(
    "💡 Importance: This helps us visually understand the "
    "relationship between Hours Studied and Exam Score and "
    "see how well a straight line represents the data."
)

fig, ax = plt.subplots(figsize=(8, 5))

# Actual student observations
ax.scatter(
    df["Hours_Studied"],
    df["Exam_Score"],
    alpha=0.5,
    label="Actual Students"
)

# Create values for the regression line
line_data = pd.DataFrame({
    "Hours_Studied": sorted(df["Hours_Studied"].unique())
})

# Predict exam scores for the line
line_predictions = model.predict(line_data)

# Draw regression line
ax.plot(
    line_data["Hours_Studied"],
    line_predictions,
    linewidth=2,
    label="Regression Line"
)

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Exam Score")
ax.set_title("Hours Studied vs Exam Score")

ax.legend()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# GRAPH 2: ACTUAL VS PREDICTED
# ============================================================

st.divider()

st.subheader("🎯 2. Actual vs Predicted Scores")

st.write(
    "This graph compares the actual exam scores with the "
    "scores predicted by the model for the test data."
)

st.info(
    "💡 Importance: This helps us see how close the model's "
    "predictions are to the actual values. The closer the "
    "points are to the diagonal line, the better the predictions."
)

fig, ax = plt.subplots(figsize=(7, 5))

# Plot actual vs predicted values
ax.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction line
lower = min(
    float(y_test.min()),
    float(y_pred.min())
)

upper = max(
    float(y_test.max()),
    float(y_pred.max())
)

ax.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    label="Perfect Prediction"
)

ax.set_xlabel("Actual Exam Score")
ax.set_ylabel("Predicted Exam Score")
ax.set_title("Actual vs Predicted Scores")

ax.legend()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# GRAPH 3: RESIDUAL PLOT
# ============================================================

st.divider()

st.subheader("📉 3. Residual Plot")

st.write(
    "A residual is the difference between the actual value "
    "and the predicted value."
)

st.latex(
    "Residual = Actual - Predicted"
)

st.info(
    "💡 Importance: This graph helps us check whether the "
    "model's errors are randomly distributed around zero. "
    "A good regression model should generally have residuals "
    "scattered around zero without a clear pattern."
)

# Calculate residuals
residuals = y_test.to_numpy() - y_pred

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

# Zero-error reference line
ax.axhline(
    0,
    linestyle="--"
)

ax.set_xlabel("Predicted Exam Score")
ax.set_ylabel("Residual")
ax.set_title("Residuals vs Predicted Scores")

st.pyplot(fig)

plt.close(fig)


# ============================================================
# MODEL LIMITATIONS
# ============================================================

st.divider()

with st.expander("⚠️ Model Limitations"):

    st.write(
        "This project uses only Hours Studied to predict Exam Score."
    )

    st.write(
        "In reality, exam performance can also depend on "
        "attendance, previous performance, sleep, tutoring, "
        "parental involvement, learning resources, and many "
        "other factors."
    )

    st.write(
        "Therefore, this model shows the relationship between "
        "study hours and exam score. It does not prove that "
        "studying more hours directly causes a particular score."
    )