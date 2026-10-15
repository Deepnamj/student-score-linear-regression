# Findings – Student Exam Score Prediction

## 1. Objective

The objective of this project is to predict a student's exam score based on the number of hours they studied.

## 2. Dataset

The dataset was obtained from Kaggle's Student Performance Factors dataset.

For this project, only two columns were used:

- `Hours_Studied` – independent variable
- `Exam_Score` – dependent variable

## 3. Machine Learning Algorithm

Simple Linear Regression was used because the project contains one independent variable and one continuous target variable.

The model follows:

`y = b₀ + b₁x`

Where:

- `y` = predicted exam score
- `x` = hours studied
- `b₀` = intercept
- `b₁` = coefficient

## 4. Relationship Between Variables

Exploratory data analysis was performed using a scatter plot and correlation.

The analysis shows a positive relationship between hours studied and exam score.

This means that students who study more hours generally tend to have higher predicted exam scores.

## 5. Model Evaluation

The model was evaluated using:

- MAE
- MSE
- RMSE
- R²

These metrics help determine how accurately the regression model predicts exam scores.

## 6. Coefficient Interpretation

The regression coefficient represents the expected change in exam score for one additional hour of studying.

If the coefficient is positive, the model predicts that exam score increases as study hours increase.

## 7. Limitations

This model uses only `Hours_Studied` to predict exam score.

However, student performance can also depend on:

- Attendance
- Previous scores
- Sleep
- Motivation
- Tutoring
- Parental involvement
- Access to learning resources

Therefore, hours studied alone cannot completely explain exam performance.

## 8. Conclusion

Simple Linear Regression can be used to establish a basic relationship between study hours and exam scores.

The project demonstrates the complete workflow of a basic regression problem, from loading a Kaggle dataset and performing exploratory analysis to training, evaluating, saving, and deploying the model.

The model was deployed using Streamlit so that users can enter study hours and receive a predicted exam score.