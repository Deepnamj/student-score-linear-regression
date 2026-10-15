# EXPLORATORY DATA ANALYSIS

# 1. IMPORT LIBRARI
import pandas as pd
import matplotlib.pyplot as plt

# 2. LOAD DATASET
df = pd.read_csv("data/StudentPerformanceFactors.csv")

# 3. BASIC INFORMATION
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 4. MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())

# 5. DUPLICATES
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 6. CORRELATION
correlation = df["Hours_Studied"].corr(
    df["Exam_Score"]
)
print("\nCorrelation:")
print(correlation)

# 7. SCATTER PLOT
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Hours_Studied"],
    df["Exam_Score"]
)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title(
    "Hours Studied vs Exam Score"
)
plt.show()