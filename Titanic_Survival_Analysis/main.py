
# Titanic Survival Analysis
# Dataset: Kaggle - Titanic: Machine Learning from Disaster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Kaggle training dataset
data = pd.read_csv("Titanic_Dataset.csv")

print("\n========== TITANIC SURVIVAL ANALYSIS ==========")

# Basic Information
print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())

# Data Cleaning
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

print("\nMissing Values After Cleaning:")
print(data[["Age", "Embarked"]].isnull().sum())

# NumPy - Numerical Analysis
print("\n========== NUMPY ANALYSIS ==========")

ages = np.array(data["Age"])
fares = np.array(data["Fare"])
survival = np.array(data["Survived"])

print("Average Age:", np.mean(ages))
print("Median Age:", np.median(ages))
print("Standard Deviation of Age:", np.std(ages))

print("Minimum Fare:", np.min(fares))
print("Maximum Fare:", np.max(fares))
print("Average Fare:", np.mean(fares))

print("25th Percentile of Fare:", np.percentile(fares, 25))
print("75th Percentile of Fare:", np.percentile(fares, 75))

# NumPy Array Operations
survived_array = np.where(survival == 1, "Survived", "Not Survived")

print("\nFirst 10 Survival Results:")
print(survived_array[:10])

print("\nTotal Survived:", np.sum(survival))
print("Total Not Survived:", np.sum(survival == 0))

# Basic Analysis
total_passengers = len(data)
survived = np.sum(survival)
not_survived = total_passengers - survived

survival_rate = np.mean(survival) * 100

print("\n========== SUMMARY ==========")
print("Total Passengers:", total_passengers)
print("Survived Passengers:", survived)
print("Not Survived Passengers:", not_survived)
print("Overall Survival Rate: {:.2f}%".format(survival_rate))

print("\nSurvival Count:")
print(data["Survived"].value_counts().rename(
    index={0: "Not Survived", 1: "Survived"}
))

print("\nSurvival by Gender:")
print(data.groupby("Sex")["Survived"].agg(["count", "sum", "mean"]))

print("\nSurvival by Passenger Class:")
print(data.groupby("Pclass")["Survived"].agg(["count", "sum", "mean"]))

# Create Age Groups
data["AgeGroup"] = pd.cut(
    data["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Adult", "Middle Age", "Senior"]
)

print("\nSurvival by Age Group:")
print(data.groupby("AgeGroup", observed=False)["Survived"].mean() * 100)

# Additional NumPy Analysis
print("\n========== ADDITIONAL NUMPY STATISTICS ==========")

print("Variance of Age:", np.var(ages))
print("Average Survival Value:", np.mean(survival))
print("Total Fare:", np.sum(fares))

# Visualizations
sns.set_theme(style="whitegrid")

# 1. Survival Count
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Survived")
plt.title("Titanic Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("1_survival_count.png")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Sex", hue="Survived")
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("2_survival_by_gender.png")
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Pclass", hue="Survived")
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("3_survival_by_class.png")
plt.show()

# 4. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="Age", bins=25, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("4_age_distribution.png")
plt.show()

# 5. Survival by Age Group
plt.figure(figsize=(8, 5))
sns.barplot(data=data, x="AgeGroup", y="Survived", errorbar=None)
plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("5_survival_by_age_group.png")
plt.show()

# 6. Fare and Passenger Class
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="Pclass", y="Fare", hue="Pclass", legend=False)
plt.title("Ticket Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("6_fare_by_class.png")
plt.show()

print("\nAll Charts Have Been Saved in the Project Folder.")
print("Project Completed Successfully!")
