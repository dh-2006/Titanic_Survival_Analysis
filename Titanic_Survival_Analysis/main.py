
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n========== TITANIC SURVIVAL ANALYSIS ==========")

data = pd.read_csv("Titanic_Dataset.csv")

print("\nDataset Loaded Successfully!")
print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
data.info()

# ==========================================
# 2. HANDLE MISSING VALUES
# ==========================================

print("\n========== DATA CLEANING ==========")

print("\nMissing Values Before Handling:")
print(data.isnull().sum())

# Calculate missing percentage
missing_percentage = data.isnull().mean() * 100

# Drop columns having more than 50% missing values
columns_to_drop = missing_percentage[
    missing_percentage > 50
].index.tolist()

if columns_to_drop:
    data.drop(columns=columns_to_drop, inplace=True)
    print("\nDropped Columns:")
    print(columns_to_drop)

# Handle numerical missing values using median
numeric_columns = data.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    if data[col].isnull().any():
        median_value = data[col].median()

        if pd.notna(median_value):
            data[col] = data[col].fillna(median_value)
        else:
            data.drop(columns=[col], inplace=True)

# Handle categorical missing values using mode
categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_columns:
    if data[col].isnull().any():
        mode_values = data[col].mode()

        if not mode_values.empty:
            data[col] = data[col].fillna(mode_values[0])
        else:
            data.drop(columns=[col], inplace=True)

print("\nMissing Values After Handling:")
print(data.isnull().sum())

print("\nMissing Values Handled Successfully!")

# ==========================================
# 3. BASIC DATA ANALYSIS
# ==========================================

print("\n========== BASIC DATA ANALYSIS ==========")

print("\nTotal Passengers:", len(data))

print("\nSurvival Count:")
print(data["Survived"].value_counts())

print("\nSurvival Percentage:")
print(data["Survived"].value_counts(normalize=True) * 100)

print("\nGender Count:")
print(data["Sex"].value_counts())

print("\nPassenger Class Count:")
print(data["Pclass"].value_counts())

# ==========================================
# 4. NUMPY ANALYSIS
# ==========================================

print("\n========== NUMPY ANALYSIS ==========")

age_array = data["Age"].to_numpy()
fare_array = data["Fare"].to_numpy()
survival_array = data["Survived"].to_numpy()

print("\nAge Array:")
print(age_array)

print("\nFare Array:")
print(fare_array)

print("\nSurvival Array:")
print(survival_array)

print("\nAge Mean:", np.mean(age_array))
print("Age Median:", np.median(age_array))
print("Age Standard Deviation:", np.std(age_array))
print("Minimum Age:", np.min(age_array))
print("Maximum Age:", np.max(age_array))

print("\nFare Mean:", np.mean(fare_array))
print("Fare Median:", np.median(fare_array))
print("Fare Standard Deviation:", np.std(fare_array))
print("Minimum Fare:", np.min(fare_array))
print("Maximum Fare:", np.max(fare_array))

print("\nAge Percentiles:")
print(np.percentile(age_array, [25, 50, 75]))

print("\nFare Percentiles:")
print(np.percentile(fare_array, [25, 50, 75]))

print("\nTotal Survivors:", np.sum(survival_array))
print("Total Non-Survivors:", len(survival_array) - np.sum(survival_array))

print("\nAge Variance:", np.var(age_array))
print("Fare Variance:", np.var(fare_array))

# Find passengers whose age is above 18
adult_passengers = np.where(data["Age"] >= 18)

print("\nNumber of Adult Passengers:", len(adult_passengers[0]))

# ==========================================
# 5. SURVIVAL ANALYSIS
# ==========================================

print("\n========== SURVIVAL ANALYSIS ==========")

print("\nSurvival by Gender:")
print(data.groupby("Sex")["Survived"].mean() * 100)

print("\nSurvival by Passenger Class:")
print(data.groupby("Pclass")["Survived"].mean() * 100)

print("\nSurvival by Gender and Class:")
print(data.groupby(["Sex", "Pclass"])["Survived"].mean() * 100)

# ==========================================
# 6. AGE GROUP ANALYSIS
# ==========================================

print("\n========== AGE GROUP ANALYSIS ==========")

data["AgeGroup"] = pd.cut(
    data["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Young Adult", "Adult", "Senior"]
)

print("\nPassenger Count by Age Group:")
print(data["AgeGroup"].value_counts())

print("\nSurvival by Age Group:")
print(data.groupby("AgeGroup", observed=False)["Survived"].mean() * 100)

# ==========================================
# 7. VISUALIZATION
# ==========================================

print("\n========== DATA VISUALIZATION ==========")

sns.set_style("whitegrid")

# Plot 1: Survival Count
plt.figure(figsize=(7, 5))
sns.countplot(x="Survived", data=data)
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.savefig("survival_count.png")
plt.show()

# Plot 2: Survival by Gender
plt.figure(figsize=(7, 5))
sns.countplot(x="Sex", hue="Survived", data=data)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.savefig("survival_by_gender.png")
plt.show()

# Plot 3: Survival by Passenger Class
plt.figure(figsize=(7, 5))
sns.countplot(x="Pclass", hue="Survived", data=data)
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.savefig("survival_by_class.png")
plt.show()

# Plot 4: Age Distribution
plt.figure(figsize=(7, 5))
sns.histplot(data["Age"], bins=20, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("age_distribution.png")
plt.show()

# Plot 5: Survival by Age Group
plt.figure(figsize=(8, 5))
sns.countplot(x="AgeGroup", hue="Survived", data=data)
plt.title("Survival by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=15)
plt.savefig("survival_by_age_group.png")
plt.show()

# Plot 6: Fare by Passenger Class
plt.figure(figsize=(7, 5))
sns.boxplot(x="Pclass", y="Fare", data=data)
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.savefig("fare_by_class.png")
plt.show()

print("\nAll Graphs Generated Successfully!")

# ==========================================
# 8. FINAL CONCLUSION
# ==========================================

print("\n========== PROJECT CONCLUSION ==========")

print("1. Female passengers had a higher survival rate than male passengers.")
print("2. First-class passengers had better survival chances.")
print("3. Passenger age also affected survival.")
print("4. Passenger class and gender were important factors in survival.")

print("\nTitanic Survival Analysis Completed Successfully!")

