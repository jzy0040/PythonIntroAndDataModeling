import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('bike_sharing_demand.csv')  # Update the filename if necessary

# Step 1: Overview of the dataset
print("Dataset Overview:")
print(df.info())  # Provides information on the DataFrame, including column names and data types
print("\nDescriptive Statistics:")
print(df.describe())  # Gives summary statistics for numeric columns
print("\nFirst Few Rows:")
print(df.head())  # Displays the first few rows of the dataset

# Step 2: Univariate Analysis
numeric_cols = ['temp', 'temp_feel', 'humidity', 'windspeed', 'demand']

for col in numeric_cols:
    # Distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot for outlier detection
    plt.figure(figsize=(12, 6))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

# Step 3: Correlation Analysis
correlation_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Correlation of features with 'demand'
demand_correlation = correlation_matrix['demand'].sort_values(ascending=False)
print("\nCorrelation with Demand:")
print(demand_correlation)

# Step 4: Feature Importance and Visualization
# Pairplot to visualize relationships between features and demand
sns.pairplot(df, vars=numeric_cols, hue='season', palette='viridis')
plt.show()

# Boxplot of demand vs. important features (example: temperature and humidity)
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='season', y='temp')
plt.title('Temperature by Season')
plt.xlabel('Season')
plt.ylabel('Temperature')
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='season', y='humidity')
plt.title('Humidity by Season')
plt.xlabel('Season')
plt.ylabel('Humidity')
plt.show()
