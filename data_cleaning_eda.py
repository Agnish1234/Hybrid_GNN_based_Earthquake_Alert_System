# data_cleaning_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visual style
sns.set_theme(style="whitegrid")

# Load the data we acquired in Mission 1.1
file_path = 'usgs_earthquakes_2010_2024.csv'
df = pd.read_csv(file_path)

print("=== MISSION 2: DATA CLEANING & EDA ===")
print(f"Dataset Shape: {df.shape}")

# --------------------------------------------------------------------
# TASK 1: CONVERT DATA TYPES
# --------------------------------------------------------------------
print("\n1. CONVERTING DATA TYPES...")

# Convert 'time' and 'updated' from strings to datetime objects
# This is CRITICAL for any time-based analysis
df['time'] = pd.to_datetime(df['time'])
df['updated'] = pd.to_datetime(df['updated'])

# Extract valuable time-based features for later use
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear

print("   ✓ Converted 'time' and 'updated' to datetime.")

# --------------------------------------------------------------------
# TASK 2: ANALYZE MISSING VALUES
# --------------------------------------------------------------------
print("\n2. ANALYZING MISSING VALUES...")

# Calculate the percentage of missing values for each column
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

print("Columns with missing values (%%):")
print(missing_percentage.round(2))

# --------------------------------------------------------------------
# TASK 3: DEEP EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------------
print("\n3. DEEP EXPLORATORY DATA ANALYSIS...")

# Create a figure for our visualizations
plt.figure(figsize=(20, 15))

# Subplot 1: Time Series of Earthquake Frequency
plt.subplot(3, 3, 1)
df.resample('ME', on='time')['mag'].count().plot() # Count events per month
plt.title('Earthquake Frequency Over Time (Monthly Count)')
plt.xlabel('Time')
plt.ylabel('Number of Earthquakes')

# Subplot 2: Depth vs. Magnitude Scatter Plot
plt.subplot(3, 3, 2)
plt.scatter(df['mag'], df['depth'], alpha=0.3, s=10)
plt.title('Depth vs. Magnitude')
plt.xlabel('Magnitude')
plt.ylabel('Depth (km)')
plt.gca().invert_yaxis() # Invert y-axis so depth increases downward

# Subplot 3: Boxplot of Magnitude by Network
plt.subplot(3, 3, 3)
# Take the top 5 most common networks to avoid clutter
top_nets = df['net'].value_counts().nlargest(5).index
df_top_nets = df[df['net'].isin(top_nets)]
sns.boxplot(data=df_top_nets, x='net', y='mag')
plt.title('Magnitude Distribution by Seismic Network (Top 5)')
plt.xlabel('Network Code')
plt.ylabel('Magnitude')

# Subplot 4: Distribution of Magnitude Types
plt.subplot(3, 3, 4)
df['magType'].value_counts().plot(kind='bar')
plt.title('Count of Magnitude Estimation Types')
plt.xlabel('Magnitude Type')
plt.ylabel('Count')

# Subplot 5: Global Map (We'll make this one bigger and more detailed)
plt.subplot(3, 1, 3) # This makes the map span the bottom width
plt.scatter(df['longitude'], df['latitude'], alpha=0.1, s=1, c='red')
plt.title('Global Distribution of Earthquakes (2010-2024)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('equal') # Use equal aspect ratio for proper map proportions

# Adjust layout and display
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# TASK 4: INITIAL FEATURE ASSESSMENT
# --------------------------------------------------------------------
print("\n4. FEATURE ASSESSMENT...")

# Calculate correlation matrix only for numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_features].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Specifically, how does magnitude correlate with other features?
print("\nCorrelation of features with 'mag' (magnitude):")
mag_correlations = correlation_matrix['mag'].sort_values(ascending=False)
print(mag_correlations)

# --------------------------------------------------------------------
# TASK 5: SAVE THE CLEANED DATASET
# --------------------------------------------------------------------
# We will save this for the next mission
output_file_cleaned = 'earthquakes_cleaned_2010_2024.csv'
df.to_csv(output_file_cleaned, index=False)
print(f"\n5. Saved cleaned dataset to: {output_file_cleaned}")
print("\n=== MISSION 2 COMPLETE ===")