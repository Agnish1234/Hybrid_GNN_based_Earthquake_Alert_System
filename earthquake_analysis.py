# earthquake_analysis.py
# Step 1: Import the essential libraries. These are the non-negotiable tools of the trade.
import pandas as pd  # For data manipulation and analysis. Our primary tool.
import numpy as np   # For numerical operations. Pandas is built on top of it.
import matplotlib.pyplot as plt # For basic visualizations.

# We will use Seaborn for more stylish plots, but it's optional.
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")  # This makes plots look better
    print("Seaborn imported. Nice-looking graphs enabled.")
except ImportError:
    print("Seaborn not found. Using plain Matplotlib.")
    pass

# Step 2: Load the data from the CSV file you just downloaded.
# Replace 'path/to/your/query.csv' with the actual path to your file.
# Pro Tip: Just put the 'query.csv' file in the same folder as this script and use './query.csv'
file_path = './query.csv'

# Read the CSV file into a Pandas DataFrame.
# A DataFrame is a 2-dimensional labeled data structure, like a spreadsheet or a SQL table.
df = pd.read_csv(file_path)

# Step 3: First Look - The Explorer's Glance
# This command shows you the first 5 rows of the DataFrame.
# It's your first look at the data's structure: what columns are there?
print("=== FIRST 5 ROWS OF THE DATASET ===")
print(df.head())
print("\n") # Print a new line for readability.

# This command gives you technical information about the DataFrame.
# It shows the number of rows/columns, column names, and their data types.
# Crucially, it shows how many non-null values are in each column, which hints at missing data.
print("=== DATASET INFO ===")
df.info()
print("\n")

# This command provides descriptive statistics for numerical columns.
# Count, mean, standard deviation, min, max, and quartiles.
# This is your first quantitative summary of the data.
print("=== DESCRIPTIVE STATISTICS ===")
print(df.describe())
print("\n")

# Step 4: Initial Visualization - The Big Picture
# Create a scatter plot to see every earthquake on a map of the world.
plt.figure(figsize=(12, 6)) # Set the figure size (width, height in inches)

# Plot longitude on the x-axis and latitude on the y-axis.
# The 'alpha' parameter makes the points slightly transparent so you can see dense clusters.
plt.scatter(df['longitude'], df['latitude'], alpha=0.3, s=5, c='red')
plt.title('Global Distribution of Earthquakes (M4.5+) since 1970')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Create a histogram of earthquake magnitudes.
plt.figure(figsize=(10, 5))
plt.hist(df['mag'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency (Count)')
plt.show()