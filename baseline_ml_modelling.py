# baseline_ml_modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the final, model-ready dataset from Mission 3
df = pd.read_csv('earthquakes_final_model_ready.csv')
print("=== MISSION 4: BASELINE MACHINE LEARNING MODELING ===")
print(f"Dataset Shape: {df.shape}")

# --------------------------------------------------------------------
# TASK 1: DEFINE THE PROBLEM & PREPARE THE DATA
# --------------------------------------------------------------------
print("\n1. DEFINING THE PREDICTION TASK...")

# We are performing Regression: predicting a continuous value ('mag').
target = 'mag'
y = df[target]  # This is what we want to predict

# Our features (X) will be all numerical columns, except the target and its derivatives.
# We drop columns that are direct proxies for the target or are not features.
features_to_drop = [target, 'energy_joules', 'log_energy', 'time', 'updated', 'id']
X = df.select_dtypes(include=[np.number]).drop(columns=features_to_drop, errors='ignore')

print(f"   Target variable: {target}")
print(f"   Number of features: {X.shape[1]}")
print(f"   Feature names: {list(X.columns)}")

# Split the data into training and testing sets (80% train, 20% test)
# The 'random_state' ensures we get the same split every time for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training set: {X_train.shape}, Testing set: {X_test.shape}")

# --------------------------------------------------------------------
# TASK 2: TRAIN BASELINE MODELS
# --------------------------------------------------------------------
print("\n2. TRAINING BASELINE MODELS...")

# Initialize the models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Dictionary to store our results
results = {}

for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train, y_train) # Train the model on the training data

    # Make predictions on the test set (data the model has never seen)
    y_pred = model.predict(X_test)

    # Calculate and store performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'R2': r2}

    print(f"     - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"     - R² Score: {r2:.4f}")

# --------------------------------------------------------------------
# TASK 3: ANALYZE MODEL PERFORMANCE & IMPORTANCE
# --------------------------------------------------------------------
print("\n3. ANALYZING MODEL PERFORMANCE...")

# Create a DataFrame to neatly compare results
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

# Plot feature importance from the Random Forest model
# This tells us which features the model found most useful for prediction.
model_rf = models["Random Forest"]
feature_importances = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
feature_importances.head(15).plot(kind='barh') # Plot top 15 most important features
plt.title('Top 15 Feature Importances (Random Forest Model)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("   ✓ Feature importance plot saved as 'feature_importance.png'.")

# --------------------------------------------------------------------
# TASK 4: VISUALIZE PREDICTIONS vs. ACTUAL VALUES
# --------------------------------------------------------------------
# Let's see how well the best model's predictions align with reality
best_model_name = results_df['R2'].idxmax() # Get the name of the model with the highest R²
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
plt.title(f'Predictions vs. Actual Values ({best_model_name})')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
print("   ✓ Predictions vs. Actual plot saved.")

print("\n=== MISSION 4 COMPLETE ===")
print("Baseline performance established. Ready for advanced modeling in Mission 5.")