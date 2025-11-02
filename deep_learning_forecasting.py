# deep_learning_forecasting_FAST.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available for dramatically faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("=== MISSION 5: DEEP LEARNING & TEMPORAL FORECASTING (FAST) ===")

# --------------------------------------------------------------------
# TASK 1: REDEFINE THE PROBLEM FOR SEQUENCE FORECASTING
# --------------------------------------------------------------------
print("\n1. PREPARING SEQUENTIAL DATA...")

# Load the final dataset
df = pd.read_csv('earthquakes_final_model_ready.csv', parse_dates=['time'])
df = df.sort_values('time').reset_index(drop=True)  # CRITICAL: Sort by time

# For a sequence model, we need to create input sequences and target values.
# Example: Use the past 10 earthquakes to predict the magnitude of the next one.
sequence_length = 10  # This is a hyperparameter we can tune

# Select and scale features (using the same features as Mission 4)
features_to_drop = ['mag', 'energy_joules', 'log_energy', 'time', 'updated', 'id']
features = df.select_dtypes(include=[np.number]).drop(columns=features_to_drop, errors='ignore').columns
X = df[features].values
y = df['mag'].values

# Scale features for better neural network performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences
X_sequences = []
y_targets = []
for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i:i+sequence_length])  # Past 'sequence_length' events
    y_targets.append(y[i+sequence_length])              # The very next event

X_sequences = np.array(X_sequences)
y_targets = np.array(y_targets)

print(f"   Total sequences created: {X_sequences.shape[0]}")
print(f"   Sequence shape (samples, timesteps, features): {X_sequences.shape}")

# Train-Test Split (by time! We cannot shuffle time series data randomly)
split_idx = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]

# STRATEGIC SUBSETTING: Use only a 20% subset of the TRAINING data for faster execution
train_sample_size = int(0.2 * len(X_train)) # Sample from the training set only
X_train_fast = X_train[:train_sample_size]
y_train_fast = y_train[:train_sample_size]

print(f"   Using reduced dataset for fast training: {X_train_fast.shape[0]} sequences")

# Convert to PyTorch tensors and send to device
X_train_t = torch.FloatTensor(X_train_fast).to(device)
y_train_t = torch.FloatTensor(y_train_fast).to(device)
X_test_t = torch.FloatTensor(X_test).to(device) # Keep full test set for honest evaluation
y_test_t = torch.FloatTensor(y_test).to(device)

# Create DataLoaders for efficient batch processing
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) # Large batch size for speed

# --------------------------------------------------------------------
# TASK 2: BUILD AN LSTM NEURAL NETWORK
# --------------------------------------------------------------------
print("\n2. BUILDING THE LSTM MODEL...")

class EarthquakeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EarthquakeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model
input_size = X_train.shape[2]  # Number of features
hidden_size = 50    # Number of LSTM units (can be tuned)
num_layers = 2      # Number of LSTM layers (can be tuned)
output_size = 1     # Predicting a single value: magnitude

model = EarthquakeLSTM(input_size, hidden_size, num_layers, output_size).to(device)
print(f"   Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# Loss and optimizer
criterion = nn.L1Loss()  # Mean Absolute Error (same as MAE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------------------------
# TASK 3: TRAIN THE MODEL
# --------------------------------------------------------------------
print("\n3. TRAINING THE LSTM...")
num_epochs = 10  # Reduced for faster execution
train_losses = []

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# --------------------------------------------------------------------
# TASK 4: EVALUATE THE MODEL
# --------------------------------------------------------------------
print("\n4. EVALUATING THE LSTM MODEL...")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t).squeeze().cpu().numpy()
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

print(f"   LSTM Test MAE: {mae:.4f}")
print(f"   LSTM Test R²: {r2:.4f}")

# Compare to our Mission 4 baseline
print("\n=== PERFORMANCE COMPARISON ===")
print("Model                 | MAE     | R²")
print("------------------------------------")
print("Random Forest        | 0.1273  | 0.7203")
print("LSTM (Sequence Model)| {:.4f}  | {:.4f}".format(mae, r2))

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('LSTM: Predictions vs. Actual Magnitude')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.savefig('lstm_predictions_fast.png')
print("   ✓ LSTM predictions plot saved.")

print("\n=== MISSION 5 COMPLETE ===")