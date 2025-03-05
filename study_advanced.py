import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file paths
data_path = r"C:\\Users\\cavus\\Desktop\\Forecasting Paper"
results_path = os.path.join(data_path, "Results")

# Load datasets
print("Loading datasets...")
file_west_midlands = os.path.join(data_path, "Survey_West_Midlands.xlsx")
file_newcastle = os.path.join(data_path, "Survey_Newcastle.xlsx")

west_midlands_data = pd.read_excel(file_west_midlands)
newcastle_data = pd.read_excel(file_newcastle)

# Combine datasets
data = pd.concat([west_midlands_data, newcastle_data], ignore_index=True)

# Select relevant features for forecasting
features = [
    "What type of EV do you own?",
    "What is the approximate all-electric range of your EV?",
    "How often do you use public charging stations for your EV?",
    "What is your daily average travel distance?",
    "How often in one week do you charge your vehicle?",
    "What is the average duration of your charge?",
    "How long are you willing to travel to get to a charging station?"
]

data = data[features]

# Preprocess categorical data using encoding
data = pd.get_dummies(data, drop_first=True)

# Normalize numerical features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    data_scaled[:, :-1], data_scaled[:, -1], test_size=0.2, random_state=42
)

# Define Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM": SVR(kernel='rbf'),
    "Linear Regression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

# Train and Predict with Traditional Models
y_preds = {}
metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_preds[name] = model.predict(X_test)
    
    # Evaluate models
    mae = mean_absolute_error(y_test, y_preds[name])
    mse = mean_squared_error(y_test, y_preds[name])
    r2 = r2_score(y_test, y_preds[name])
    
    metrics[name] = {"MAE": mae, "MSE": mse, "R2": r2}

# Train LSTM Model
lstm_model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
y_preds["LSTM"] = lstm_model.predict(X_test).flatten()

# Train GRU Model
gru_model = Sequential([
    GRU(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    GRU(100, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
gru_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
y_preds["GRU"] = gru_model.predict(X_test).flatten()

# Train CNN Model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1)
])
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
cnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
y_preds["CNN"] = cnn_model.predict(X_test).flatten()

# Generate Hybrid Model Predictions
y_preds["LSTM + SVM"] = (y_preds["LSTM"] + y_preds["SVM"]) / 2
y_preds["LSTM + Random Forest"] = (y_preds["LSTM"] + y_preds["Random Forest"]) / 2
y_preds["LSTM + XGBoost"] = (y_preds["LSTM"] + y_preds["XGBoost"]) / 2
y_preds["CNN + XGBoost (HCB + Net)"] = (y_preds["CNN"] + y_preds["XGBoost"]) / 2
y_preds["XGBoost + Random Forest"] = (y_preds["XGBoost"] + y_preds["Random Forest"]) / 2

# Set up the plot grid (3 columns, 4 rows)
num_models = len(y_preds)
cols = 3  # Fixed columns
rows = 4  # Fixed rows
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4), dpi=600)

axes = axes.flatten()  # Flatten to loop easily
colors = sns.color_palette("husl", num_models)

# Iterate through each model and plot
for i, (name, y_pred) in enumerate(y_preds.items()):
    axes[i].plot(y_test, label="Actual", linestyle="dashed", color='black', linewidth=2)
    axes[i].plot(y_pred, label=f"Predicted ({name})", linestyle="solid", color=colors[i], linewidth=2)
    axes[i].set_title(f"{name}", fontsize=14, fontweight='bold')
    axes[i].set_xlabel("Test Sample Index", fontsize=12)
    axes[i].set_ylabel("Charging Demand (Scaled)", fontsize=12)
    axes[i].legend()
    axes[i].grid(True, linestyle='--', alpha=0.6)

# Hide unused subplots if any (in case num_models < 12)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(results_path, "model_comparison.png"))
plt.show()

print("Model training, evaluation, and visualization completed.")
