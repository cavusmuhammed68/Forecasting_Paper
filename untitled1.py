import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
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

# Hyperparameter tuning settings
lstm_units = 256
dropout_rate = 0.3
batch_size = 32
epochs = 150
learning_rate = 0.0005

# Define and train LSTM model
lstm_model = Sequential([
    LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(dropout_rate),
    LSTM(lstm_units, return_sequences=False),
    Dropout(dropout_rate),
    Dense(128, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
history_lstm = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Train XGBoost with optimized parameters
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Combined XGBoost and LSTM model predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lstm = lstm_model.predict(X_test).flatten()
y_pred_xgb_lstm = (y_pred_xgb + y_pred_lstm) / 2  # Averaging predictions

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R^2 Score: {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_test, y_pred_xgb_lstm, "XGBoost + LSTM Optimized")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual", linestyle="dashed", marker="o", color='black', alpha=0.7)
plt.plot(y_pred_xgb_lstm, label="Predicted (XGBoost + LSTM Optimized)", linestyle="solid", marker="x", alpha=0.7)
plt.xlabel("Test Sample Index")
plt.ylabel("Charging Demand (Scaled)")
plt.title("Actual vs Predicted Charging Demand - XGBoost + LSTM Optimized")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Model training completed with enhanced hyperparameters and results visualized.")
