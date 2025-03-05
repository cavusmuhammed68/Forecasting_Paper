# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:07:03 2025

@author: cavus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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

# Define and train GRU model
gru_model = Sequential([
    GRU(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(dropout_rate),
    GRU(lstm_units, return_sequences=False),
    Dropout(dropout_rate),
    Dense(128, activation='relu'),
    Dense(1)
])
gru_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
history_gru = gru_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Support Vector Regressor model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Generate predictions
y_pred_lstm = lstm_model.predict(X_test).flatten()
y_pred_gru = gru_model.predict(X_test).flatten()
y_pred_rf = rf_model.predict(X_test)
y_pred_svr = svr_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb_lstm = (y_pred_xgb + y_pred_lstm) / 2  # Averaging predictions

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R^2 Score: {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_test, y_pred_lstm, "LSTM")
evaluate_model(y_test, y_pred_gru, "GRU")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_svr, "SVR")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_xgb_lstm, "XGBoost + LSTM Optimized")

# Additional Visualization for Deep Learning Methods
plt.figure(figsize=(12, 6))
plt.plot(history_lstm.history['loss'], label="LSTM Train Loss")
plt.plot(history_lstm.history['val_loss'], label="LSTM Validation Loss")
plt.plot(history_gru.history['loss'], label="GRU Train Loss")
plt.plot(history_gru.history['val_loss'], label="GRU Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_lstm, alpha=0.5, label="LSTM Predictions")
plt.scatter(y_test, y_pred_gru, alpha=0.5, label="GRU Predictions")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot of Predicted vs Actual Values for LSTM & GRU")
plt.legend()
plt.grid(True)
plt.show()

print("Model training completed with enhanced visualizations.")
