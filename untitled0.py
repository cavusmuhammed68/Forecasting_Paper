import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
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

# Define and train LSTM model
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
history_lstm = lstm_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Define and train GRU model
gru_model = Sequential([
    GRU(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    GRU(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
history_gru = gru_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Support Vector Regressor model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

# Generate predictions
y_pred_lstm = lstm_model.predict(X_test)
y_pred_gru = gru_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_svr = svr_model.predict(X_test)

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

# Visualization
models = {"LSTM": y_pred_lstm, "GRU": y_pred_gru, "Random Forest": y_pred_rf, "SVR": y_pred_svr}

for model_name, predictions in models.items():
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", linestyle="dashed", marker="o", color='black', alpha=0.7)
    plt.plot(predictions, label=f"Predicted ({model_name})", linestyle="solid", marker="x", alpha=0.7)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Charging Demand (Scaled)")
    plt.title(f"Actual vs Predicted Charging Demand - {model_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

print("Model training completed and results visualized.")


# Additional Visualization

# Residual Plots
plt.figure(figsize=(12, 6))
for model_name, predictions in models.items():
    residuals = y_test - predictions.flatten()
    plt.scatter(range(len(residuals)), residuals, label=f"Residuals ({model_name})", alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.2)
plt.xlabel("Test Sample Index")
plt.ylabel("Residuals")
plt.title("Residual Plot for Predictions")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Histogram of Prediction Errors
plt.figure(figsize=(12, 6))
for model_name, predictions in models.items():
    residuals = y_test - predictions.flatten()
    plt.hist(residuals, bins=30, alpha=0.5, label=f"Error Distribution ({model_name})", edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Scatter Plot: Predicted vs Actual Values
plt.figure(figsize=(12, 6))
for model_name, predictions in models.items():
    plt.scatter(y_test, predictions.flatten(), alpha=0.5, label=f"{model_name} Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="black")
plt.xlabel("Actual Charging Demand (Scaled)")
plt.ylabel("Predicted Charging Demand (Scaled)")
plt.title("Scatter Plot of Predicted vs Actual Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Line Plot of Loss During Training
plt.figure(figsize=(12, 6))
plt.plot(history_lstm.history['loss'], label="LSTM Train Loss", linestyle="solid")
plt.plot(history_lstm.history['val_loss'], label="LSTM Validation Loss", linestyle="dashed")
plt.plot(history_gru.history['loss'], label="GRU Train Loss", linestyle="solid")
plt.plot(history_gru.history['val_loss'], label="GRU Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Across Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
