import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

# Train and Predict with Traditional Models
y_preds = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_preds[name] = model.predict(X_test)

# LSTM Model
lstm_model = Sequential([
    LSTM(300, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(300, return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
lstm_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
y_preds["LSTM"] = lstm_model.predict(X_test).flatten()

# GRU Model
gru_model = Sequential([
    GRU(300, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.4),
    GRU(300, return_sequences=False),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dense(1)
])
gru_model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
gru_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
y_preds["GRU"] = gru_model.predict(X_test).flatten()

# CNN Model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)
])
cnn_model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
cnn_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
y_preds["CNN"] = cnn_model.predict(X_test).flatten()

# Hybrid Models
hybrid_models = {
    "LSTM + Random Forest": (y_preds["LSTM"] + y_preds["Random Forest"]) / 2,
    "LSTM + XGBoost": (y_preds["LSTM"] + y_preds["XGBoost"]) / 2,
    "LSTM + SVM": (y_preds["LSTM"] + y_preds["SVM"]) / 2,
    "CNN + XGBoost": (y_preds["CNN"] + y_preds["XGBoost"]) / 2,
    "XGBoost + Random Forest": (y_preds["XGBoost"] + y_preds["Random Forest"]) / 2
}
y_preds.update(hybrid_models)

# Evaluate Models
for name, y_pred in y_preds.items():
    print(f"\n{name} Performance:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")

print("Model training and evaluation completed.")

# Visualization
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(y_preds))
plt.title("Boxplot of Predictions by Different Models")
plt.xlabel("Models")
plt.ylabel("Charging Demand (Scaled)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for name, y_pred in y_preds.items():
    plt.plot(y_pred, label=name)
plt.plot(y_test, label="Actual", linestyle="dashed", color='black')
plt.title("Actual vs Predicted Charging Demand")
plt.xlabel("Test Sample Index")
plt.ylabel("Charging Demand (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

print("Model training, evaluation, and visualization completed.")








# Generate Individual Model Visualizations
for name, y_pred in y_preds.items():
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", linestyle="dashed", color='black')
    plt.plot(y_pred, label=f"Predicted ({name})", linestyle="solid")
    plt.title(f"Actual vs Predicted Charging Demand - {name}")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Charging Demand (Scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()

print("Model training, evaluation, and visualization completed.")








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
    
    print(f"\n{name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# Generate Individual Model Visualizations
for name, y_pred in y_preds.items():
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", linestyle="dashed", color='black')
    plt.plot(y_pred, label=f"Predicted ({name})", linestyle="solid")
    plt.title(f"Actual vs Predicted Charging Demand - {name}")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Charging Demand (Scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Display Performance Metrics as a Table
metrics_df = pd.DataFrame(metrics).T
print("\nPerformance Metrics for All Models:")
print(metrics_df)

print("Model training, evaluation, and visualization completed.")













