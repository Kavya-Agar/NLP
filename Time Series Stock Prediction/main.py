import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler # Changed to MinMaxScaler for better time series scaling
from tensorflow import keras

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the data
data = pd.read_csv("../datasets/AAPL.csv")
print(data.head())
print(data.info())
print(data.describe())

# Initial Data Visualization
# Convert the 'date' column to datetime objects
data['date'] = pd.to_datetime(data['date'])

# Plot 1 - Open and Close Prices over time
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['open'], label='Open', color='blue')
plt.plot(data['date'], data['close'], label='Close', color='red')
plt.title("Open-Close Price over Time")
plt.legend()
plt.show() # Added plt.show() to display the plot

# Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['volume'], label='Volume', color='orange')
plt.title('Stock Volume over Time')
plt.show() # Added plt.show() to display the plot

# Drop non-numeric columns for correlation heatmap
# Note: 'date' is now a datetime object, so it will be excluded.
numeric_data = data.select_dtypes(include=["int64", "float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show() # Added plt.show() to display the plot

# The original code had a typo, 'data' column was created but not used.
# The filter was also incorrect as it tried to filter on the original 'date' column.
# The `prediction` variable was created but not used, so I've commented it out.
# prediction = data.loc[
#     (data['date'] > datetime(2013,1,1)) &
#     (data['date'] < datetime(2018,1,1))
# ]

plt.figure(figsize=(12,6))
plt.plot(data['date'], data['close'], label='Close', color='blue')
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over Time")
plt.show() # Added plt.show() to display the plot

# Prepare for the LSTM Model (Sequential)
stock_close = data.filter(["close"])
dataset = stock_close.values

# Define the training data length
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Preprocessing - Using MinMaxScaler is more common for LSTM
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Training data
training_data = scaled_data[0:training_data_len]

X_train, y_train = [], []

# Create a sliding window for our stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the training data to be 3D for the LSTM model
# The shape should be (number of samples, timesteps, number of features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the model
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# Third Layer (Dense)
model.add(keras.layers.Dense(32, activation="relu")) # Reduced units to avoid overfitting

# Fourth Layer (Dropout)
model.add(keras.layers.Dropout(0.2)) # Reduced dropout rate

# Final Output Layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",
              loss="mean_squared_error") # Changed loss to MSE for a more stable training

# Fit the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prep the test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)

# Reshape the test data to be 3D for the LSTM model
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the model's predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['date'], train['close'], label="Train (Actual)", color='blue')
plt.plot(test['date'], test['close'], label="Test (Actual)", color='orange')
plt.plot(test['date'], test['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show() # Added plt.show() to display the final plot
