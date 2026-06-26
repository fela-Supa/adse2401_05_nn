"""
RNN is a type of ANN designed for sequential data processing. In contrast
to FNNs, RNNs have connections that form a directed cycle, allowing them
to maintain a hidden state.
"""

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
from io import StringIO

# Download the Air Passengers dataset from URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Extract the 'Passengers' column for time series prediction
passengers = data['Passengers'].values.astype(float)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
passengers = scaler.fit_transform(passengers.reshape(-1, 1))

# Split the dataset into train and test sets
train_size = int(len(passengers) * 0.67)
test_size = len(passengers) - train_size
train, test = passengers[0:train_size], passengers[train_size:len(passengers)]


# Convert the time series data into input-output pairs
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)
time_steps = 10
X_train, y_train = create_dataset(train, time_steps)
X_test, y_test = create_dataset(test, time_steps)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], time_steps, 1))
X_test = np.reshape(X_test, (X_test.shape[0], time_steps, 1))

# Define the model
model = Sequential([
    SimpleRNN(32, input_shape=(time_steps, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train, epochs=100,batch_size=1, verbose=2
)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(y_train[0], train_predict[:, 0]))
print('Train RMSE:', train_score)
test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))

print('Test RMSE:', test_score)

# Plotting
plt.plot(scaler.inverse_transform(passengers), label='Original Data')
plt.plot(np.concatenate((train_predict, test_predict)), label='Predictions')
plt.legend()
plt.show()