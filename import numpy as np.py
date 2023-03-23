import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Load the data
data = pd.read_csv("D:\Poweball\Test.csv", index_col=False)

# Normalize the data
data_normalized = (data - data.min()) / (data.max() - data.min())

# Prepare the input and output data for training
X = data_normalized[:-1].values.reshape(-1, 1, 8)
y = data_normalized[1:].values

# Create the GRU model
model = Sequential()
model.add(GRU(units=64, activation="tanh", input_shape=(1, 8)))
model.add(Dense(8, activation="linear"))

# Compile and train the model
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=690, verbose=0)

def predict_next_row(last_row_normalized):
    next_row_normalized = model.predict(last_row_normalized)
    min_values = data.min().values.reshape(1, -1)
    max_values = data.max().values.reshape(1, -1)
    next_row = next_row_normalized * (max_values - min_values) + min_values
    return next_row

def get_unique_numbers(row):
    unique_numbers = np.unique(row)
    while unique_numbers.size < 7:
        # add some randomness to the process of selecting unique numbers
        new_number = np.random.randint(1, 36)
        if new_number not in unique_numbers:
            unique_numbers = np.append(unique_numbers, new_number)
    return unique_numbers[:7]

last_row_normalized = data_normalized.iloc[-1].values.reshape(1, 1, -1)

# Iterate 10 times and print predictions
for i in range(10):
    next_row = predict_next_row(last_row_normalized)
    next_row = next_row.round().astype(int)
    next_row[0, :7] = get_unique_numbers(next_row[0, :7])

    print(f"Iteration {i + 1}:")
    print("Predicted next row (1-35):", ", ".join(map(str, next_row[0, :7])))
    print("Predicted next exclusive number (1-20):", next_row[0, 7])
    last_row_normalized = (next_row - data.min().values) / (data.max().values - data.min().values)
    last_row_normalized = last_row_normalized.reshape(1, 1, -1)
    print()
