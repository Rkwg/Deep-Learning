import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to calculate distance between receiver and transmitter
def calculate_log_distance(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return -10 * np.log10(distance)

# Computing weights for the linear regression
def log_distance_regression(X, Y):
    X_transpose = np.transpose(X)
    X_X_transpose = X_transpose @ X
    X_X_transpose_inv = np.linalg.inv(X_X_transpose)
    w = np.matmul(np.matmul(X_X_transpose_inv, X_transpose), Y)
    return w

# Read data from files
X = pd.read_csv('locations.txt', header=None, names=['X', 'Y'], delimiter='\t')
y_original = pd.read_csv('rss_values.txt', header=None, delimiter=' ')

# Calculate log distance matrix
log_distance = np.zeros((len(y_original), len(X)))
for i in range(len(y_original)):
    for j in range(len(X)):
        if y_original.iloc[i, j] == np.inf:
            for k in range(len(X)):
                log_distance[i][k] = calculate_log_distance(X.iloc[j]['X'], X.iloc[j]['Y'], X.iloc[k]['X'], X.iloc[k]['Y'])

# Prepare data for training
x = []
y = []
for i in range(len(y_original)):
    for j in range(len(X)): #44
        if log_distance[i][j] != np.inf and y_original.iloc[i, j] != np.inf:
            x.append(log_distance[i][j])
            y.append(y_original.iloc[i, j])

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# Building Neural Network Model
def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),  # Input layer with shape (1,), representing a single feature
        tf.keras.layers.Dense(720, activation='relu'),
        tf.keras.layers.Dense(480, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Using Mean Absolute Error (MAE) 
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# Splitting data into train and test sets (80:20)
# Get the number of samples
n_samples = len(x)

# Create shuffled indices
indices = np.arange(n_samples)
np.random.shuffle(indices)

# Define the split ratio
split_ratio = 0.8

# Calculate the split index
split_index = int(n_samples * split_ratio)

# Split the data based on shuffled indices
train_indices = indices[:split_index]
test_indices = indices[split_index:]

x_train, x_test = x[train_indices], x[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Set hyperparameters
learning_rate = 0.00001
batch_size = 16
epochs = 600

# Linear Regression

# Create the design matrix with a column of ones (for intercept) and the feature values
X_design_train = np.hstack((np.ones((len(x_train), 1)), x_train))
X_design_test = np.hstack((np.ones((len(x_test), 1)), x_test))

# Perform linear regression
beta = log_distance_regression(X_design_train, y_train)

# Extract coefficients
intercept = beta[0][0]  # Intercept (beta0)
slope = beta[1][0]      # Slope (beta1)

# Calculate predictions
y_lr_train = X_design_train @ beta
y_lr_test =  X_design_test @ beta
lr_train_loss = np.mean((y_train - y_lr_train) ** 2)
lr_test_loss = np.mean((y_test - y_lr_test) ** 2)

# Neural Network Model
model = build_model(learning_rate)
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))

model.save('final.keras')
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(x_train, y_train, label='Train Data', color='b')
plt.scatter(x_test, y_test, label='Test Data', color='y')

# Plot linear regression
plt.plot(x_train, y_lr_train, label='Linear Regression (Train)', color='green')
plt.plot(x_test, y_lr_test, label='Linear Regression (Test)', color='black', linestyle='--')

# Plot fitted curve using neural network
x_pred = np.linspace(min(x_train), max(x_train))
y_pred = model.predict(x_pred.reshape(-1, 1))
plt.plot(x_pred, y_pred, color='red', label='Neural Network Fitted Curve')

plt.xlabel('Log Distance')
plt.ylabel('RSS Value')
plt.title('RSS Value vs Log Distance')
plt.legend()
plt.grid(True)
plt.show()

# Print loss (MSE)
train_loss, train_mae = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)

print("Neural Network Train Loss:", train_loss)
print("Neural Network Test Loss:", test_loss)
print("Linear Regression Train Loss:", lr_train_loss)
print("Linear Regression Test Loss:", lr_test_loss)

# Plotting training and validation/testing loss versus epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Bar plot showing training error and validation/testing error
labels = ['Train Loss', 'Test Loss']
values = [train_loss, test_loss]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Model Performance Metrics')
plt.show()
