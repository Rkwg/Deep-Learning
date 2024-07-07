
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def calculate_log_distance(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return -10 * np.log10(distance)

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
# Load the saved model
model = tf.keras.models.load_model('final.keras')


y_pred = model.predict(x_test)
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)

print("Neural Network Test Loss:", test_loss)


# Plotting
plt.figure(figsize=(10, 6))

# Plot the test data
plt.scatter(x_test, y_test, label='Test Data', color='b', marker='o', s=50, alpha=0.5, edgecolor='black')
plt.scatter(x_test, y_pred, label='Predictions', color='r', marker='x', s=50)

plt.xlabel('Log Distance')
plt.ylabel('RSS Value')
plt.title('Test Data and Predictions')
plt.legend()
plt.grid(True)
plt.show()