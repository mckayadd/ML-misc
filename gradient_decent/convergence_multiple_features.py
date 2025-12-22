import numpy as np

# Data
'''
    Contains three features, e.g., size, bedrooms, age
    y = 2x + 1.5k + 3z
'''
X = np.array([
    [0.5, 2, 1],
    [1, 1, 1],
    [3, 4, 2],
    [2, 10, 1],
    [10, 20, 30],
    [0.1, 0.2, 0.1]
])

y = np.array([11, 10.5, 22, 26, 144, 4.8]) # 

# Initialization

m, n = X.shape # m examples, n features
w = np.zeros(n)
b = 0.0
learning_rate = 1.0e-4 # needs to be small for an example like housing price prediction, e.g., 1.0e-7 
iterations = 10**6
epsilon = 10**-10 # convergence treshold
cost_history = []

# Vectorized gradient descent
for i in range(iterations):
    y_pred = np.dot(X, w) + b
    error = y_pred - y

    cost = np.mean((y_pred - y)**2)
    cost_history.append(cost)

    dj_dw = (2/m) * np.dot(X.T, error) # Transpose aligns features with errors
    dj_db = (2/m) * np.sum(error)

    w -= learning_rate * dj_dw
    b -= learning_rate * dj_db

    # check for convergence
    if i > 0 and abs(cost_history[i-1] - cost_history[i]) < epsilon:
        print(f"Converged at iteration {i}.")
        break

print(f"Final weights: {w}")
print(f"Final bias: {b}")
