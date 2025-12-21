import numpy as np
import matplotlib.pyplot as plt

# Setup dummy data
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13]) # y = 2x + 3

# Initialize parameters 
w = 0.0
b = 0.0
m = len(X)
learning_rate = 0.02
epsilon = 0.0000001 # convergence treshold
max_iter = 1000
cost_history = []

# Finding parameters 
for i in range(max_iter):
    y_pred = w * X + b
    cost = np.mean((y_pred - y)**2)
    cost_history.append(cost)

    # calculate gradients
    dw = (2/m) * np.sum((y_pred - y) * X)
    db = (2/m) * np.sum((y_pred - y))

    # update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # check for convergence
    if i > 0 and abs(cost_history[i-1] - cost_history[i]) < epsilon:
        print(f"Converged at iteration {i}.")
        break

print(f"w: {w}")
print(f"b: {b}")