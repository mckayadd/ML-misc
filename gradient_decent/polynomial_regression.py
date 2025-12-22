# import numpy as np
# import matplotlib.pyplot as plt

# # Setup dummy data
# X = np.array([1, 2, 3, 4, 5])
# y = np.array([1, 8, 27, 64, 125]) # y = x**3

# # Initialize parameters 
# w = 0.0
# b = 0.0
# m = len(X)
# learning_rate = 0.005
# epsilon = 0.0001 # convergence treshold
# max_iter = 1000
# cost_history = []

# # Finding parameters 
# for i in range(max_iter):
#     y_pred = w * X + b # first, try with a linear function 
#     cost = np.mean((y_pred - y)**2)
#     cost_history.append(cost)

#     # calculate gradients
#     dw = (2/m) * np.sum((y_pred - y) * X)
#     db = (2/m) * np.sum((y_pred - y))

#     # update parameters
#     w -= learning_rate * dw
#     b -= learning_rate * db

#     # check for convergence
#     if i > 0 and abs(cost_history[i-1] - cost_history[i]) < epsilon:
#         print(f"Converged at iteration {i}.")
#         break

# print(f"w: {w:.3}")
# print(f"b: {b:.3}")


# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(range(len(cost_history)), cost_history, color='red', linewidth=2)
# plt.title("Learning Curve: Cost vs. Iterations")
# plt.xlabel("Step Number")
# plt.ylabel("Cost J(w,b)")
# plt.yscale('log') # to better observe the convergence
# plt.grid(True, alpha=0.3)

# # Plot 2: The Final Prediction Line
# plt.subplot(1, 2, 2)
# plt.scatter(X, y, color='blue', label='Data Points')
# plt.plot(X, w*X + b, color='green', label=f'Final Fit: {w:.2f}x + {b:.2f}')
# plt.title("Final Model Fit")
# plt.legend()

# plt.tight_layout()
# plt.show()


############################################################################

import numpy as np
import matplotlib.pyplot as plt

# Setup dummy data
X1 = np.array([1, 2, 3, 4, 5])
X2 = X1**2
X3 = X1**3
y = np.array([1, 8, 27, 64, 125]) # y = x**3

# Initialize parameters 
w1 = 0.0
w2 = 0.0
w3 = 0.0
b = 0.0
m = len(X1)
learning_rate = 0.000001
epsilon = 0.00001 # convergence treshold
max_iter = 10000
cost_history = []

# Finding parameters 
for i in range(max_iter):
    y_pred = w1 * X1 + w2 * X2 + w3 * X3 + b # first, try with a linear function 
    cost = np.mean((y_pred - y)**2)
    cost_history.append(cost)

    # calculate gradients
    dw1 = (2/m) * np.sum((y_pred - y) * X1)
    dw2 = (2/m) * np.sum((y_pred - y) * X2)
    dw3 = (2/m) * np.sum((y_pred - y) * X3)
    db = (2/m) * np.sum((y_pred - y))

    # update parameters
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    w3 -= learning_rate * dw3
    b -= learning_rate * db

    # check for convergence
    if i > 0 and abs(cost_history[i-1] - cost_history[i]) < epsilon:
        print(f"Converged at iteration {i}.")
        break

print(f"w1: {w1:.3}")
print(f"w2: {w2:.3}")
print(f"w2: {w3:.3}")
print(f"b: {b:.3}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(cost_history)), cost_history, color='red', linewidth=2)
plt.title("Learning Curve: Cost vs. Iterations")
plt.xlabel("Step Number")
plt.ylabel("Cost J(w,b)")
plt.yscale('log') # to better observe the convergence
plt.grid(True, alpha=0.3)

# Plot 2: The Final Prediction Line
plt.subplot(1, 2, 2)
plt.scatter(X1, y, color='blue', label='Data Points')
plt.plot(X1, w1 * X1 + w2 * X2 + w3 * X3 + b, color='green', label=f'Final Fit: x + {b:.2f}')
plt.title("Final Model Fit")
plt.legend()

plt.tight_layout()
plt.show()