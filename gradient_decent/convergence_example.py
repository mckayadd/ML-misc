import numpy as np
import matplotlib.pyplot as plt

# Setup dummy data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([5.1, 6.9, 9.2, 11.3, 12.9, 15.0, 17.1, 18.78, 21.02, 23.01]) # y = 2x + 3, some noise added

# Initialize parameters 
w = 0.0
b = 0.0
m = len(X)
learning_rate = 0.005
epsilon = 0.0001 # convergence treshold
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

print(f"w: {w:.3}")
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
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, w*X + b, color='green', label=f'Final Fit: {w:.2f}x + {b:.2f}')
plt.title("Final Model Fit")
plt.legend()

plt.tight_layout()
plt.show()