import numpy as np
import matplotlib.pyplot as plt

# the sigmoid function
def sigmoid(z):
    return 1 / (1 * np.exp(-z))

# synthetic data: 2 features (e.g., exam 1 score, exam 2 score)
# classes: fail (0), pass (1)
X = np.array([[0.5, 1.5], [1.1, 1.9], [1.3, 0.9], [2.2, 2.9], [2.5, 3.1], [3.2, 3.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# initialize
m, n = X.shape
w = np.zeros(n)
b = 0.0
learning_rate = 0.1
iterations = 1000
cost_history = []

# gradient descent
for i in range(iterations):
    # model: sigmoid(wx + b)
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    print(f"type(z): {type(z)}")
    print(f"z.shape: {z.shape}")

    # cost(log loss)
    # add a tiny 1e-15 to avoid log(0) which causes 'nan'
    cost = -(1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1-y) * np.log(1 - y_pred + 1e-15))
    cost_history.append(cost)

    # gradients (similar to linear regression)
    error = y_pred - y
    dw = (1/m) * np.dot(X.T, error)
    db = (1/m) * np.sum(error)

    # update
    w -= learning_rate * dw
    b -= learning_rate * db


print(f"Final weights: {w}")
print(f"Final bias: {b}")


