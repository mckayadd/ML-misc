import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dataset for OR gate
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
expected_output = np.array([[0], [1], [1], [1]])

np.random.seed(1)

# Now we have Weights AND a Bias
weights = np.random.random((2, 1))
bias = np.random.random((1, 1))

learning_rate = 0.5

for epoch in range(20000):
    # Forward Pass: (Inputs * Weights) + Bias
    linear_layer = np.dot(inputs, weights) + bias
    actual_output = sigmoid(linear_layer)
    
    # Calculate Error
    error = expected_output - actual_output
    
    # Backpropagation
    adjustments = error * sigmoid_derivative(actual_output)
    
    # Update Weights and Bias
    weights += np.dot(inputs.T, adjustments) * learning_rate
    bias += np.sum(adjustments) * learning_rate

print("Results with Bias:")
print(actual_output)
print("\nFinal Learned Bias:", bias)