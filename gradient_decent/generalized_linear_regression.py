import numpy as np

# 1. Setup: m=4 examples, n=3 features
# Features: [Size, Bedrooms, Age]
X = np.array([
    [2104, 5, 45],
    [1416, 3, 40],
    [1534, 3, 30],
    [852, 2, 36]
])
y = np.array([460, 232, 315, 178]) # Prices in $1000s

# 2. Initialize
m, n = X.shape
w = np.zeros(n)
b = 0.0
alpha = 1.0e-7  # Small learning rate for large feature values
lambda_reg = 0.1 
iterations = 1000

# 3. Gradient Descent Loop
for i in range(iterations):
    # Vectorized Prediction
    y_pred = np.dot(X, w) + b
    error = y_pred - y
    
    # Gradient Calculation with Regularization
    # N.B. We divide the reg term by m to keep it scaled with the data
    dw = (1/m) * (np.dot(X.T, error) + (lambda_reg * w))
    db = (1/m) * np.sum(error)
    
    # Update
    w -= alpha * dw
    b -= alpha * db

print(f"Weights for Size, Bed, Age: {w}")
print(f"Base Price (Bias): {b}")