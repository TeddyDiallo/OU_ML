import numpy as np

# Training data
X = np.array([1.2, 2.8, 2, 0.9, 5.1])  # input features
Y = [3.2, 8.5, 4.7, 2.9, 11]  # output values

# Initialize weights
w0, w1 = 1, 1

# Learning rate
learning_rate = 0.01

# Perform three iterations of gradient descent
for i in range(3):
    # Calculate the predictions
    Y_pred = w0 + w1 * X
    
    # Calculate the residuals/errors
    residuals = Y - Y_pred
    
    # Compute the gradient for each weight
    grad_w0 = -np.sum(residuals)
    grad_w1 = -np.sum(residuals * X)
    
    # Update the weights
    w0 -= learning_rate * grad_w0
    w1 -= learning_rate * grad_w1

    # Output the weights for verification
    print(f"After iteration {i+1}: w0 = {w0:.6f}, w1 = {w1:.6f}")

# Predictions with the updated weights
x1 = 1.5
x2 = 4.5
Y_pred_x1 = w0 + w1 * x1
Y_pred_x2 = w0 + w1 * x2

# Print the predictions
print(f"Prediction at x1 = {x1}: {Y_pred_x1:.6f}")
print(f"Prediction at x2 = {x2}: {Y_pred_x2:.6f}")
