import numpy as np

# the training examples with the format: (input_features, label)
training_examples = [
    (np.array([-3, 5]), 1),  # Positive example
    (np.array([-4, -2]), 1), # Positive example
    (np.array([2, 1]), -1),  # Negative example
    (np.array([4, 3]), -1)   # Negative example
]

# Initialize weights w0 (bias), w1, and w2 to 0
weights = np.array([0.0, 0.0, 0.0])
learning_rate = 0.1

# Perceptron training algorithm
for iteration in range(3):  # Complete at most 3 iterations
    print(f"\n--- Iteration {iteration + 1} ---")
    for input_features, label in training_examples:
        # Calculate the weighted sum, include the bias (weights[0])
        weighted_sum = np.dot(weights[1:], input_features) + weights[0]
        # Apply the step function to determine the output
        output = 1 if weighted_sum > 0 else -1
        # Update weights if the output doesn't match the label
        if output != label:
            weights[1:] += learning_rate * label * input_features
            weights[0] += learning_rate * label  # Update bias
        print(f"Point: {input_features}, Expected Label: {label}, Output: {output}")
        print(f"Updated Weights: {weights}")
    
    # Show weights after each iteration
    print(f"Weights after iteration {iteration + 1}: {weights}")

# Check if the weights are final
is_final = True
for input_features, label in training_examples:
    weighted_sum = np.dot(weights[1:], input_features) + weights[0]
    output = 1 if weighted_sum > 0 else -1
    if output != label:
        is_final = False
        break

if is_final:
    print("\nThese weights are final.")
else:
    print("\nThese weights are not final.")
