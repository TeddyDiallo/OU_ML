from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import numpy as np


# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
  
y_encoded = y.iloc[:, 0].map({"Cammeo": 0, "Osmancik": 1})

X_list = X.values.tolist()

#Taking 20% of the data at random as the testing set 
X_train, X_test, y_train, y_test = train_test_split(X_list, y_encoded, test_size=0.2, random_state=42)

#Split the raining into 75% for training and 25% for validation
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#Q.a)
# Define and train the first neural network with one hidden layer of 30 units
mlp_1_hidden_layer = MLPClassifier(hidden_layer_sizes=(30,), max_iter=400, random_state=42)
mlp_1_hidden_layer.fit(X_train_final, y_train_final)

# Print out the training completion statement
print(f"Training completed for the neural network with 1 hidden layer of 30 units.")

# Define and train the second neural network with two hidden layers of 20 units each
mlp_2_hidden_layers = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=400, random_state=42)
mlp_2_hidden_layers.fit(X_train_final, y_train_final)

# Print out the training completion statement
print(f"Training completed for the neural network with 2 hidden layers of 20 units each.")

#Q.b)
# Making probabilistic predictions for the validation set
prob_predictions_1 = mlp_1_hidden_layer.predict_proba(X_valid)
prob_predictions_2 = mlp_2_hidden_layers.predict_proba(X_valid)

def calculate_cross_entropy(y_true, y_pred_probs):
    # Calculate the cross-entropy using the provided formula
    cross_entropy = -np.mean([
        y * np.log(p) + (1 - y) * np.log(1 - p) 
        for y, p in zip(y_true, y_pred_probs[:, 1])
    ])
    return cross_entropy

# Calculate cross-entropy for both models
cross_entropy_1_manual = calculate_cross_entropy(y_valid, prob_predictions_1)
cross_entropy_2_manual = calculate_cross_entropy(y_valid, prob_predictions_2)

# Print the results
print(f"Manually calculated cross-entropy for the model with 1 hidden layer: {cross_entropy_1_manual:.4f}")
print(f"Manually calculated cross-entropy for the model with 2 hidden layers: {cross_entropy_2_manual:.4f}")
