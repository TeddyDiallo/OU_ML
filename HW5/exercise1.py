from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import Ridge
import pandas as pd

abalone = fetch_ucirepo(id=1) 
  
X = abalone.data.features 
y = abalone.data.targets 
X_list = X.values.tolist()

def one_hot_encode(data, index):
    mapping = {'M': [1, 0, 0], 'F': [0, 1, 0], 'I': [0, 0, 1]}
    for row in data:
        category = row[index]
        row[index:index+1] = mapping[category]
    return data

index_of_categorical_attribute = 0
X_encoded = one_hot_encode(X_list, index_of_categorical_attribute)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
#print(len(X_train), len(X_test), len(y_train), len(y_test))

# Split the training data into training and validation sets
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
#print(len(X_train_final), len(X_valid), len(y_train_final), len(y_valid))

# Prepare the grid of parameters
lambda1_values = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
lambda2_values = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

# Create the grid as follows
parameter_grid = []
for l1 in lambda1_values:
    for l2 in lambda2_values:
        if l2 == 0 and l1 != 0:
            # In this case, we have an L1 penalty only, so l1_ratio is 1.
            parameter_grid.append((l1, 1))
        elif l1 == 0 and l2 != 0:
            # Here, we have an L2 penalty only, so l1_ratio is 0.
            parameter_grid.append((l2, 0))
        elif l1 == 0 and l2 == 0:
            # No regularization
            parameter_grid.append((0, 0))
        else:
            # Both L1 and L2 penalties are present
            alpha = l1 + l2
            l1_ratio = l1 / (l1 + l2)
            parameter_grid.append((alpha, l1_ratio))

# Initialize variables to store the best parameters and corresponding RMSE
best_alpha = None
best_l1_ratio = None
best_rmse = float('inf')
best_model = None

# Initialize a list to store MSE for each model
model_performance = [] #for question b, regarding the training set predictions
validation_performance = []

# Iterate over the parameter grid
for alpha, l1_ratio in parameter_grid:
    # Create and fit the ElasticNet model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train_final, y_train_final)
    
    # Predict on the training set
    y_train_pred = model.predict(X_train_final) #the prediction with that model
    mse_train = mean_squared_error(y_train_final, y_train_pred)
    
    # Predict on the validation set
    y_val_pred = model.predict(X_valid)
    mse_val = mean_squared_error(y_valid, y_val_pred)

    # Store the alpha, l1_ratio, and training MSE in the list
    model_performance.append((alpha, l1_ratio, mse_train))
    validation_performance.append((alpha, l1_ratio, mse_val))

    # Calculate the RMSE for the current model on the validation data to see how the model performs on new data
    rmse_val = mean_squared_error(y_valid, y_val_pred, squared=False)
    # If this model is the best so far, based on validation RMSE, store its parameters and RMSE
    if rmse_val < best_rmse:
        best_alpha = alpha
        best_l1_ratio = l1_ratio
        best_rmse = rmse_val
        best_model = model

# Sort the list by training MSE so I can report it
model_performance.sort(key=lambda x: x[2])
validation_performance.sort(key=lambda x: x[2])
# Report the best model's parameters and RMSE QA
print(f"\nBest model based on validation RMSE: alpha={best_alpha}, l1_ratio={best_l1_ratio}, RMSE: {best_rmse}")
print("\n")

# Report all models' MSE on the training set QB
for alpha, l1_ratio, mse_train in model_performance:
    print(f"alpha: {alpha}, l1_ratio: {l1_ratio}, Training MSE: {mse_train}")

print("\n")
#QC
for alpha, l1_ratio, mse_val in validation_performance:
    print(f"alpha: {alpha}, l1_ratio: {l1_ratio}, Validation MSE: {mse_val}")

print("\n")
#QD
best_training_model = model_performance[0]
best_validation_model = validation_performance[0]

# Extract the parameters and calculate lambda1 and lambda2 for the best training model
alpha_training, l1_ratio_training, _ = best_training_model
lambda1_training = alpha_training * l1_ratio_training
lambda2_training = alpha_training * (1 - l1_ratio_training)

# the best validation model
alpha_validation, l1_ratio_validation, _ = best_validation_model
lambda1_validation = alpha_validation * l1_ratio_validation
lambda2_validation = alpha_validation * (1 - l1_ratio_validation)

# Report
print(f"Best model on training data: lambda1={lambda1_training}, lambda2={lambda2_training}")
print(f"Best model on validation data: lambda1={lambda1_validation}, lambda2={lambda2_validation}")

#QE
print("-------------------------------------------")

X_train_val_combined = X_train_final + X_valid
y_train_val_combined = y_train_final.iloc[:, 0].tolist() + y_valid.iloc[:, 0].tolist()

'''# Assuming alpha_best and l1_ratio_best are the best hyperparameters you've found
model_best = ElasticNet(alpha=alpha_validation, l1_ratio=l1_ratio_validation, max_iter=10000)
model_best.fit(X_train_val_combined, y_train_val_combined)

# Make predictions on the test set
y_test_pred = model_best.predict(X_test)

# Calculate the MSE on the test set
mse_test = mean_squared_error(y_test, y_test_pred)

# Print the MSE on the test set
print(f"MSE on the test set: {mse_test}")'''


# Instantiate the Ridge regression model
model_ridge = Ridge(alpha=alpha_validation, max_iter=10000, solver='auto')

# Now you can fit this model to your data
model_ridge.fit(X_train_val_combined, y_train_val_combined)
y_test_pred = model_ridge.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"MSE on the test set: {mse_test}")



#Printing out the dataset the broken down dataset
# Convert the subsets to pandas DataFrames
train_df = pd.DataFrame(X_train_final)
valid_df = pd.DataFrame(X_valid)
test_df = pd.DataFrame(X_test)

# Add the target variable to each DataFrame
train_df['target'] = y_train_final
valid_df['target'] = y_valid
test_df['target'] = y_test

# Save the DataFrames to CSV files
train_df.to_csv('train_data.csv', index=False)
valid_df.to_csv('valid_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
