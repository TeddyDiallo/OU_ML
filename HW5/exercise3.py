from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
import pandas as pd

# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
  
y_encoded = y.iloc[:, 0].map({"Cammeo": 0, "Osmancik": 1})

X_list = X.values.tolist()

#Taking 20% of the data at random as the testing set 
X_train, X_test, y_train, y_test = train_test_split(X_list, y_encoded, test_size=0.2, random_state=42)
#print(len(X_train), len(X_test), len(y_train), len(y_test))

#Split the raining into 75% for training and 25% for validation
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
#print(len(X_train_final), len(X_valid), len(y_train_final), len(y_valid))

#print(X_train[:5])

# (a) Fit two logistic regression models

# Model with L2 regularization (default)
model_l2 = LogisticRegression(penalty='l2', random_state=42)
model_l2.fit(X_train_final, y_train_final)

# Model with no regularization
model_none = LogisticRegression(penalty=None, random_state=42)
model_none.fit(X_train_final, y_train_final)

# (b) Make predictions and evaluate the models

# Predictions with L2 regularization model
y_pred_l2 = model_l2.predict(X_valid)

# Predictions with no regularization model
y_pred_none = model_none.predict(X_valid)

# Empirical risk using the 0-1 loss for both models
risk_l2 = 1 - accuracy_score(y_valid, y_pred_l2)
risk_none = 1 - accuracy_score(y_valid, y_pred_none)

# Confusion matrices for both models
conf_matrix_l2 = confusion_matrix(y_valid, y_pred_l2)
conf_matrix_none = confusion_matrix(y_valid, y_pred_none)

# Printing the results
print(f"Empirical risk (0-1 loss) with L2 regularization: {risk_l2}")
print(f"Empirical risk (0-1 loss) with no regularization: {risk_none}")
print("\nConfusion matrix with L2 regularization:")
print(conf_matrix_l2)
print("Confusion matrix with no regularization:")
print(conf_matrix_none)

# (c) Analysis of performance
precision_l2 = precision_score(y_valid, y_pred_l2)
precision_none = precision_score(y_valid, y_pred_none)
print(f"\nPrecision with L2 regularization: {precision_l2}")
print(f"Precision with no regularization: {precision_none}")


# Combine the training and validation sets
X_train_val = X_train_final + X_valid  
y_train_val = y_train_final.tolist() + y_valid.tolist()

# Train a new logistic regression model on the combined training and validation set
model_final = LogisticRegression(random_state=52)
model_final.fit(X_train_val, y_train_val)

# Make a prediction on the test set
y_pred_test = model_final.predict(X_test)

# Evaluate the model's performance
risk_test = 1 - accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)

# Print out the final empirical risk and confusion matrix
print(f"\nFinal empirical risk (0-1 loss) on the test set: {risk_test}")
print("Final confusion matrix on the test set:")
print(conf_matrix_test)

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
train_df.to_csv('train_data3.csv', index=False)
valid_df.to_csv('valid_data3.csv', index=False)
test_df.to_csv('test_data3.csv', index=False)
