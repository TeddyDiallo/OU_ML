from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
import csv

# fetch dataset 
abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
X = abalone.data.features 
y = abalone.data.targets 
  
# metadata 
#print(abalone.metadata) 
  
# variable information 
#print(abalone.variables) 

#print(X.head())

# Convert the DataFrame to a list of lists
X_list = X.values.tolist()

def one_hot_encode(data, index):
    # Define a mapping from category to one-hot encoding
    mapping = {'M': [1, 0, 0], 'F': [0, 1, 0], 'I': [0, 0, 1]}
    for row in data:
        # Replace the categorical value with its one-hot encoded list
        category = row[index]
        row[index:index+1] = mapping[category]
    return data

# Assuming the first attribute is at index 0 for the categorical variable
index_of_categorical_attribute = 0
X_encoded = one_hot_encode(X_list, index_of_categorical_attribute)

#print(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
#print(len(X_train), len(X_test), len(y_train), len(y_test))

# Split the training data into training and validation sets
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
print(len(X_train_final), len(X_valid), len(y_train_final), len(y_valid))