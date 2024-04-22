from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import log_loss

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
#Instantiate the decision tree classifier with Gini impurity and max depth of 5
clf = DecisionTreeClassifier(criterion='gini', max_depth=5)

# Fit the model to the training data
clf.fit(X_train_final, y_train_final)

#Q.b)
# Make probabilistic predictions on the validation data
prob_predictions = clf.predict_proba(X_valid)

# Calculate the cross-entropy (log loss)
cross_entropy = log_loss(y_valid, prob_predictions)

print(f'Cross-entropy on the validation predictions: {cross_entropy}')

#Q.c)
# Instantiate the decision tree classifier with information gain (entropy) and max depth of 5
clf_info_gain = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# Fit the model to the training data
clf_info_gain.fit(X_train_final, y_train_final)

#Q.d)
# Make probabilistic predictions on the validation data using the model trained with information gain
prob_predictions_info_gain = clf_info_gain.predict_proba(X_valid)

# Calculate the cross-entropy (log loss)
cross_entropy_info_gain = log_loss(y_valid, prob_predictions_info_gain)

print(f'Cross-entropy on the validation predictions (with information gain): {cross_entropy_info_gain}')