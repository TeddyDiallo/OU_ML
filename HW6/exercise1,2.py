from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

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