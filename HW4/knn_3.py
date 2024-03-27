import matplotlib.pyplot as plt

#(instance and label)
point1 = (1.2, 3.2)
point2 = (2.8, 8.5)
point3 = (2, 4.7)
point4 = (0.9, 2.9)
point5 = (5.1, 11)

def euclidian_distance(instance1, instance2):
    return sum((x - y)**2 for x, y in zip(instance1, instance2)) ** 0.5


# KNN regression code
def knn_regression_predict(instance, training_data, labels, k=3):
    distances = []
    for data, label in zip(training_data, labels):
        dist = euclidian_distance(instance, data)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0]) #Sorts the distances in ascending order based on the first element
    neighbors = distances[:k] #This gives us the first k elements of the tuples of distances and the labels

    labels = [label for (_, label) in neighbors] #This gives us just the labels without the distances

    return sum(labels) / k

training_data = [point1, point2, point3, point4, point5] #((x,y),(x,y),(x,y))
#I want to add all the first elements of each points in a list called instances
labels = [point[1] for point in training_data]

print("The predicted label for (1.5) :" , knn_regression_predict((1.5,), training_data, labels))
print("The predicted label for (4.5) :" , knn_regression_predict((4.5,), training_data, labels))

def weighted_knn_regression_predict(instance, training_data, labels, k=3):
    distances = []
    for data, label in zip(training_data, labels):
        dist = euclidian_distance(instance, data)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0]) #Sorts the distances in ascending order based on the first element
    neighbors = distances[:k]

    #Initialize the weights I want to use
    weighted_sum = 0
    total_weight = 0

    for distance, label in neighbors:
        #The weights are the inverse of the distance
        weight = 1 / (distance + 0.0001)
        weighted_sum += weight * label
        total_weight += weight

    return weighted_sum / total_weight

print("The weighted predicted label for (1.5) :" , weighted_knn_regression_predict((1.5,), training_data, labels))
print("The weighted predicted label for (4.5) :" , weighted_knn_regression_predict((4.5,), training_data, labels))

#Adding a graphical overview
x_ticks = [i * 0.1 for i in range(61)]

equal_weight_predictions = [knn_regression_predict((x,), training_data, labels) for x in x_ticks]
distance_weight_predictions = [weighted_knn_regression_predict((x,), training_data, labels) for x in x_ticks]

#Plotting 
plt.figure(figsize=(10, 10))
x_values, y_values = zip(*training_data)
plt.scatter(x_values, y_values, color="black", label="Training data points")

# Plot predicted labels for equal weight method
plt.plot(x_ticks, equal_weight_predictions, label='Equal Weight Predictions', color='red')

# Plot predicted labels for distance-weighted method
plt.plot(x_ticks, distance_weight_predictions, label='Distance Weighted Predictions', color='blue')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Predicted Label')
plt.title('Predicted Labels with Different Methods')
plt.legend()

# Show plot
plt.grid(True)
plt.show()