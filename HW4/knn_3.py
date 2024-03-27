#(instance and label)
point1 = (1.2, 3.2)
point2 = (2.8, 8.5)
point3 = (2, 4.7)
point4 = (0.9, 2.9)
point5 = (5.1, 11)

def euclidian_distance(instance1, instance2):
    """
    Calculates the Euclidean distance between two instances.

    Parameters:
    instance1 (tuple): The coordinates of the first instance.(x,y)
    instance2 (tuple): The coordinates of the second instance.(x,y)

    Returns:
    float: The Euclidean distance between the two instances.
    """
    return sum((x - y)**2 for x, y in zip(instance1, instance2)) ** 0.5

# Example usage
#print(euclidian_distance((1.2, 3.2), (2.8, 8.5)))


# Previous generation
def knn_regression_predict(instance, training_data, labels, k=3):
    """
    Predicts the label for a given instance using the KNN regression algorithm.

    Parameters:
    instance (tuple): The coordinates of the instance to predict the label for.
    training_data (list): The list of tuples containing the coordinates and labels of the training data. ((x,y),(x,y))
    labels (list): The list of labels for each instance in the training data.
    k (int): The number of nearest neighbors to consider for prediction. Default is 3.

    Returns:
    int or float: The predicted label for the given instance.
    """
    distances = []
    for data, label in zip(training_data, labels):
        dist = euclidian_distance(instance, data)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0]) #Sorts the distances in ascending order based on the first element
    neighbors = distances[:k]

    labels = [label for (_, label) in neighbors] #This gives us just the labels without the distances

    return sum(labels) / k



training_data = [point1, point2, point3, point4, point5] #((x,y),(x,y),(x,y))
#I want to add all the first elements of each points in a list called instances
labels = [point[1] for point in training_data]

print("The predicted label for (1.5) :" , knn_regression_predict((1.5,), training_data, labels))
print("The predicted label for (4.5) :" , knn_regression_predict((4.5,), training_data, labels))

