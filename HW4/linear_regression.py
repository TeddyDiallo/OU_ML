point1 = (1.2, 3.2)
point2 = (2.8, 8.5)
point3 = (2, 4.7)
point4 = (0.9, 2.9)
point5 = (5.1, 11)

training_data = [point1, point2, point3, point4, point5]

wo, w1 = 1,1
learning_rate = 0.01
num_iterations = 3

#Function to predict the label for a given point and weights
def predict_label(point, wo, w1):
    return wo + w1*point 


#initialize the gradients
grad_w0 = 0
grad_w1 = 0

#Perform the gradient descent 
for i in range(num_iterations):

    #Compute the prediction for each training point
    for point in training_data:
        prediction = predict_label(point[0], wo, w1)
        #Compute the gradients
        grad_w0 += prediction - point[1]
        grad_w1 += (prediction - point[1])*point[0]

    #Update the weights
    wo -= learning_rate*grad_w0 / len(training_data)
    w1 -= learning_rate*grad_w1 / len(training_data)

    print(f"Iteration {i+1}: w0 = {wo}, w1 = {w1}")

print("\n The predicted value at the point (1.5) is : ", predict_label(1.5, wo, w1))
print("The predicted value at the point (4.5) is : ", predict_label(4.5, wo, w1))