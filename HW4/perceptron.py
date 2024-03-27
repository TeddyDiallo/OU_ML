#Declare a list of points
point1 = [(-3,5), ("+")]
point2 = [(-4,-2), ("+")]
point3 = [(2,1), ("-")]
point4 = [(4,3), ("-")]

#Declare the weights
w0, w1, w2 = 0 , 0, 0
learning_rate = 0.1
pos = +1
neg = -1

#Perceptron training algorithm
#iterations
for i in range(3):
    print(f"\n--- Iteration {i+1} ---")
    for x, y in [point1, point2, point4, point3]:
        #Note that x is a coordinate tuple and y is the label either + or -
        #expected output
        expected_output = pos if y == '+' else neg
        #calculate output
        output = 1 if ((w0 + w1*x[1] + w2*x[0]) > 0) else -1
        print(f"\nPoint: {x}, Expected output: {expected_output}, Output: {output}")
        #update weights
        if output != expected_output:
            w0 += learning_rate*(expected_output - output)
            w1 += learning_rate*(expected_output - output)*x[0]
            w2 += learning_rate*(expected_output - output)*x[1]
        print(f"Updated weights: w0 = {w0}, w1 = {w1}, w2 = {w2}")
    #print weights
    print(f"\nFinal weights after the 3 iterations: w0 = {w0}, w1 = {w1}, w2 = {w2}\n")

    #We can check of these are the final weights by finding the output for a point and checking if that was the expected output 
    #For point 1 for instance, the expected output should be +, let's check if that is the case
    output = 1 if ((w0 + w1*point1[0][1] + w2*point1[0][0]) > 0) else -1
    if output == pos:
        print(f"Point {point1} is now correctly classified. These could be the final weights.")
    else:
        print(f"Point {point1} is still incorrectly classified. These are not the final weights.")
