'''
For this question, we need to calculate the euclidean distance between the instances  
'''

def euclidian_distance(x,y):
    return (x-y)**2

#Declare a list of points
point1 = [(-3,5), ("+")]
point2 = [(-4,-2), ("+")]
point3 = [(2,1), ("-")]
point4 = [(4,3), ("-")]

instance1 = (-2,0)
instance2 = (-1,5)

#Calculate the euclidean distance between the points and the instance
print("Distances for Instance1:")
for point in [point1, point2, point3, point4]:
    distance = euclidian_distance(instance1[0],point[0][0]) + euclidian_distance(instance1[1],point[0][1])
    print("Distance to {}: {:.2f}".format(point[1], distance))

print("\nDistances for Instance2:")
for point in [point1, point2, point3, point4]:
    distance = euclidian_distance(instance2[0],point[0][0]) + euclidian_distance(instance2[1],point[0][1])
    print("Distance to {}: {:.2f}".format(point[1], distance))

'''
a. The 1-nearest neighbor to instance (-2,0) is point 2, which is +.
   The 1-nearest neighbor to instance (-1,5) is point 3, which is -.

b. The 3-nearest neighbor to instance (-2,0) is point 1, which is +.
   The 3-nearest neighbor to instance (-1,5) is point 4, which is -.
'''
