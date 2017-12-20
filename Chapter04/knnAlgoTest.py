'''
Created on 02-Sep-2017

@author: DX
'''
#Import math for calculations of square roots
import numpy as np

from Chapter_04 import KNN 
dataset = [[5.1,    3.5,    1.4,    0.2,    1],
           [4.9,    3.0,    1.4,    0.2,    1],
           [4.7,    3.2,    1.3,    0.2,    1],
           [4.6,    3.1,    1.5,    0.2,    1],
           [5.0,    3.6,    1.4,    0.2,    1],
           [7.0,    3.2,    4.7,    1.4,    2],
           [6.4,    6.2,    4.5,    1.5,    2],           
           [6.9,    3.1,    4.9,    1.5,    2],
           [5.5,    2.3,    4.0,    1.3,    2],
           [6.5,    2.8,    4.6,    1.5,    2],
           [6.3,    3.3,    6.0,    2.5,    3],
           [5.8,    2.7,    5.1,    1.9,    3],
           [7.1,    3.0,    5.9,    2.1,    3],
           [6.3,    2.9,    5.6,    1.8,    3],
           [6.5,    3.0,    5.8,    2.2,    3]]
 
np.random.shuffle(dataset)
 
#Lets put our test instance.
testInstance=[4.8,3.1,3.0,1.3,1]
 
#Now lets find out 3 neighbors for our test instance using getNeighbor
k = 5
neighbors = KNN.getNeighbors(dataset, testInstance, k)

#Print neighbors
print(neighbors)

#Get the class prediction out of neighbors
prediction = KNN.getPrediction(neighbors) 

#Print predicion
print("Predicted class for the test instance is: %d"%prediction)