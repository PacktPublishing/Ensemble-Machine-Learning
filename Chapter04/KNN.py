#Import math for calculations of square roots
import math
import operator
from random import randrange

#Function to get distance between test instance and training set 
def DistanceMetric(instance1, instance2, isClass=None):
    
    #If Class variable is in the instance
    if isClass:
        length = len(instance1)-1
    else:
        length = len(instance1)
    
    #Initialize variable to store distance
    distance = 0
    
    #Lets run a loop to calculate element wise differences
    for x in range(length):
        
        #Euclidean distance
        distance += pow((instance1[x] - instance2[x]), 2)
        
    return math.sqrt(distance)

#Function to get nearest neighbors 
def getNeighbors(trainingSet, testInstance, k):
   
    #Create a list variable to store distances between test and training instance.
    distances = []
    
    #Get distance between each instance in the training set and the test instance.  
    for x in range(len(trainingSet)):
        
        #As we will going to have class variable in the training set isClass will be true
        dist = DistanceMetric(testInstance, trainingSet[x], isClass=True)
        
        #Append the distance of each instance to the distance list
        distances.append((trainingSet[x], dist))
        
    #Sort the distances in ascending order 
    distances.sort(key=operator.itemgetter(1))
    
    #Create a list to store the neighbors
    neighbors = []
    
    #Run a loop to get k neighbors from the sorted distances. 
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#Function to get prediction
def getPrediction(neighbors):
    
    #Create a dictionary variable to store votes from the neighbors
    #We will use class attribute as the dictionary keys and their occurrence as key value. 
    classVotes = {}
    
    #Go to each neighbor and take the vote for the class
    for x in range(len(neighbors)):
        
        #Get the class value of the neighbor 
        response = neighbors[x][-1]
        
        #Create class key if its not there;
        #If class key is in the dictionary increase it by one.  
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #Sort the dictionary keys on the basis of key values in descending order 
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    
    #Return the key name (class) with the highest value
    return sortedVotes[0][0]
####### KNN Bagging ############

def DistanceMetricBagged(instance1, instance2,n_features):
        
    #Initialize variable to store distance
    distance = 0
    features = list()
    
    #Select random features to apply sub space bagging
    while len(features) < n_features:
        index = randrange(len(instance1)-1)
        if index not in features:
            features.append(index)
            
    #Lets run a loop to calculate element wise differences for the selected features only.
    for x in features:        
        #Euclidean distance
        distance += pow((instance1[x] - instance2[x]), 2)
        
    return math.sqrt(distance)

def getNeighborsBagged(trainingSet, testInstance, k,n_features):
   
    #Create a list variable to store distances between test and training instance.
    distances = []
    
    #Get distance between each instance in the training set and the test instance.  
    for x in range(len(trainingSet)):        
        #As we will going to have class variable in the training set isClass will be true
        dist = DistanceMetricBagged(testInstance, trainingSet[x],n_features)
        
        #Append the distance of each instance to the distance list
        distances.append((trainingSet[x], dist))
        
    #Sort the distances in ascending order 
    distances.sort(key=operator.itemgetter(1))
    
    #Create a list to store the neighbors
    neighbors = []
    
    #Run a loop to get k neighbors from the sorted distances. 
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors