'''
Created on 21-Jun-2017

@author: aii32199
'''
import sys

import numpy as np


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    
    #Initialize Gini variable
    gini = 0.0
    
    #Calculate propertion for each class
    for class_value in class_values:
        #Extract groups
        for group in groups:
            #Number of instance in the group
            size = len(group)
            if size == 0:
                continue            
            #Initialize a list to store class index of the instances
            r = []
            #get class of each instance in the group 
            for row in group:
                r.append(row[-1]) 
            #Count number of instances belongs to current class    
            class_count = r.count(class_value)
            #Calculate class proportion
            proportion = class_count/float(size)
            #Calculate Gini index                
            gini += (proportion * (1.0 - proportion))
    return gini

def createSplit(attribute,threshold,dataset):
    
    #Initialize two lists to store the sub sets
    lesser, greater = list(),list()
    
    #Loop through the attribute values and create sub set out of it
    for values in dataset:
        #Apply threshold
        if values[attribute]<threshold:
            lesser.append(values)
        else:
            greater.append(values)
    return lesser,greater                

def get_split(dataset):
    #class_values = list(set(row[-1] for row in dataset))
    class_values = []
    for row in dataset:
        class_values.append(row[-1])
    #initialize variables to store gini score, attribute index and split groups    
    winnerAttribute = sys.maxsize
    attributeValue = sys.maxsize
    gScore = sys.maxsize
    leftGroup = None
    
    #Run loop to access each attribute and attribute values
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = createSplit(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < gScore:
                winnerAttribute, attributeValue, gScore, leftGroup = index, row[index], gini, groups
    #Once done create a dictionary for node 
    node = {'attribute':winnerAttribute,'value':attributeValue,'groups':leftGroup}            
    return  node   

def terminalNode(dataset):
    #Create a vaiable to store the class value and count the class occurance
    classes = []    
    for row in dataset:
        classes.append(row[-1])    
    return max(set(classes), key=classes.count)

data = [[-1.2,0],[-3.2,0],[2.1,1],[1.5,1]]

data1 = [[[-1.2,0],[-3.2,0]],[[2.1,1],[1.5,1]]]
data2 = [[[-1.2,1],[-3.2,0]],[[2.1,1],[1.5,1]]]
classes =  [0, 1]

[lesser,greater] = createSplit(0, 0, data)

print('Group of negative values: ',lesser)
print('Group of positive values: ',greater)
#proportion = [row[-1] for row in group].count(class_value) / float(size)
print("Gini index for Table 1 data set is: %.2f"%gini_index([lesser,greater], classes))
print("Gini index for Table 2 data set is: %.2f"%gini_index(data2, classes))
# df = pd.DataFrame(data)