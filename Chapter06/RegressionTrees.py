'''
Created on 03-Oct-2017

@author: DX
'''
import numpy as np
import sys

def terminalNodeReg(group):
    
    #Get all the target labels into the List
    class_values = [row[-1] for row in group]
    
    #Return the Mean value of the list 
    return np.mean(class_values)

# Calculate the SSE index for a split dataset
def SquaredError(groups):
    
    #Initialize the variable for SSE
    sse = 0.0 
    
    #Iterate for both the groups   
    for group in groups:
        size = len(group)
        
        #If length is 0 continue for the next group
        if size == 0:
            continue 
        
        #Take all the class values into a list
        class_values = [row[-1] for row in group]
        
        #Calculate SSE for the group       
        sse += np.sum((class_values-np.mean(class_values))**2)
    return sse

#Function to get new node
def getNode(dataset):
    
    #initialize variables to store error score, attribute index and split groups    
    winnerAttribute = sys.maxsize
    attributeValue = sys.maxsize
    errorScore = sys.maxsize
    leftGroup = None
    
    #Run loop to access each attribute and attribute values
    for index in range(len(dataset[0])-1):
        for row in dataset:
            
            #Get split for the attribute value
            groups = createSplit(index, row[index], dataset)
            
            #Calculate SSE for the group
            sse = SquaredError(groups)
            #print("SSE for the attribute %.2f's value %.2f is %.3f"%(index+1,row[index],sse))
            #If SSE is less than previous attribute's SSE return attribute value as Node
            if sse < errorScore:
                winnerAttribute, attributeValue, errorScore, leftGroup = index, row[index], sse, groups
                
    #Once done create a dictionary for node 
    node = {'attribute':winnerAttribute,'value':attributeValue,'groups':leftGroup}            
    return  node

#Create splits to test for node values
def createSplit(attribute,threshold,dataset):
    
    #Initialize two lists to store the sub sets
    lesser, greater = list(),list()
    
    #Loop through the attribute values and create sub set out of it
    for values in dataset:
        #Apply threshold
        if values[attribute]<=threshold:
            lesser.append(values)
        else:
            greater.append(values)
    return lesser,greater        

# Create child splits for a node or make terminal
def buildTreeReg(node, max_depth, min_size, depth):
    #Lets get groups information first.
    left, right = node['groups']
    del(node['groups'])
    # check if there are any element in the left and right group
    if not left or not right:
        #If there is no element in the groups call terminal Node
        combined = left+right
        node['left'] = terminalNodeReg(combined)
        node['right']= terminalNodeReg(combined)
        return
    # check if we have reached to maximum depth
    if depth >= max_depth:
        node['left']=terminalNodeReg(left)
        node['right'] = terminalNodeReg(right)
        return
    # if all okey lest start building tree for left side nodes
    # if minimum instances are done by the node stop further build 
    if len(left) <= min_size:
        node['left'] = terminalNodeReg(left)
        
    else:
        #Create new node under left side of the tree
        node['left'] = getNode(left)        
        #append node under the tree and increase depth by one.
        buildTreeReg(node['left'], max_depth, min_size, depth+1) #recursion will take place in here
        
    
    # Similar procedure for the right side nodes
    if len(right) <= min_size:
        node['right'] = terminalNodeReg(right)
       
    else:
        node['right'] = getNode(right)        
        buildTreeReg(node['right'], max_depth, min_size, depth+1)

    
# Build a decision tree
def build_tree(train, max_depth, min_size):
    
    #Add the root node to the tree
    root = getNode(train)    
    
    #Start building the from the root's branches tree 
    buildTreeReg(root, max_depth, min_size, 1)
    return root

#Function to get prediction from input tree
def predict(node, row):
    
    #Get the node value and check whether the attribute value is less than or equal.  
    if row[node['attribute']] <= node['value']:
        #If yes enter into left branch and check whether it has another node or the class value.
        if isinstance(node['left'], dict):            
            return predict(node['left'], row)#Recursion
        else:
            #If there is no node in the branch 
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def getResidual(actual,pred):
    
    #Create an empty list to store individual error of the instances
    residual = []
    
    # Run a loop to get difference between output and prediction of each instance 
    for i in range(len(actual)):
        
        #Get the difference and add the difference to the list of residuals
        diff = (actual[i]-pred[i])
        residual.append(diff)
    
    #Calculate the Sum of squared error between output and prediction
    mse = np.sum(np.array(residual)**2)
    return residual,mse

def GradientBoost(dataset,depth,mincount,iterations):
    
    dataset = np.array(dataset)
    
    #Create a list to add weak learners(decision stumps)
    weaks = [] 
    
    #Lets run the loop for number of iteration(number of classifiers)
    for itr in range(iterations):
                
        #Create decision tree from the data-set 
        ds = build_tree(dataset,depth,mincount)
        
        #Create a list to store the predictions of the decision stump    
        pred=[]
        
        #Create a list to store actual outputs
        actual = []
        
        #Let's predict output for each instance in the data set
        for row in dataset:
            actual.append(row[-1])
            pred.append(predict(ds, row))

        #Here we will find out difference between predicted and actual output
        residuals,error = getResidual(actual, pred)
        
        #Print the error status 
        print("\nClassifier %i error is %.5f"%(itr,error))
        
        #Check for the convergence
        if error<=0.00001:
            break   
        
        #Replace the previous labels with the current differences(Residuals)
        dataset[:,-1] = residuals
        
        #Append the weak learner to the list
        weaks.append(ds)
        
    return weaks

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0