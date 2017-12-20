import sys
from csv import reader
import numpy as np

#Function to read csv file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Function to create Train and Test set from the original dataset
def getTrainTestData(dataset,split):
    np.random.seed(0)
    training = []
    testing = []    
    
    np.random.shuffle(dataset)
    shape = np.shape(dataset)
    trainlength = np.uint16(np.floor(split*shape[0]))
    
    for i in range(trainlength):    
        training.append(dataset[i])
        
    for i in range(trainlength,shape[0]):    
        testing.append(dataset[i])
    
    return training,testing

#Create splits to validate gini score
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

# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini

#Function to get new node
def getNode(dataset):
    
    class_values = []
    for row in dataset:
        class_values.append(row[-1])
    
    #Extract unique class values present in the dataset    
    class_values = np.unique(np.array(class_values))    
    
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



# Create a terminal node value
def terminalNode(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def buildTree(node, max_depth, min_size, depth):
    #Lets get groups information first.
    left, right = node['groups']
    del(node['groups'])
    # check if there are any element in the left and right group
    if not left or not right:
        #If there is no element in the groups call terminal Node
        combined = left+right
        node['left'] = terminalNode(combined)
        node['right']= terminalNode(combined)
        return
    # check if we have reached to maximum depth
    if depth >= max_depth:
        node['left']=terminalNode(left)
        node['right'] = terminalNode(right)
        return
    # if all okey lest start building tree for left side nodes
    # if minimum instances are done by the node stop further build 
    if len(left) <= min_size:
        node['left'] = terminalNode(left)
        
    else:
        #Create new node under left side of the tree
        node['left'] = getNode(left)        
        #append node under the tree and increase depth by one.
        buildTree(node['left'], max_depth, min_size, depth+1) #recursion will take place in here
        
    
    # Similar procedure for the right side nodes
    if len(right) <= min_size:
        node['right'] = terminalNode(right)
       
    else:
        node['right'] = getNode(right)        
        buildTree(node['right'], max_depth, min_size, depth+1)

     

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = getNode(train)    
    buildTree(root, max_depth, min_size, 1)
    return root

   
# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.2f]' % ((depth*' ', (node['attribute']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

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

#Function to check accuracy of the data set
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#Function to convert string attribute values to float
def str_column_to_float(dataset, column):
    for row in dataset:
        if row[column]=='?':
            row[column] = 0
        else:
            row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
################# Functions for Random Forest ##############
# Build a decision tree
def build_tree_RF(train, max_depth, min_size,nfeatures):
    root = getNodeRF(train,nfeatures)    
    buildTreeRF(root, max_depth, min_size, 1,nfeatures)
    return root

# Create child splits for a node or make terminal
def buildTreeRF(node, max_depth, min_size, depth,nfeatures):
    #Lets get groups information first.
    left, right = node['groups']
    del(node['groups'])
    # check if there are any element in the left and right group
    if not left or not right:
        #If there is no element in the groups call terminal Node
        combined = left+right
        node['left'] = terminalNode(combined)
        node['right']= terminalNode(combined)
        return
    # check if we have reached to maximum depth
    if depth >= max_depth:
        node['left']=terminalNode(left)
        node['right'] = terminalNode(right)
        return
    # if all okey lest start building tree for left side nodes
    # if minimum instances are done by the node stop further build 
    if len(left) <= min_size:
        node['left'] = terminalNode(left)
        
    else:
        #Create new node under left side of the tree
        node['left'] = getNodeRF(left,nfeatures)        
        #append node under the tree and increase depth by one.
        buildTree(node['left'], max_depth, min_size, depth+1) #recursion will take place in here
        
    
    # Similar procedure for the right side nodes
    if len(right) <= min_size:
        node['right'] = terminalNode(right)
       
    else:
        node['right'] = getNodeRF(right,nfeatures)        
        buildTree(node['right'], max_depth, min_size, depth+1)     

# Select the best split point for a dataset
from random import randrange
def getNodeRF(dataset,n_features):
    
    class_values = []
    for row in dataset:
        class_values.append(row[-1])
    
    #Extract unique class values present in the dataset    
    class_values = np.unique(np.array(class_values))    
    
    #Initialize variables to store gini score, attribute index and split groups    
    winnerAttribute = sys.maxsize
    attributeValue = sys.maxsize
    gScore = sys.maxsize
    leftGroup = None
    
    #Select Random features
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    
    #Run loop to access each attribute and attribute values
    for index in features:
        for row in dataset:
            groups = createSplit(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < gScore:
                winnerAttribute, attributeValue, gScore, leftGroup = index, row[index], gini, groups
    #Once done create a dictionary for node 
    node = {'attribute':winnerAttribute,'value':attributeValue,'groups':leftGroup}            
    return  node          

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree_RF(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)

#Create cross validation sets
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores