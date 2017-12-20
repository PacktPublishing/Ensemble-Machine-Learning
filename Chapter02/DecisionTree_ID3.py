import numpy as np
import pandas as pd

np.random.seed(1337) # for reproducibility

#Function to get Information Gain of the attribute using class entropy
def getInformationGain(subtable,classEntropy):
             
    #Initialize a variable for storing probability of Classes
    fraction = 0

    #Calculate total number of instances
    denom = np.sum(np.sum(subtable))
    
    #Initialize variable for storing total entropies of attrribute values  
    EntropyAtt = 0
     
    #Now we will run a loop to access each attribute and its information gain 
    for key in subtable.keys():
    
        #Extract Attribute
        attribute = subtable[key]
        entropy = 0    #Initialize variable for entropy calculation
        coeff = 0      #Initialize variable to store coefficient
        
        #Find out sum of class attributes(in our case Yes and No)
        denom2 = np.sum(attribute)
        
        #Run a loop to get entropy of distinct values of attribute
        for value in attribute:
        
            #Calculate coeff
            coeff+= float(value)/denom 
            
            #Calculate probability of the attribute value
            fraction = float(value)/denom2
            
            #Calculate Entropy
            eps = np.finfo(float).eps            
            entropy+= -fraction*np.log2(fraction+eps)
        EntropyAtt+= coeff*entropy
    
    #Calculate Information Gain using class entropy
    InfGain = classEntropy - EntropyAtt
    return InfGain,EntropyAtt
    
def getClassEntropy(classAttributes):
    
    #Get distinct classes and how many time they occure 
    _,counts = np.unique(classAttributes,return_counts=True)
    denom = len(classAttributes)
    entropy = 0 #Initialize entropy variable
    
    #Run a loop to calculate entropy of dataset
    for count in counts:
        fraction = float(count)/denom
        entropy+= -fraction*np.log2(fraction)
    return entropy


def getHistTable(df,attribute):
    #This function create a subtable for the given attribute
    #Get values for the attribute
    value = df[attribute]
    
    #Extract class
    classes = df['Class']
    
    #Get distinct classes
    classunique = df['Class'].unique()
    
    #Get distinct values from attribute e.g. Low, High and Med for Salary
    valunique = df[attribute].unique()
    
    #Create an empty table to store attribute value and their respective class occurance
    temp = np.zeros((len(classunique),len(valunique)),dtype='uint8')    
    subtable = pd.DataFrame(temp,index=classunique,columns=valunique)
    
    #Calculate class occurance for each value for Med salary how many time class attribute is Yes
    for i in range(len(classes)):    
        subtable[value[i]][classes[i]]+= 1
    
    return subtable

def getNode(df):
    #This function is written for getting winner attribute to assign node    
    
    #Get Classes
    classAttributes = df['Class']
    
    #Create empty list to store Information gain for respected attributes
    InformationGain = []
    AttributeName = []
    
    #Extract each attribute 
    for attribute in df.keys():
        if attribute is not 'Class':
            #Get class occurance for each attribute value
            subtable = getHistTable(df,attribute)
            
            #Get class entropy of the data
            Ec = getClassEntropy(classAttributes)
            
            #Calculate Information Gain for each attribute 
            InfoGain,EntropyAtt = getInformationGain(subtable, Ec)
            
            #Append the value into the list
            InformationGain.append(InfoGain)
            AttributeName.append(attribute)
            #print("Information Gain for %s: %.2f and Entropy: %.2f"%(attribute,InfoGain,EntropyAtt))
    
    #Find out attribute with maximum information gain
    indx = np.argmax(InformationGain)    
    winnerNode = AttributeName[indx]
    #print("\nWinner attrbute is: %s"%(winnerNode))
        
    return winnerNode

def getSubtable(df,node,atValues):
    #This function is written to get subtable for given attribute values(such as table for those persons whose salary is Medium)
    subtable = []

    #run a loop through the dataset and create subtable
    for i in range(len(df[node])):
        if df[node][i]==atValues:
            row = df.loc[i,df.keys()]
            subtable.append(row)
        
    for c in range(len(df.keys())):
        if df.keys()[c]==node:            
            break;        
                
    #Create a new dataframe 
    subtable = pd.DataFrame(subtable,index=range(len(subtable)))
    #print(subtable)
    return subtable

def buildTree(df,tree=None):    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = getNode(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
    #Loop below is written for building tree using recursion of the function,
    #We will create subtable of each attribute value and try to find whether it have a pure subset or not,
    #if it is a pure subset we will stop tree growing for that node. if it is not a pure set then we will.. 
    #again call the same function.
    for value in attValue:
        
        #print("Value: %s"%value)
        subtable = getSubtable(df,node,value)
        clValue,counts = np.unique(subtable['Class'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            #print("Class: %s\n"%clValue)
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable)#Recursion of the function
                   
    return tree
                        
def predict(inst,tree):
    #This function will predict an input instace's class using given tree
    
    #We will use recursion to traverse through the tree same as we have done in case
    #of tree building
    
    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction

def preProcess(dataset):
    #Create a dataframe out of our dataset with attribute names
    df = pd.DataFrame(dataset,columns=['Name','Salary','Sex','Marital','Class'])
    
    #Remove name attribute as it is not required for the calculations
    df.pop('Name')
    
    #Make sure last attribute of our data set must be Class attribute
    cols = list(df)
    cols.insert(len(cols), cols.pop(cols.index('Class')))
    df = df.ix[:,cols]
    print(df)
    
    return df

def BatchTest(instances,tree):
    
    prediction = []
    instances.pop("Class")
    for i in range(len(instances.index)):        
        inst = instances.ix[i] 
        pred = predict(inst, tree)
        prediction.append(pred)
    return prediction

def split_data(df,percentage):
    
    split_indx = np.int32(np.floor(percentage*len(df.index)))    
    #We will shuffle the rows of data to mix out its well 
    df = df.sample(frac=1).reset_index(drop=True)
    
    #split training data for creating tree 
    train_data = df[:split_indx]    
    temp = df[split_indx:len(df.index)]
    temp = temp.as_matrix()
    test_data = pd.DataFrame(temp,index=range(len(temp)),columns=[key for key in df.keys()])    
    
    return train_data,test_data

def getAccuracy(testClass,predictedClass):
    
    match = 0
    for i in range(len(testClass)):
        if testClass[i]==predictedClass[i]:
            match+=1
    
    accuracy = 100*match/len(testClass)
    
    return accuracy,match