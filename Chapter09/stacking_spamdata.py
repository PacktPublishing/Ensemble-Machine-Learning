'''
Created on 04-Nov-2017

@author: DX
'''
from Chapter_03.DecisionTree_CART_RF import load_csv,cross_validation_split,str_column_to_float

#Import numpy for array based operations 
import numpy as np

#Import Support vector machine
from sklearn.svm.classes import SVC

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

#KNN
from sklearn.neighbors import KNeighborsClassifier

#Logistic Regression
from sklearn.linear_model import LogisticRegression

#Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier

#Ada-boost Classifier
from sklearn.ensemble import AdaBoostClassifier

#Set Random seed
np.random.seed(1)

#Convert string variables numerical
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

#Stacking the predictions from the models
def stacking(dataset,models):
    
    stackedData = []
    
    for model in models:
        pred = model.predict(dataset)
        stackedData.append(pred)
    
    return np.transpose(stackedData)

#Train the models
def stack_fit(model,x,y):
    return model.fit(x,y)
    
#Function to evaluate model performance 
def getAccuracy(pre,ytest):
    count = 0
    for i in range(len(ytest)):
        if ytest[i]==pre[i]:
            count+=1
     
    acc = float(count)/len(ytest)
    return acc

#Separate the input and output variable 
def getXY(dataset):
    dataset = np.array(dataset)
    shape = np.shape(dataset)
    X = dataset[:,0:shape[1]-1]
    Y = dataset[:,shape[1]-1]
    return X,Y

#Specify the file name
dataName = 'spamData.csv'

#Use function load_csv 
dataset = load_csv(dataName)

#Create an empty list to store the data set
dataset_new = []

#We will remove incomplete instance from the data set
for i in range(len(dataset)-1):
    dataset_new.append(dataset[i])
dataset = dataset_new

#Use function str_column_to_float from chapter 3 to convert string values to float
for i in range(0, len(dataset[0])-1):
    str_column_to_float(dataset, i)

#Convert class variable to the numerical value
str_column_to_int(dataset, len(dataset[0])-1)

#Shuffle the data set
np.random.shuffle(dataset)

#Load all the classifiers
clf1 = AdaBoostClassifier()#SVC(kernel='rbf')
clf2 = DecisionTreeClassifier(max_depth=25)
clf3 = KNeighborsClassifier(n_neighbors=1)
clf4 = RandomForestClassifier(n_estimators=25,max_depth=15)
clf5 = LogisticRegression()
clf6 = SVC(kernel='rbf')

#Stack all the classifier
models = [clf1,clf2,clf3,clf4,clf5]

#Create the sample out of data sets
splits = cross_validation_split(dataset,len(models))

#Initialize the variable for trained classifier
trained =[]

#Train the model and add to the stack
for i in range(len(models)):
    model = models[i]
    x,y = getXY(splits[i])    
    trained.append(stack_fit(model, x, y))

#Create test data from left split
xtest,ytest = getXY(splits[len(models)-1])

#Generate the stacked predictions
stackedData = stacking(xtest, trained)

#Here we will calculate individual accuracies of models
for i in range(np.shape(stackedData)[1]):
    acc = getAccuracy(stackedData[:,i], ytest)
    print("Accuracy of model %i is %.2f"%(i,(100*acc)))

#Take the vote of each classifier and create final prediction
predLr =[np.bincount(np.array(pred,dtype="int64")).argmax() for pred in stackedData]

#Evaluate the stacked model performance 
accLr  = getAccuracy(ytest, predLr)
print("\nAccuracy of stacking is %.2f"%(100*accLr))