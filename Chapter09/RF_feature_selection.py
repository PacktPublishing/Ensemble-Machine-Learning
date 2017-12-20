'''
Created on 28-Oct-2017

@author: DX
'''
#Import the supporting libraries

#Import pandas to load the data set from csv file
from pandas import read_csv

#Import numpy for array based operations and calculations 
import numpy as np 

#Import Random Forest classifier class from sklearn
from sklearn.ensemble import RandomForestClassifier

#Import feature selector class select model of sklearn
from sklearn.feature_selection import SelectFromModel

np.random.seed(1)

#Function to create Train and Test set from the original data set
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
    training = np.array(training)
    testing = np.array(testing)
    return training,testing

#Function to evaluate model performance 
def getAccuracy(pre,ytest):
    count = 0
    for i in range(len(ytest)):
        if ytest[i]==pre[i]:
            count+=1
     
    acc = float(count)/len(ytest)
    return acc


#Load data set as pandas data frame
data = read_csv('train.csv')

#Extract attribute names from the data frame
feat = data.keys()
feat_labels = feat.get_values()

#Extract data values from the data frame
dataset = data.values

#Shuffle the data set
np.random.shuffle(dataset)

#We will select 10000 instances to train the classifier
inst = 50000

#Extract 10000 instances from the data set 
dataset = dataset[0:inst,:]

#Create Training and Testing data for performance evaluation
train,test = getTrainTestData(dataset, 0.7)

#Split data into input and output variable with selected features
Xtrain = train[:,0:94]
ytrain = train[:,94]

shape = np.shape(Xtrain)
print("Shape of the data set ",shape)
#Print the size of Data in MBs
print("Size of Data set before feature selection: %.2f MB"%(Xtrain.nbytes/1e6))

#Lets select the test data for model evaluation purpose
Xtest = test[:,0:94]
ytest = test[:,94]

#Create a random forest classifier with following Parameters
trees      = 250
max_feat   = 7   
max_depth  = 30
min_sample = 2

clf = RandomForestClassifier(n_estimators=trees,
                             max_features=max_feat,
                             max_depth=max_depth,
                             min_samples_split= min_sample,
                             random_state=0, 
                             n_jobs=-1)

#Train the classifier and calculate the training time
import time
start = time.time()
clf.fit(Xtrain, ytrain)
end = time.time()

#Lets Note down the model training time
print("Execution time for building the Tree is: %f"%(float(end)-float(start)))
pre = clf.predict(Xtest)

#Evaluate the model performance for the test data
acc = getAccuracy(pre, ytest)
print("Accuracy of model before feature selection is %.2f"%(100*acc)) 

#Once we have trained the model we will rank all the features

for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)

#Select features which have higher contribution in the final prediction 
sfm = SelectFromModel(clf, threshold=0.01)
sfm.fit(Xtrain,ytrain)

#Transform input data set  
Xtrain_1 = sfm.transform(Xtrain)
Xtest_1  = sfm.transform(Xtest)

#Let's see the size and shape of new data set  
print("Size of Data set before featre selection: %.2f MB"%(Xtrain_1.nbytes/1e6))
shape = np.shape(Xtrain_1)
print("Shape of the data set ",shape)

#Model training time
start = time.time()
clf.fit(Xtrain_1, ytrain)
end = time.time()
print("Execution time for building the Tree is: %f"%(float(end)-float(start)))

#Let's evaluate the model on test data
pre = clf.predict(Xtest_1)
count = 0
 
acc2 = getAccuracy(pre, ytest) 

print("accuracy after feature selection %.2f"%(100*acc2))