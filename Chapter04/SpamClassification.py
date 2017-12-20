'''
Created on 02-Sep-2017

@author: DX
'''

#Import math for calculations of square roots
from Chapter_03.DecisionTree_CART_RF import load_csv, getTrainTestData, accuracy_metric, str_column_to_float 
from Chapter_04 import KNN
import numpy as np


#Read CSV file
dataName = 'spamData.csv'

#Use function load_csv from chapter 3
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

str_column_to_float(dataset, len(dataset[0])-1)

#Split train and test data set using function getTrainTestData
#We will use 80% of the data set as training set and rest for testing
train,test = getTrainTestData(dataset,0.8)

train = np.array(train)
test = np.array(test)

shape = np.shape(train)
xtrain = train[:,0:shape[1]-1]
ytrain = train[:,shape[1]-1] 

xtest = test[:,0:shape[1]-1]
ytest = test[:,shape[1]-1]

#Create empty list to store predictions and actual output
testPredictions=[]
testActual=[]

#Select number of neighbors for each classifier
k = 7

#Select sample size
sample_size = 500

#Select number of random features 
n_features = 20

#Calculate number of classifier on the basis of number of samples.
n_classifier = np.uint8(len(train)/sample_size)

#Get prediction for each test instance and store them into the list 
for i in range(0,len(test)):
    predictions = []
    
    #Run loop for each sample
    for cl in range(1,n_classifier):
        
        #Randomly shuffle training set and create sample out of it 
        np.random.shuffle(train)        
        sample = [train[row] for row in range(sample_size)]
        
        #Pick test instance                        
        test_instance = test[i]
        
        #Get neighbors and prediction on the basis of neighbor            
        neighbors = KNN.getNeighborsBagged(sample, test_instance, k,n_features)
        pred = KNN.getPrediction(neighbors)
        
        #Append prediction against each sample with random features
        predictions.append(pred)
    
    #Get final prediction using majority voting from each classifier    
    fin_pred = max(set(predictions), key=predictions.count)    
    testActual.append(test_instance[-1])
    testPredictions.append(fin_pred)
    print ("Actual: %s   Predicted: %s"%(test_instance[-1],pred))

#Use accurcay_metric function to evaluate our results
accuracy = accuracy_metric(testActual,testPredictions)

#Print accuracy 
print("Accuracy of the classification: %0.2f"%accuracy)