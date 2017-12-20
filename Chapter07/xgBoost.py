'''
Created on 23-Oct-2017

@author: aii32199
'''

# First XGBoost model for Pima Indians dataset

#Load the required libraries
#Numpy for reading the csv file
from numpy import loadtxt

#Import XGBoost classifier 
from xgboost import XGBClassifier

#We will use sklearn to divide our data set into training and test set
from sklearn.model_selection import train_test_split

#We will use sklearn's accuracy metric to evaluate the performance of the trained model
from sklearn.metrics import accuracy_score

#Let's load the dataset into the numpy array
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

#split data into X (input variables)and y(output variable/Class)
X = dataset[:,0:8]
Y = dataset[:,8]

#Create training and test set with 33% data in test set and 66% for the training of the model
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#Train our first model on created training set
model = XGBClassifier()
model.fit(X_train, y_train)

#Lets see the prediction from the trained model
y_pred = model.predict(X_test)

#Create a list of predictions for evaluation purpose
predictions = [round(value) for value in y_pred]

#Evaluate predictions using accuracy metric
accuracy = accuracy_score(y_test, predictions)

#Print the accuracy
print("Accuracy of the trained model is: %.2f%%" % (accuracy * 100.0))

print(model)