'''
Created on May 22, 2017

@author: DX
'''
# Import All the required packages from sklearn
from sklearn import model_selection
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier  # Boosting Algorithm
from sklearn.tree import DecisionTreeClassifier

import numpy as np


#Load data 
iris = load_iris()
X = iris.data
Y = iris.target

#Split data in training and testing set 
X_fit, X_eval, y_fit, y_test= model_selection.train_test_split( X, Y, test_size=0.20, random_state=1 )

#Define a decision tree classifier
cart = DecisionTreeClassifier()
num_trees = 25

#Create classification model for bagging
model = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, learning_rate = 0.1)

#Train Classification model
model.fit(X_fit, y_fit)

#Test trained model over test set
pred_label = model.predict(X_eval)
nnz = np.float(np.shape(y_test)[0] - np.count_nonzero(pred_label - y_test))
acc = 100*nnz/np.shape(y_test)[0]

#Print accuracy of the model
print('accuracy is: '+str(acc))