'''
Created on May 20, 2017

@author: DX
'''
# Import All the required packages from sklearn
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

#Load data 
iris = load_iris()
X = iris.data
Y = iris.target

#Split data in training and testing set 
X_fit, X_eval, y_fit, y_test= model_selection.train_test_split( X, Y, test_size=0.30, random_state=1 )

#Create random sub sample to train multiple models
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

#Define a decision tree classifier
cart = DecisionTreeClassifier()
num_trees = 100

#Create classification model for bagging
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

#Train different models and print their accuracy
results = model_selection.cross_val_score(model, X_fit, y_fit,cv=kfold)
for i in range(len(results)):
    print("Model: "+str(i)+" Accuracy is: "+str(results[i]))
    
print("Mean Accuracy is: "+str(results.mean()))

model.fit(X_fit, y_fit)
pred_label = model.predict(X_eval)
nnz = np.shape(y_test)[0] - np.count_nonzero(pred_label - y_test)
acc = 100*nnz/np.shape(y_test)[0]
print('accuracy is: '+str(acc))