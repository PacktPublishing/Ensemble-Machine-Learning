'''
Created on 02-Nov-2017

@author: aii32199
'''

#Import the required packages

#Import pandas to read csv
import pandas

#Import numpy for array related operations
import numpy

#Import sklearn's feature selection algorithm
from sklearn.feature_selection import RFE

#Import LogisticRegression for performing chi square test
from sklearn.linear_model import LogisticRegression

#URL for loading the data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

#Define the attribute names
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#Create pandas data frame by loading the data from URL
dataframe = pandas.read_csv(url, names=names)

#Create array from data values 
array = dataframe.values

#Split the data into input and target
X = array[:,0:8]
Y = array[:,8]

#Feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Num Features: %d"% fit.n_features_)
print("Selected Features: %s"% fit.support_)
print("Feature Ranking: %s"% fit.ranking_)
