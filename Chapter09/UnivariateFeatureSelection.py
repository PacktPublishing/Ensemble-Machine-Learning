'''
Created on 02-Nov-2017

@author: aii32199
'''
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

#Import the required packages

#Import pandas to read csv
import pandas

#Import numpy for array related operations
import numpy

#Import sklearn's feature selection algorithm
from sklearn.feature_selection import SelectKBest

#Import chi2 for performing chi square test
from sklearn.feature_selection import chi2

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

#We will select the features using chi square
test = SelectKBest(score_func=chi2, k=4)

#Fit the function for ranking the features by score
fit = test.fit(X, Y)

#Summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)

#Apply the transformation on to data set 
features = fit.transform(X)

#Summarize selected features
print(features[0:5,:])