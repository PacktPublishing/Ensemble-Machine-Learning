#Import the required packages

#Import pandas to read csv
import pandas

#Import numpy for array related operations
import numpy

#Import sklearn's PCA algorithm
from sklearn.decomposition import PCA

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
pca = PCA(n_components=3)
fit = pca.fit(X)

#Summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)