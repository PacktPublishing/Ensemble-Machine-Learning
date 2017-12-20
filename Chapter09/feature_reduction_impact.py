'''
Created on 31-Oct-2017

@author: DX
'''

#Import Sklearn Datasets of IRIS flower classification
import sklearn.datasets as datasets

#Import Pandas library to create data frame from the data
import pandas as pd

#Load the data set
iris=datasets.load_iris()

#Extract data part from the data set
data = iris.data

#Select dimension of data
data = data[:,2:4]

#Load data set into the data frame
df=pd.DataFrame(data)

#Extract target variable from the data set
y=iris.target

#Import decision tree classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

#We will create a tree with maximum depth of 5, other parameters will be default
dtree=DecisionTreeClassifier(max_depth=5)

#Train the classifier
dtree.fit(df,y)

#Import graphwiz from sklearn to create the graph out of tree
from sklearn.tree import export_graphviz

#We will use StringIO to create graph with all characters
from sklearn.externals.six import StringIO
dot_data = StringIO()

#Import pydotplus to create tree as a graph and store it on the disk
import pydotplus

#Create Graph out of tree and store it on the disk
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("graph_feat_4.png")