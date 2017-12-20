'''
Created on 04-Nov-2017

@author: DX
'''
#We will use sklearns make circle to create the data 
from sklearn.datasets import make_circles

#Numpy will help us for array related operations
import numpy as np

#We will use pylab for visualization of plots
import pylab as pl

#Import our SVM classifier from sklearn
from sklearn.svm import SVC

#Generate the data set using make_circle function
X, Y = make_circles(n_samples=800, noise=0.07, factor=0.4)

#Let's Plot the Point and see 
# print "...Showing dataset in new window..."
pl.figure(figsize=(10, 8))
pl.subplot(111)
pl.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
# pl.show()

#Kernel to convert sub space of data
def fn_kernel(x1, x2):
    
    # Implements a kernel phi(x1,y1) = [x1, y1, x1^2 + y1^2]
    return np.array([x1, x2, x1**2.0 + x2**2.0])

#Create a list to store transformed points
transformed = []

#Transform each point to the new sub space
for points in X:
    transformed.append(fn_kernel(points[0], points[1]))
transformed = np.array(transformed)

#We will 3D plots to visualize data in higher dimension
from mpl_toolkits.mplot3d import Axes3D

#Import matplotlib to plot the data
import matplotlib.pyplot as plt

#Let's plot the original data first
fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(121)
ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title("Data in 2D (Non-separable)")

#Here we will plot the transformed data
ax = fig.add_subplot(122, projection='3d')
ax.scatter(transformed[:, 0], transformed[:, 1],transformed[:, 2], marker='o', c=Y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title("Data in 3D (separable)")

#Finally show all the plots
plt.show()

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

#Let's merge input and output variable to create train and test data 
dataset = np.c_[X,Y]

#We will use our train and test split function
train,test = getTrainTestData(dataset, 0.7)

#Extract training input and output
x_train = train[:,0:2]
y_train = train[:,2]

#Extract testing input and output
x_test = test[:,0:2]
y_test = test[:,2]

#First we will train our classifier with linear kernel
clf = SVC(kernel='linear')
clf.fit(x_train,y_train)

#Predict the output on test set
pred = clf.predict(x_test)
acc = getAccuracy(pred, y_test)
print("Accuracy of the classifier with linear kernel is %.2f"%(100*acc))

#Now we will train our classifier with RBF kernel
clf = SVC(kernel='rbf',C=3.0)
clf.fit(x_train,y_train)

#Predict the output on test set
pred = clf.predict(x_test)
acc = getAccuracy(pred, y_test)
print("Accuracy of the classifier with rbf kernel is %.2f"%(100*acc))

    
