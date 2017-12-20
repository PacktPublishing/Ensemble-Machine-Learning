'''
Created on 03-Oct-2017
 
@author: DX
'''
import pprint 
from Chapter_06 import RegressionTrees as rg
import matplotlib.pyplot as plt
import numpy as np
 
#Create a Sine wave for demonstration of non-linearity

#Set the number of samples  
N = 256

#Create time value
ix = np.arange(N)

#Create the sine wave using the formula sin(2*pi*f)
signal = np.sin(2*np.pi*ix/float(N/2))

#Combine both time and amplitude
dataset = range(0,N)
dataset = np.c_[ix,signal]
dataset_ = dataset.copy()  

#Call Gradient boost
weaks = rg.GradientBoost(dataset,5,1,100) 
 
prediction=[]
actual = []
  
#Run a loop to extract each instance from the data set
for row in dataset_:
      
    #Create a list to store predictions from different ckassifier for the test instance
    preds = []
      
    #Feed the instance to different classifiers
    for i in range(len(weaks)):
          
        #Multiply the predicted ouput with the alpha value of the classifier
        p = rg.predict(weaks[i], row)
          
        #Add the weighted prediction to the list 
        preds.append(p)
      
    #Sum up output of all the classifiers and take their sign as the prediction
    final = (sum(preds))
      
    #Append the final output to the prediction list and actual ouput to the actual list    
    prediction.append(final)
    actual.append(row[-1])

#Append the error of the current configuration
_,mse = rg.getResidual(actual, prediction)        
      
    
#Lets Plot the error in each configuration    
plt.figure()
plt.plot(ix,signal,marker='*',markersize=8)
plt.plot(ix,prediction,marker='+',markersize=8)
plt.show()