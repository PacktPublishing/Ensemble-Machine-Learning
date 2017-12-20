'''
Created on 18-Sep-2017

@author: DX
'''

import numpy as np; 
from Chapter_05 import Adaboost as ad

dataset = [[0.25000,  1.75000,  1.00000],
           [1.25000,  1.75000, -1.00000],
           [0.50000,  1.50000,  1.00000],
           [1.00000,  0.50000, -1.00000],
           [1.25000,  3.50000,  1.00000],
           [1.50000,  4.00000,  1.00000],
           [2.00000,  2.00000, -1.00000],
           [2.50000,  2.50000,  1.00000],
           [3.75000,  3.00000, -1.00000],
           [4.00000,  1.00000, -1.00000]]

[weaks,alphas] = ad.AdaBoostAlgorithm(dataset,9)

prediction=[]
actual = []
for row in dataset:
    preds = []
    for i in range(len(weaks)):
        p = alphas[i]*ad.predict(weaks[i], row)
        #p = predict(weaks[i], row)
        preds.append(p)
    final = np.sign(sum(preds))
    #final = max(set(preds), key=preds.count)    
    prediction.append(final)
    actual.append(row[-1])
    print('Expected=%d, Got=%d' % (row[-1], final)) 

acc = ad.accuracy_metric(actual, prediction)
print("accuracy: %.2f"%acc)