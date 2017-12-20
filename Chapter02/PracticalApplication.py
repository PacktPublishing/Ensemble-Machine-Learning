import pandas as pd
import pprint
import json
import numpy as np 
np.random.seed(1337) # for reproducibility
from Chapter_02 import DecisionTree_ID3 as DT

datapath = 'E:/PyDevWorkSpaceTest/Ensembles/Chapter_02/Data/CarDataset.csv'
path2save =  'E:/PyDevWorkSpaceTest/Ensembles/Chapter_02/Data/TreeModel.json'
trainDataPath = 'E:/PyDevWorkSpaceTest/Ensembles/Chapter_02/Data/trainData.csv'
testDataPath = 'E:/PyDevWorkSpaceTest/Ensembles/Chapter_02/Data/testData.csv'
  
# testData = pd.read_csv(testDataPath)

cardata = pd.read_csv(datapath)
mat = cardata.as_matrix()
df = pd.DataFrame(mat,columns=['buying','maint','doors','persons','lug_boot','safety','Class']) 
trainData,testData = DT.split_data(df, 0.995)
  
trainData.to_csv(trainDataPath,columns=['buying','maint','doors','persons','lug_boot','safety','Class'])
testData.to_csv(testDataPath,columns=['buying','maint','doors','persons','lug_boot','safety','Class'])
     
tree = DT.buildTree(trainData)
pprint.pprint(tree)
  
with open(path2save,'w') as f:
    json.dump(tree,f)
 
with open(path2save) as f:
    model = json.load(f)
  
pprint.pprint(model)
actualClass = testData['Class']
predictions = DT.BatchTest(testData, model)
accuracy,match = DT.getAccuracy(actualClass, predictions)
 
print("Accuracy of the model is: %.2f and matched results are %i out of %i"%(accuracy,match,len(actualClass))) 



