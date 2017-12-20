'''
Created on 19-Jun-2017

@author: aii32199
''' 
import numpy as np
def getNewNode(data):
    node = {'data':[],'left':[],'right':[]}
    node['data'] = data   
    print(node) 
    return node

def createBinaryTree(tree,data):

#Check whether we have any node in the tree if not create one    
    if not tree:
        tree = getNewNode(data)

    #Now if current value is less than parent node put it in left     
    elif data<=tree['data']:
        tree['left'] = createBinaryTree(tree['left'],data) 
    #else put it in right 
    else:
        tree['right'] = createBinaryTree(tree['right'],data)                          
    return tree


# data = [0.7,0.65,0.83,0.54,0.9,0.11,0.44,0.35,0.75,0.3,0.78,0.15]
data = [0.7,0.65,0.83,0.54,0.9,0.11,0.44,0.35,0.75,0.3,0.78,0.15]
med = np.median(data)
print("Median of array is: %.2f"%med)

tree = []
tree = createBinaryTree(tree,med)
for i in range(len(data)):    
    value = data[i]    
    tree = createBinaryTree(tree,value)
    
import pprint
pprint.pprint(tree)
