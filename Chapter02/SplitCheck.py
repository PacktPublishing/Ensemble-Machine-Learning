'''
Created on Jun 24, 2017

@author: DX
'''

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini
 
# Select the best split point for a dataset

def get_split(dataset):
    
    class_values = extractClasses(dataset)
    
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
def extractClasses(dataset):
    
    class_values = []
    
    for rows in dataset:
        class_values.append(rows[-1]) 
     
    return class_values
 
dataset = [[0.50000,  1.50000,  1.00000],
           [1.00000,  0.50000, -1.00000],
           [1.25000,  3.50000,  1.00000],
           [1.50000,  4.00000,  1.00000],
           [2.00000,  2.00000, -1.00000],
           [2.50000,  2.50000,  1.00000],
           [3.75000,  3.00000, -1.00000],
           [4.00000,  1.00000, -1.00000]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
