'''
Created on 02-Sep-2017

@author: DX
'''
from csv import reader
from math import sqrt
from random import seed
from random import randrange
import numpy as np

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

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
    
    return training,testing

# Convert string column to float
def str_column_to_float(dataset, column,length):
    
    #for row in dataset:
    for i in range(length):
        row = dataset[i]        
        if row[column]=='?':
            row[column] = 0
        else:
            row[column] = float(row[column].strip())
    
# Convert string column to integer
def str_column_to_int(dataset, column,length):
    
    class_values=[]
    for i in range(length):
        row = dataset[i]
        class_values.append(row[column]) 
#     class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(length):
        row = dataset[i]  
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def subsample(dataset, n_sample):
    sample = list()
    #n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0