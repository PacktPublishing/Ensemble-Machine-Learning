'''
Created on 27-Oct-2017

@author: aii32199
'''
# Test stacking on the sonar dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import exp
 
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
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
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
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
 
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
 
# Locate neighbors for a new row
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
 
# Make a prediction with kNN
def knn_predict(model, test_row, num_neighbors=2):
    neighbors = get_neighbors(model, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
 
# Prepare the kNN model
def knn_model(train):
    return train
 
# Make a prediction with weights
def perceptron_predict(weights,row):
    #Row is the input instance
    
    #We will consider first weight as the bias for simplyfied the calculations
    activation = weights[0]
    
    #Now run a loop to multiply each attribute value of the instance with the weight 
    #And add the result to the activation of previous attribute
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    
    #Here we will return 1 if activation is a non negative value and zero in other case  
    return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def perceptron_model(train, l_rate=0.01, n_epoch=5000):
    
    #Lets initialize the weights by 0
    weights = [0.0 for i in range(len(train[0]))]
    
    #We will update the weights for given number of epoch
    for epoch in range(n_epoch):
        
        #Extract each row from the training set
        for row in train:
            
            #Predict the value for the instance
            prediction = perceptron_predict(weights,row)
            
            #Calculate the difference(gradient) between actual and predicted value 
            error = row[-1] - prediction
            
            #Update the bias value using given learning rate and error
            weights[0] = weights[0] + l_rate * error
            
            #Update the weights for each attribute using learning rate 
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            
    #Return the updated weights and biases
    return weights
 
# Make a prediction with coefficients
def logistic_regression_predict(model, row):
    
    #First weight of the model will be bias similar as Perceptron function     
    yhat = model[0]
    
    #We will run a loop to multiply each attribute value with the corresponding weights
    #This is similar to activation calculation in perceptron algorithm
    for i in range(len(row)-1):
        yhat += model[i + 1] * row[i]
    
    #Here we will apply logistic function on the linear combination of weights and attributes
    #This is the place where linear and logistic regression differs  
    return 1.0 / (1.0 + exp(-yhat))
 
# Estimate logistic regression coefficients using stochastic gradient descent
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
    
    #Initialize the weights with the zero values
    coef = [0.0 for i in range(len(train[0]))]
    
    #Repeat the procedure for given number of epochs
    for epoch in range(n_epoch):
        
        #Get prediction for each row and update weights based on error value
        for row in train:
            
            #Predict y for the given x
            yhat = logistic_regression_predict(coef, row)
            
            #Get the error value (gradient/slope/change)
            error = row[-1] - yhat
            
            #Apply gradient descent here to update the weights and biases
            #Update Bias first            
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            
            #Now update the Weights 
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    #Return the trained weights and biases
    return coef
 
# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
    
    #Let's Create an empty list to store predictions from sub models
    stacked_row = list()
    
    #Run a loop to fetch stored models in the List
    for i in range(len(models)):
        
        #Start prediction for each row by each model
        prediction = predict_list[i](models[i], row)
        
        #Store the prediction in the list
        stacked_row.append(prediction)
    
    #Append class values to the new row
    stacked_row.append(row[-1])
    
    #Extend the old row aby adding stacked row 
    return row[0:len(row)-1] + stacked_row
 
# Stacked Generalization Algorithm
def stacking(train, test):
    
    #Let's define the sub model first
    model_list = [knn_model, perceptron_model]
    
    #We will create a prediction list to create new row
    predict_list = [knn_predict, perceptron_predict]
    
    #Create an empty list to store the trained models
    models = list()
    
    #Lets train each sub model individually on the dataset
    for i in range(len(model_list)):        
        model = model_list[i](train)
        models.append(model)
    
    #Create a new stacked data set from prediction of sub models
    stacked_dataset = list()
    for row in train:
        
        #Get new row 
        stacked_row = to_stacked_row(models, predict_list, row)
        
        #Append it to new dataset
        stacked_dataset.append(stacked_row)
    
    #We will train our final classifier on the stacked dataset
    stacked_model = logistic_regression_model(stacked_dataset)
    
    #lets create a list of prediction of the stacked output
    predictions = list()
    
    #Here we will combine all the classifier together to make stack of classifiers
    for row in test:
        
        #Get new row from prediction of sub models
        stacked_row = to_stacked_row(models, predict_list, row)
        
        #Append new row to the new dataset
        stacked_dataset.append(stacked_row)
        
        #Classify the new row using final classifier
        prediction = logistic_regression_predict(stacked_model, stacked_row)
        
        #As final classifier gives a continuous value round it to nearest integer
        prediction = round(prediction)
        
        #Append the prediction to the final list of predictions
        predictions.append(prediction)
    return predictions
 
# Test stacking on the sonar dataset
seed(1)

# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)

# convert string attributes to integers
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
n_folds = 5
scores = evaluate_algorithm(dataset, stacking, n_folds)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))