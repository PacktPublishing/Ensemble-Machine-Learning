from Chapter_03 import DecisionTree_CART_RF as CART
import pprint
filename = 'bcancer.csv'
dataset = CART.load_csv(filename)
# convert string attributes to integers
for i in range(0, len(dataset[0])):
    CART.str_column_to_float(dataset, i)

#Now remove index column from the data set
dataset_new = []
for row in dataset:
    dataset_new.append([row[i] for i in range(1,len(row))])

#Get training and testing data split
training,testing = CART.getTrainTestData(dataset_new, 0.7)
tree = CART.build_tree(training,11,5)
pprint.pprint(tree)

pre = []
act = []
for row in training:
    prediction = CART.predict(tree, row)
    pre.append(prediction)
    actual = act.append(row[-1]) 
#     print('Expected=%d, Got=%d' % (row[-1], prediction))
# print_tree(tree)
acc = CART.accuracy_metric(act, pre)

print('training accuracy: %.2f'%acc)

for row in testing:
    prediction = CART.predict(tree, row)
    pre.append(prediction)
    actual = act.append(row[-1])
    acc = CART.accuracy_metric(act, pre)
# pprint.pprint(tree)
print('testing accuracy: %.2f'%acc) 