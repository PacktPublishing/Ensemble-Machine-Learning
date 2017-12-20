from Chapter_03 import DecisionTree_CART_RF as rf
filename = 'bcancer.csv'
dataset = rf.load_csv(filename)
# convert string attributes to integers
for i in range(0, len(dataset[0])-1):
    rf.str_column_to_float(dataset, i)
# convert class column to integers
rf.str_column_to_int(dataset, len(dataset[0])-1)

dataset_new = []
for row in dataset:
    dataset_new.append([row[i] for i in range(1,len(row))])
# # evaluate algorithm
dataset = dataset_new
n_folds = 5
max_depth = 3
min_size = 1
sample_size = 0.5
n_features = 5#int(sqrt(len(dataset[0])-1))
print("features: %d"%n_features)

for n_trees in [1, 5, 10]:
    scores = rf.evaluate_algorithm(dataset, rf.random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))