'''
Created on 24-May-2017

@author: aii32199
'''

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from sklearn import cross_validation
import numpy as np
from sklearn.tree import DecisionTreeClassifier
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

def CalculateAccuracy(y_test,pred_label):
    nnz = np.shape(y_test)[0] - np.count_nonzero(pred_label - y_test)
    acc = 100*nnz/float(np.shape(y_test)[0])
    return acc

clf1 = KNeighborsClassifier(n_neighbors=2)
clf2 = RandomForestClassifier(n_estimators = 2,random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)

f1 = clf1.predict(X)
acc1 = CalculateAccuracy(y, f1)
print("accuracy from KNN: "+str(acc1) )
 
f2 = clf2.predict(X)
acc2 = CalculateAccuracy(y, f2)
print("accuracy from Random Forest: "+str(acc2) )
 
f3 = clf3.predict(X)
acc3 = CalculateAccuracy(y, f3)
print("accuracy from Naive Bays: "+str(acc3) )
 
f = [f1,f2,f3]
f = np.transpose(f)
 
lr.fit(f, y)
final = lr.predict(f)
 
acc4 = CalculateAccuracy(y, final)
print("accuracy from Stacking: "+str(acc4) )

# accuracy from KNN: 96.66666666666667
# accuracy from Random Forest: 94.66666666666667
# accuracy from Naive Bays: 92.0
# accuracy from Stacking: 97.33333333333333

# sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
#                           meta_classifier=lr)
# 
# print('3-fold cross validation:\n')
# 
# for clf, label in zip([clf1, clf2, clf3, sclf], 
#                       ['KNN', 
#                        'Random Forest', 
#                        'Naive Bayes',
#                        'StackingClassifier']):
# 
#     scores = cross_validation.cross_val_score(clf, X, y, 
#                                               cv=3, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
#           % (scores.mean(), scores.std(), label))

