# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:37:33 2017

@author: DX
"""
'''
Created on 15-May-2017

@author: aii32199
'''
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
#Generate sample data
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
# Compute clustering with Means
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
# Compute clustering with MiniBatchKMeans

# Plot result
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
#k_means_cluster_centers = np.load('E:/PyDevWorkSpaceTest/Ensembles/Chapter_01/data/kmenasCenter.npy')
# np.save('E:/PyDevWorkSpaceTest/Ensembles/Chapter_01/data/kmenasCenter.npy',k_means_cluster_centers)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)



ax = fig.add_subplot(1, 2,1)
# ax.plot(X[:, 0], X[:, 1], 'w',markerfacecolor='k', marker='.',markersize=8)
# KMeans
ax = fig.add_subplot(1,2,1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',markerfacecolor=col, marker='.',markersize=8)
#     plt.text(X[my_members, 0], X[my_members, 1],  '%i' % (k))
    ax.plot(cluster_center[0], cluster_center[1], marker='o', markerfacecolor=col,
            markeredgecolor='k', markersize=10)
    plt.text(cluster_center[0], cluster_center[1],  'Cluster: %i' % (k))

# ax.set_title('KMeans')


test_point = [-1.3,1.3]
ax.plot(test_point[0],test_point[1],marker='x',markerfacecolor='r',markersize=12)
#plt.text(test_point[0],test_point[1],  'point:%.1f,%.1f' % (test_point[0],test_point[1]))
#Check out its distance from each of the cluster
dist = []
for center in k_means_cluster_centers:
    dist.append((sum(np.square((center) - (test_point)))))

min = np.argmin(dist)
test_point = [-1.3,1.3]

ax = fig.add_subplot(1,2,2)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',markerfacecolor=col, marker='.',markersize=8)
#     plt.text(X[my_members, 0], X[my_members, 1],  '%i' % (k))
    ax.plot(cluster_center[0], cluster_center[1], marker='o', markerfacecolor=col,
            markeredgecolor='k', markersize=10)
    plt.text(cluster_center[0], cluster_center[1],  'Cluster: %i' % (k))
ax.plot(test_point[0],test_point[1],marker='x',markerfacecolor='r',markersize=8)
plt.text(test_point[0],test_point[1],  '%i' % (min))

print('distances are: '+ str(dist))
print('Minimum distance index: '+str(min))        


#Supervised algorithm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import log_loss
y = k_means_labels

X_train, y_train = X[:2000], y[:2000]
X_valid, y_valid = X[2000:2500], y[2000:2500]
X_train_valid, y_train_valid = X[:2500], y[:2500]
X_test, y_test = X[2500:], y[2500:]

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = rf(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)
clf_probs = clf.predict_proba(X_test)

pred_label = np.argmax(clf_probs,axis=1)
# score = log_loss(y_test, clf_probs)
nnz = np.shape(y_test)[0] - np.count_nonzero(pred_label - y_test)
acc = 100*nnz/np.shape(y_test)[0]
print('accuracy is: '+str(acc))

clf_probs = clf.predict_proba(test_point)
pred_label = np.argmax(clf_probs,axis=1)
print('RF predicted label: '+str(pred_label))
plt.show()
# ax.set_xticks(())
# ax.set_yticks(())
# plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
#     t_batch, k_means.inertia_))

# MiniBatchKMeans
# ax = fig.add_subplot(1, 3, 2)
# for k, col in zip(range(n_clusters), colors):
#     my_members = mbk_means_labels == order[k]
#     cluster_center = mbk_means_cluster_centers[order[k]]
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w',
#             markerfacecolor=col, marker='.')
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=6)
# ax.set_title('MiniBatchKMeans')
# ax.set_xticks(())
# ax.set_yticks(())
# # plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
# #          (t_mini_batch, mbk.inertia_))
# 
# # Initialise the different array to all False
# different = (mbk_means_labels == 4)
# ax = fig.add_subplot(1, 3, 3)
# 
# for k in range(n_clusters):
#     different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
# 
# identic = np.logical_not(different)
# ax.plot(X[identic, 0], X[identic, 1], 'w',
#         markerfacecolor='#bbbbbb', marker='.')
# ax.plot(X[different, 0], X[different, 1], 'w',
#         markerfacecolor='m', marker='.')
# ax.set_title('Difference')
# ax.set_xticks(())
# ax.set_yticks(())


