'''
Created on 24-Nov-2017

@author: aii32199
'''
# Imports for array-handling and plotting
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Keras imports for the data set and building our neural network
from keras.datasets import mnist

#Import Sequential and Load model for creating and loading model
from keras.models import Sequential, load_model

#We will use Dense, Drop out and Activation layers
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

#Let's Start by loading our data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#Plot the digits to verify
plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()

#Lets analyze histogram of the image
plt.figure()
plt.subplot(2,1,1)
plt.imshow(X_train[0], cmap='gray', interpolation='none')
plt.title("Digit: {}".format(y_train[0]))
plt.xticks([])
plt.yticks([])
plt.subplot(2,1,2)
plt.hist(X_train[0].reshape(784))
plt.title("Pixel Value Distribution")

# Print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# As we have data in image form convert it to row vectors
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the data to between 0 and 1 to help with the training
X_train /= 255
X_test /= 255

# Print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

# One-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# Here we will create model of our ANN
# Create a linear stack of layers with the sequential model
model = Sequential()

#Input Layer with 512 Weights
model.add(Dense(512, input_shape=(784,)))

#We will use relu as Activation
model.add(Activation('relu'))

#Put Drop out to prevent over-fitting
model.add(Dropout(0.2))

#Add Hidden layer with 512 neurons with relu activation
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#This is our Output layer with 10 neurons
model.add(Dense(10))
model.add(Activation('softmax'))

#Here we will be compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
optimizer='adam')

# Start training the model and saving metrics in history
history = model.fit(X_train, Y_train,
                    batch_size=128, epochs=20,
                    verbose=2,
                    validation_data=(X_test, Y_test))

# Saving the model on disk
path2save = 'E:/PyDevWorkSpaceTest/Ensembles/Chapter_10/keras_mnist.h5'
model.save(path2save)
print('Saved trained model at %s ' % path2save)
# Plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.show()

#Let's load the model for testing data
path2save ='keras_mnist.h5'
mnist_model = load_model(path2save)

#We will use Evaluate function
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

#Load the model and create predictions on the test set
mnist_model = load_model(path2save)
predicted_classes = mnist_model.predict_classes(X_test)

#See which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

#Adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)
plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray',
    interpolation='none')
    plt.title(
    "Predicted: {}, Truth: {}".format(predicted_classes[correct],
    y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray',
    interpolation='none')
    plt.title(
    "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
    y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])
    
plt.show()    