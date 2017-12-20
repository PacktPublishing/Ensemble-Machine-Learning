# Ensemble Machine Learning
This is the code repository for [Ensemble Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/ensemble-machine-learning?utm_source=github&utm_medium=repository&utm_campaign=9781788297752), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
Ensembling  is a technique of combining two or more similar or dissimilar machine learning algorithms to create a model that delivers superior prediction power. This book will show you how you can use many weak algorithms to make a strong predictive model. This book contains Python code for different machine learning algorithms so that you can easily understand and implement it in your own systems.

This book covers different machine learning algorithms that are widely used in the practical world to make predictions and classifications. It addresses different aspects of a prediction framework, such as data pre-processing, model training, validation of the model, and more. You will gain knowledge of different machine learning aspects such as bagging (decision trees and random forests), Boosting (Ada-boost) and stacking (a combination of bagging and boosting algorithms).

Then you’ll learn how to implement them by building ensemble models using TensorFlow and Python libraries such as scikit-learn and NumPy. As machine learning touches almost every field of the digital world, you’ll see how these algorithms can be used in different applications such as computer vision, speech recognition, making recommendations, grouping and document classification, fitting regression on data, and more.

By the end of this book, you’ll understand how to combine machine learning algorithms to work behind the scenes and reduce challenges and common problems.

## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
# Import All the required packages from sklearn
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

#Load data 
iris = load_iris()
X = iris.data
Y = iris.target
```

This book is a practical walkthrough of the machine learning technologies that require implementation of algorithms by you to understand the concepts in a more concrete way. I have used Python as the language to implement the algorithms in the form of code. You need not be a Python expert to code these algorithms; a simple understanding of Python is enough to get started with the implementation.

The code included in this book can run on Python 2.7 and 3, but you will need the NumPy and scikit-learn packages to implement most of the code discussed in this book.

For the implementation of ANNs, I have used Keras and TensorFlow libraries; again, basic a understanding of these libraries is enough for the code implementation.

## Related Products
* [Mastering Machine Learning Algorithms](https://www.packtpub.com/big-data-and-business-intelligence/mastering-machine-learning-algorithms?utm_source=github&utm_medium=repository&utm_campaign=9781788621113)

* [Machine Learning with the Elastic Stack](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-elastic-stack?utm_source=github&utm_medium=repository&utm_campaign=9781788477543)

* [Applied Machine Learning with Python](https://www.packtpub.com/big-data-and-business-intelligence/applied-machine-learning-python?utm_source=github&utm_medium=repository&utm_campaign=9781788297066)

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSe5qwunkGf6PUvzPirPDtuy1Du5Rlzew23UBp2S-P3wB-GcwQ/viewform) if you have any feedback or suggestions.
