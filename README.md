## Machine Learning – project:
# GenderClassification-


# Gender Classification By Voice Records

## Name: Moran Oshia & Amit Bibi

### Moderator: Dr. Lee-Ad Gottlieb


## Table of contents:

 1.Introduction

 2. Database

 3. Project Description

 4. The techniques

   4.1 SVM

   4.2 KNN

   4.3 Logistic Regression

   4.4 Decision Tree

   4.5 CNN

 5. Challenges

 6. Conclusions


## Introduction:

In this paper, we will summarize our ML final project.

We will describe the techniques, database, libraries, code we used and
the difficulties we had during the process, what we did to solve them and
the results of the project.

We decided to choose for our final project in the Machine Learning
course “Gender Recognition by Voice Records”, our models are built to
classify between men and women.

We used 5 techniques: CNN, KNN, logistic regression, decision tree and
SVM.

The main purpose of machines is to predict according to the 20
characteristics of the record whether it is male or female.

## Database:

The Database is taken from Kaggle.

The Database was created to identify a voice as male or female, based
upon acoustic properties of the voice and speech. The dataset consists of
3,168 recorded voice samples, collected from male and female speakers,
divided equally between male and female.

The database is containing 21 acoustic properties of each voice:

```
● meanfreq: mean frequency (in kHz)
● sd: standard deviation of frequency
● median: median frequency (in kHz)
● Q25: first quantile (in kHz)
● Q75: third quantile (in kHz)
● IQR: interquantile range (in kHz)
● skew: skewness (see note in specprop description)
● kurt: kurtosis (see note in specprop description)
● sp.ent: spectral entropy
● sfm: spectral flatness
● mode: mode frequency
● centroid: frequency centroid.
● peakf: peak frequency (frequency with highest energy).
● meanfun: average of fundamental frequency measured across
acoustic signal.
● minfun: minimum fundamental frequency measured across
acoustic signal.
● maxfun: maximum fundamental frequency measured across
acoustic signal.
● meandom: average of dominant frequency measured across
acoustic signal.
● mindom: minimum of dominant frequency measured across
acoustic signal.
● maxdom: maximum of dominant frequency measured across
acoustic signal.
● dfrange: range of dominant frequency measured across acoustic
signal.
● modindx: modulation index. Calculated as the accumulated
absolute difference between adjacent measurements of
fundamental frequencies divided by the frequency range.
● label: male or female 1 for man and 0 for woman
```
**print(data.describe()):**

Link to the Database:
https://www.kaggle.com/primaryobjects/voicegender


## Project Description:

We used the Panda library to read the data from the CSV database file.

We split our data into groups: Y and X, Y which holds the label and X
which holds all the 20 acoustic properties.

Afterward, we splitted these groups to 2 parts of train and test sets, with
the function **train_test_split** import from sklearn.model_selection, we
chose to split it to 20% test and 80% train.

We ran in for 100 rounds and splitted each round to train and test,
afterward we sent the train and test group to all of the different
techniques we chose.at the end we calculate the average result for the
training for all of the techniques.

## The techniques:

### SVM:

There are several kernel types that can be sent to the machine, each type
is cutting the information in a different way.

The SVM model is known as a model capable of handling high-
dimensional samples, even when the feature dimension is greater than
the number of samples as in our case with voice subtleties between men
and women.

We sent 4 types of kernel: with random state =1, Linear, polyand and rbf.

The results:

![alt text](https://github.com/MoranOshia/GenderClassification-/blob/main/Photos/svm.PNG)


### KNN:

The k-nearest neighbour’s algorithm assumes that similar things exist
nearby. In other words, similar things are near to each other.

The K-NN is a non-parametric method proposed for classification and
regression.

Search for a nearby neighbour, get the classification according to the
nearest neighbours.

We implemented the model twice with the library Sklearn.

The first implementation is when k = 3 we run like the others technique
100 times and take the average of the training test we got.

The second implementation is BestKnn which runs with k =1 to 100 for
50 rounds and found out which of the k has the best result. We got that
the best result was when k = 7

The results:

K = 3:

![alt text](https://github.com/MoranOshia/GenderClassification-/blob/main/Photos/knn3.PNG)


Find the best k:

![alt text](https://github.com/MoranOshia/GenderClassification-/blob/main/Photos/knn.PNG)


### Logistic Regression:

Logistic regression is a function that translates the input into one of two
categories, the “classic” application of logistic regression model is a
binary classification.

You can think of logistic regression as an on-off switch.

It can stand alone, or some version of it may be used as a mathematical
component to form switches, or gates, that relay or block the flow of
information.

We implemented the model with the library called Sklearn as well like
the KNN model.

The results:

![alt text](https://github.com/MoranOshia/GenderClassification-/blob/main/Photos/lr.PNG)


### Decision Tree:

The goal is to create a model that predicts the value of a target variable
by learning simple decision rules derived from data attributes.

The results:

![alt text](https://github.com/MoranOshia/GenderClassification-/blob/main/Photos/dt.PNG)


### CNN:

A Convolutional Neural Network (ConvNet / CNN) is a Deep Learning
algorithm that can take in an input image, assign importance (learnable
weights and biases) to various aspects/objects in the image, and be able
to differentiate one from the other.

The results:

Cnn:

![alt text](https://github.com/MoranOshia/GenderClassification-/blob/main/Photos/cnn.PNG)


Cnn 2D:

![alt text](https://github.com/MoranOshia/GenderClassification-/blob/main/Photos/cnn2d.PNG)


## Challenges:

Our difficulty was mostly figuring out how to make the division properly
so that everything would work. We wanted the reading of the data to be
done another time and to fit all types of functions.

We worked on it for quite some time, read online and tried, until the
distribution was successful. Once we understood how to make the
division and how to deal with it in the libraries the work on the database
began to become clearer.

Understanding the sklearn library and using it properly and its functions
was also challenging as it is a library with many options which we are not
familiar with.

In addition, writing and implementing CNN's own functions would have
been challenging, we tried to implement CNN with the help of
TensorFlow and very quickly realized that this is a difficult task that may
not bear fruit, and we started to investigate and discovered more about
TensorFlow library and it was helpful because we started to understand
how to install and use it.


## Conclusions:

Result for all techniques:

```
Techniques Result
svm 97.
Svm Linear 97.
Svm Poly 97.
Svm Rbf 97.
Knn k=3 97.
Knn - Find best K (k = 7) 97.
Logistic regression 96.
Decision tree 96.
Cnn 98.
Cnn 2D 95.
```
We did our seminarian presentation on "How AI and ML help to fight
the Coruna virus" and we decided to take this project seriously because
our final project is in ML.

Our understanding of the methods grew and expanded.
Undoubtedly it can be seen that the methods of deep learning are much
better than different methods of ML.

In the Knn method it can be seen that the number of neighbours in the
KNN technique affects the success rate of the machine. As the number of
neighbours in the machine increases, the success rates increase. I.e.
people who are similar to each other in terms of characteristics.
Also, in the Svm, we can see that the results are very similar for the
different kernel, so it can be concluded that the kernel is not very
effected on the result.

All our techniques came out with fairly close results.
In our opinion, possible reasons that there is no significant technique
that is the best is because the database is relatively small and contains
only about 3K data.


