# Felipe Derewlany Gutierrez
# Kaggle Titanic Competition
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import   KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from itertools import combinations
from sklearn.feature_selection import SelectFromModel


class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


#Importing data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#print(df_train)

#print(df_train.isnull().sum())
#print(df_train['Name'])

# Getting the objective parameter and dropping it from train dataset
y_train = df_train['Survived']
df_train = df_train.drop(['Survived'], axis=1)

# Dropping Passenger ID as it is unuseful
df_train = df_train.drop(['PassengerId'], axis=1)
df_test = df_test.drop(['PassengerId'], axis=1)

# Filling empty ages with the mean
data = [df_train, df_test]

for dataset in data:
    mean = dataset['Age'].mean()
    std = dataset['Age'].std()
    is_null = dataset['Age'].isnull().sum()

    # Generate random numbers between
    rand_age = np.random.randint(mean-std, mean+std, size=is_null)

    # Fill the empty age fields with the random numbers from above
    age_slice = dataset['Age'].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset['Age'] = age_slice.astype(int)

#print(df_train['Age'].isnull().sum())

# Extracting information from the names as titles
data = [df_train, df_test]
titles = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract( ' ([A-Za-z]+)\.', expand=False)

    # replace titles as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Major', 'Don', 'Dr', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    # replace titles that are in other languages
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)

    # fill empty ones with 0
    dataset['Title'] = dataset['Title'].fillna(0)


# Dropping Names
df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)

# Mapping genders - ok
genders = {'male': 0, 'female': 1}
data = [df_train, df_test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# Dropping Ticket as it is useless
df_train = df_train.drop(['Ticket'], axis=1)
df_test = df_test.drop(['Ticket'], axis=1)

# Dropping Cabin and Embarked for preliminary tests
df_train = df_train.drop(['Cabin', 'Embarked'], axis=1)
df_test = df_test.drop(['Cabin', 'Embarked'], axis=1)

#print(df_train.info())

# Min Max Scaling
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(df_train)
X_test_norm = mms.fit_transform(df_test)

# Standard Scaling
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(df_train)
X_test_std = stdsc.fit_transform(df_test)

'''
# Running Machine Learning Algorithms
# Perceptron
print('Perceptron with min max scaled data')
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_norm, y_train)
print('Training accuracy:', ppn.score(X_train_norm, y_train))

print('Perceptron with standard scaled data')
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
print('Training accuracy:', ppn.score(X_train_std, y_train))

# Logistic Regression
print('Logistic Regression (L1 norm) with min max scaled data')
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_norm, y_train)
print('Training accuracy:', lr.score(X_train_norm, y_train))
#print('Test accuracy:', lr.score(X_test_norm, y_test))

print('Logistic Regression (L1 norm) with standard scaled data')
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
#print('Test accuracy:', lr.score(X_test_std, y_test))
'''

# Train Test Splitting for checking test accuracy on non trained data
# using standard scaled data due to its better results o previous tests
X_train, X_test, Y_train, Y_test = train_test_split(X_train_std, y_train, test_size=0.1, random_state=0, stratify=y_train)

# Perceptron
print('Perceptron with standard scaled data')
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train, Y_train)
print('Training accuracy:', ppn.score(X_train, Y_train))
print('Test accuracy:', ppn.score(X_test, Y_test))

# Logistic Regression
print('\nLogistic Regression (L1 norm) with standard scaled data')
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train, Y_train)
print('Training accuracy:', lr.score(X_train, Y_train))
print('Test accuracy:', lr.score(X_test, Y_test))


# KNN
knn = KNeighborsClassifier(n_neighbors=5)

#selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

#plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

