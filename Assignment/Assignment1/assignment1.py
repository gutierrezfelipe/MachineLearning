# Felipe Derewlany Gutierrez
# Kaggle Titanic Competition
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from matplotlib.colors import ListedColormap
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from itertools import combinations
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


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


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='Test set')


# Importing data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#print(df_train)

#print(df_train.isnull().sum())

## Preprocessing
# Getting the objective parameter and dropping it from train dataset
y_train = df_train['Survived']
df_train = df_train.drop(['Survived'], axis=1)

# Dropping Passenger ID and Ticket number as they are useless
df_train = df_train.drop(['PassengerId', 'Ticket'], axis=1)
df_test = df_test.drop(['PassengerId', 'Ticket'], axis=1)

# Filling empty ages with random values based on the mean and standard deviation
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

# Mapping genders
genders = {'male': 0, 'female': 1}
data = [df_train, df_test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

# Combining SibSp and Parch into Relatives
data = [df_train, df_test]

for dataset in data:
    dataset['Relative'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['Relative'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['Relative'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)


# Dropping SibSp and Parch
df_train = df_train.drop(['SibSp', 'Parch'], axis=1)
df_test = df_test.drop(['SibSp', 'Parch'], axis=1)

# Mapping Embarked
#df_train['Embarked'].describe()
# As embarked has only two missing values, will replace it with the most commom one
commom_value = 'S'
data = [df_train, df_test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(commom_value)

embarked = {'S': 0, 'C': 1, 'Q': 2}
data = [df_train, df_test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked)


# Filling empty and converting to int
data = [df_train, df_test]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

# Dropping Cabin for preliminary tests
df_train = df_train.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Cabin'], axis=1)

# Final data check
#print(df_train.info())

# Min Max Scaling
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(df_train)
X_test_norm = mms.fit_transform(df_test)

# Standard Scaling
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(df_train)
X_test_std = stdsc.fit_transform(df_test)

# Running Machine Learning Algorithms - Primary tests
print('Primary Tests')

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

# In various tests perceptron seems to get better results with min max scaled data
# So perceptron will be ran using min max scaled
# Logistic Regression oscillated between the two scaling methods
# Using standard scaled data due to better results on previous tests for other algorithms

## Train Test Splitting for checking test accuracy on non trained data
X_train, X_test, Y_train, Y_test = train_test_split(X_train_std, y_train, test_size=0.1, random_state=0, stratify=y_train)

## Final Tests
print('Final Tests')
# Perceptron
print('Perceptron with standard scaled data')
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train, Y_train)
print('Training accuracy:', ppn.score(X_train, Y_train))
print('Test accuracy:', ppn.score(X_test, Y_test))


# Stochastic Gradient Descent
print('\nSGD with standard scaled data')
sgd = sklearn.linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
print('Training accuracy:', sgd.score(X_train, Y_train))
print('Test accuracy:', sgd.score(X_test, Y_test))


# SVM
print('\nSVM with standard scaled data')
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_train, Y_train)
#plot_decision_regions(X_train[:, 0:2], Y_train, classifier=svm)
#plt.legend(loc='upper left')
#plt.tight_layout()
#plt.show()

Y_pred = svm.predict(X_test)
print('Training accuracy:', svm.score(X_train, Y_train))
print('Test accuracy:', svm.score(X_test, Y_test))


# Logistic Regression
print('\nLogistic Regression (L1 norm) with standard scaled data')
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train, Y_train)
print('Training accuracy:', lr.score(X_train, Y_train))
print('Test accuracy:', lr.score(X_test, Y_test))


# Decision Tree
print('\nDecision Tree with standard scaled data')
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

tree_model.fit(X_train, Y_train)
#plot_decision_regions(X_train_std, y_train, classifier=tree_model, test_idx=range(105, 150))
#plt.xlabel('Sex [cm]')
#plt.ylabel('Age [cm]')
#plt.legend(loc='upper left')
#plt.tight_layout()
#plt.show()

#y_pred_tree = tree_model.predict(X_test[:, 0:2])
Y_pred_tree = tree_model.predict(X_test)
print('Training accuracy:', tree_model.score(X_train, Y_train))
print('Test accuracy:', tree_model.score(X_test, Y_test))


# Random Forest
print('\nRandom Forest with standard scaled data')
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print('Training accuracy:', random_forest.score(X_train, Y_train))
print('Test accuracy:', random_forest.score(X_test, Y_test))
# Random Forest seems to overfit as the training accuracy is very high (close to 97%) and
# the test accuracy is considerably lower (close to 75%)

# K-Nearest Neighbors
print('\nK-Nearest Neighbors with standard scaled data')
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


knn.fit(X_train, Y_train)
#knn.score(X_train[:, 0:2], Y_train)
#y_predict_knn = knn.predict(X_test[:, 0:2])
print('Training accuracy:', knn.score(X_train, Y_train))
print('Test accuracy:', knn.score(X_test, Y_test))

