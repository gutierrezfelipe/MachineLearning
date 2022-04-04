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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


#Importing data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#df_train = df_train.drop(['PassengerId', 'Cabin'], axis=1)


print(df_train)
#print(df_train.isnull().sum())
#print(df_train['Name'])


