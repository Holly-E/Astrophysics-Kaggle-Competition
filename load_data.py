#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:21:58 2018

@author: hollyerickson
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#%%

train_lc = pd.read_csv('./Data/training_set.csv') #lc = light curve data
sample_test_lc = pd.read_csv('./Data/test_set_sample.csv') # first 1,000,000 entries

train_lc['freq'] = train_lc.groupby('object_id')['object_id'].transform('count')

#%%
train_lc.iloc[0:352].to_csv('./Data/obj615.csv', index=False) 
test = pd.read_csv('./Data/obj615.csv')
test.info

#%%
train_lc.iloc[352:702].to_csv('./Data/obj713.csv', index=False) 
test2 = pd.read_csv('./Data/obj713.csv')
test2.info

#%%
print(train_lc.info())
print(sample_test_lc.info())

print(train_lc['object_id'].nunique())
print(train_lc['mjd'].nunique())
print(train_lc['passband'].nunique())
print(train_lc['flux'].nunique())
print(train_lc['flux_err'].nunique())
print(train_lc['detected'].nunique())

#%%
print(train_lc['passband'].value_counts())
print(train_lc['detected'].value_counts())

print(train_lc.head(5))

#%%

print(sample_test_lc['object_id'].nunique())
print(sample_test_lc['mjd'].nunique())
print(sample_test_lc['passband'].nunique())
print(sample_test_lc['flux'].nunique())
print(sample_test_lc['flux_err'].nunique())
print(sample_test_lc['detected'].nunique())

print(sample_test_lc['passband'].value_counts())
print(sample_test_lc['detected'].value_counts())

print(sample_test_lc.head(5))

#%%

#get the % of objects per passband
dftr = pd.DataFrame(train_lc['passband'].value_counts())
dftr.columns = ['count']
print(dftr)
dftr['%obj'] = dftr / 7848
print(dftr)

#%%
# Load metadata aka 'header file' containing target
train_md = pd.read_csv('./Data/training_set_metadata.csv')
test_md = pd.read_csv('./Data/test_set_metadata.csv') # Full test set

print(train_md.info())
print(test_md.info())

print (test_md['object_id'].nunique())

#%%
# preliminary logreg

X = train_md.drop(['target', 'distmod', 'object_id'], axis=1)
y = train_md['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 43)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

predict = logreg.predict(X_test)
print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict)) #normalize=True, sample_weight=None))

#%%
def add_mkt_return(grp):
    train_lc['target'] = train_md['target']
    return grp

train_lc.groupby('object_id').apply(add_mkt_return)

#%%
# combine header and metadata on object id - Can drop all meta columns except target
train_merge = pd.merge(train_lc, train_md, on='object_id')
print(train_merge.head(5))

#%%
"""
X = train_merge.drop(['target', 'distmod', 'object_id'], axis=1)
y = train_merge['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 43)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

predict = logreg.predict(X_test)
print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict)) #normalize=True, sample_weight=None))
"""