#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:13:56 2018

@author: hollyerickson
"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle

#%%
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


#%%
train_lc = pd.read_csv('./Data/training_set.csv')
train_md = pd.read_csv('./Data/training_set_metadata.csv')

#%% 
test_lc = pd.read_csv('./Data/test_set.csv', nrows = 10)


#%%
cols = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected']

reader = pd.read_csv('./Data/test_set.csv', nrows = 10000000, names=cols, skiprows=14903870)

#%%
first = reader.iloc[0: 1]
last = reader.iloc[9999100: 10000000]

#%%
reader.columns = cols


#%%
len = reader['object_id'].nunique()
print(len)

#%%
counter = 142648
current = 1
for name, group in reader.groupby('object_id'): 
    if current != len:
        df = pd.DataFrame()
        df1 = df.append(group, ignore_index=True)
        name = './Objects/obj_' + str(counter) + '.pkl'
        df1.to_pickle(name)
        counter +=1
        current += 1
    
    

#%%
test_md = pd.read_csv('./Data/test_set_metadata.csv')


#%%
"""
#add target col to train_lc

train_target = train_md[['object_id', 'target']]
print(train_target.head())

train_lc = pd.merge(train_target,train_lc, on='object_id')
print(train_lc.head(5))
"""

#%%
"""
# Get KMeans Labels by nearest sky and galaxy coords, MUST DO AT SAME TIME AS TEST SET!!

X_sky = train_md[['ra', 'decl']]
X_gal = train_md[['gal_l', 'gal_b']]
print(X_sky.head())
print(X_gal.head())

kmeans = KMeans(n_clusters=10).fit(X_sky)
train_md['labels_sky'] = kmeans.predict(X_sky)

kmeans = KMeans(n_clusters=10).fit(X_gal)
train_md['labels_gal'] = kmeans.predict(X_gal)
"""

#%%

def add_mjd_diff(df):
    df['mjd_detected'] = np.NaN
    df.loc[df.detected == 1, 'mjd_detected'] = df.loc[df.detected == 1, 'mjd']
    gr_mjd = df.groupby('object_id').mjd_detected 
    df['mjd_diff']  = gr_mjd.transform('max') - gr_mjd.transform('min')
    return df

#%%
"""
This code creates a series - if put into a DataFrame would make a column for each object id, and same # rows of original dataframe. In each column, says NaN if that object_id was not in that row, and if it it was in that row but detected == 0. If that row contained the columns object id, and dectected == 1, then contains entry for mjd.

gr_mjd = df.groupby('object_id').mjd_detected 

All rows for unique object_ids will have same mjd_diff entry (different accross objects).

df['mjd_diff']  = gr_mjd.transform('max') - gr_mjd.transform('min')
"""
#%%
def flux_max_min(df):
    df['flux_max'] = df.groupby(['object_id'])['flux'].transform(max)
    df['flux_min'] = df.groupby(['object_id'])['flux'].transform(min)
    df['flux_mean'] = df.groupby(['object_id'])['flux'].transform(np.mean)
    return df

def diff_flux(df):
    # can do this on train_md after merging feats over from train_lc
    df['diff_flux'] = df['flux_max'] - df['flux_min']
    return df

#%%
train_lc = flux_max_min(train_lc)
print(train_lc.info())
print(train_lc.head())

#%%
train_lc = add_mjd_diff(train_lc)
feat = pd.DataFrame( train_lc.groupby(['object_id'])['mjd_diff', 'flux_max', 'flux_min', 'flux_mean'].mean())

print(feat.info())
print(feat.head())


#%%
# Add mjd_diff to test data
test_lc = add_mjd_diff(test_lc)
feat_test = pd.DataFrame( test_lc.groupby(['object_id'])['mjd_diff'].mean())

print(feat_test.info())
#%%
print(train_md.head())

train_md = pd.merge(train_md, feat, left_on='object_id', right_index= True)
print('\n')
print(train_md.head())

#%%

train_md = diff_flux(train_md)

#%%
print(train_md.info())
X = train_md.drop(['target','distmod', 'object_id', 'diff_flux'], axis=1) #'diff_flux'
y = train_md['target']

print(X.head())

#%%

# standard scaler on features
print(X.info())
cols = list(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=cols)

print(X_scaled.head())
print(X_scaled.info())


#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state = 43)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

predict = logreg.predict(X_test)
print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict)) #normalize=True, sample_weight=None))

#%%
"""
# probabilities for each output
predict_all = logreg.predict_proba(X_test)
print(predict_all[0])
#%%

X_test['predict'] = predict
print(X_test.head())

#%%
def avg_predict(df):
    series_avg_predict = df.groupby('object_id').predict
    df['avg_predict']  = series_avg_predict.transform('mean')
    df.avg_predict = df.avg_predict.astype(int)
    return df

X_test = avg_predict(X_test)

#%%
print(X_test.head())
X_test['diff'] = X_test['predict'] - X_test['avg_predict']


print(X_test.info())
#%%
#print(X_test['diff'].value_counts())
print(classification_report(y_test, X_test.loc[:,'predict']))
print(accuracy_score(y_test, X_test.loc[:,'predict']))

#%%

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)
print(pred[0])
"""

#%%
# pickle table
train_md.to_pickle('train_md.pkl') 

#%%
test_md.to_pickle('test_md.pkl')
test_lc.to_pickle('test_lc.pkl')