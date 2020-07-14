#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 01:41:03 2018

@author: hollyerickson
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from cesium import featurize

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#%%
train_md = pd.read_pickle('train_md.pkl')
train_lc = pd.read_csv('./Data/training_set.csv')
print(train_md.info())
print(train_lc.info())

#%%
features_to_use = ["amplitude", "flux_percentile_ratio_mid20", "flux_percentile_ratio_mid35", "flux_percentile_ratio_mid50", "flux_percentile_ratio_mid65", "flux_percentile_ratio_mid80", "max_slope", "maximum", "median", "median_absolute_deviation", "minimum", "percent_amplitude", "percent_beyond_1_std", "percent_close_to_median",  "percent_difference_flux_percentile", "period_fast", "qso_log_chi2_qsonu", "qso_log_chi2nuNULL_chi2nu", "skew", "std", "stetson_j", "stetson_k", "weighted_average", "fold2P_slope_10percentile", "fold2P_slope_90percentile", "freq1_amplitude1", "freq1_amplitude2", "freq1_amplitude3", "freq1_amplitude4", "freq1_freq", "freq1_lambda", "freq1_rel_phase2", "freq1_rel_phase3", "freq1_rel_phase4", "freq1_signif", "freq2_amplitude1", "freq2_amplitude2", "freq2_amplitude3", "freq2_amplitude4", "freq2_freq", "freq2_rel_phase2", "freq2_rel_phase3", "freq2_rel_phase4", "freq3_amplitude1", "freq3_amplitude2", "freq3_amplitude3", "freq3_amplitude4", "freq3_freq", "freq3_rel_phase2", "freq3_rel_phase3", "freq3_rel_phase4", "freq_amplitude_ratio_21", "freq_amplitude_ratio_31", "freq_frequency_ratio_21", "freq_frequency_ratio_31", "freq_model_max_delta_mags", "freq_model_min_delta_mags", "freq_model_phi1_phi2", "freq_n_alias", "freq_signif_ratio_21", "freq_signif_ratio_31", "freq_varrat", "freq_y_offset", "linear_trend", "medperc90_2p_p", "p2p_scatter_2praw", "p2p_scatter_over_mad","p2p_scatter_pfold_over_mad", "p2p_ssqr_diff_over_var", "scatter_res_raw", "all_times_nhist_numpeaks", "all_times_nhist_peak1_bin", "all_times_nhist_peak2_bin", "all_times_nhist_peak3_bin", "all_times_nhist_peak4_bin", "all_times_nhist_peak_1_to_2", "all_times_nhist_peak_1_to_3", "all_times_nhist_peak_1_to_4", "all_times_nhist_peak_2_to_3", "all_times_nhist_peak_2_to_4", "all_times_nhist_peak_3_to_4", "all_times_nhist_peak_val", "avg_double_to_single_step", "avg_err", "avgt", "cad_probs_1", "cad_probs_10", "cad_probs_20", "cad_probs_30", "cad_probs_40", "cad_probs_50", "cad_probs_100", "cad_probs_500", "cad_probs_1000", "cad_probs_5000", "cad_probs_10000", "cad_probs_50000", "cad_probs_100000", "cad_probs_500000", "cad_probs_1000000", "cad_probs_5000000", "cad_probs_10000000", "cads_avg", "cads_med", "cads_std", "mean", "med_double_to_single_step", "med_err", "n_epochs", "std_double_to_single_step", "std_err", "total_time"]
                  

cesium_feat = pd.DataFrame()
for name, group in train_lc.groupby('object_id'): 
    df = pd.DataFrame()
    df1 = df.append(group, ignore_index=True)
    fset_cesium = featurize.featurize_time_series(times=df1['mjd'],
                                              values=df1['flux'],
                                              errors=df1['flux_err'],
                                              names = df1['object_id'],
                                              features_to_use=features_to_use)
    cesium_feat = pd.concat([cesium_feat,fset_cesium])
    
#%%    
    
print(cesium_feat.info())

#%%
# Change weird names of cesium columns - note numbers below may change!!
cols = list(cesium_feat) #0: 112
for ind in range(0, 112):
    cols[ind] = cols[ind][0]
    

cesium_feat.columns=cols
print(cesium_feat.info(verbose=True))

#%%
# add object_id column
cesium_feat.reset_index(level=0, inplace=True)
cesium_feat =cesium_feat.rename(columns = {'index':'object_id'})

#%%

train_md = pd.merge(train_md, cesium_feat, on='object_id')
print('\n')
print(train_md.info())

#%%
train_md.to_pickle('train_md_cesium.pkl') 


#%%
# change inf to NaN's for removal
train_md.replace([np.inf, -np.inf], np.nan, inplace = True)

#%%
# Get names of columns containing NaN
nan_cols = train_md.columns[train_md.isna().any()].tolist()
cads = ["avgt", "cad_probs_1", "cad_probs_10", "cad_probs_20", "cad_probs_30", "cad_probs_40", "cad_probs_50", "cad_probs_100", "cad_probs_500", "cad_probs_1000", "cad_probs_5000", "cad_probs_10000", "cad_probs_50000", "cad_probs_100000", "cad_probs_500000", "cad_probs_1000000", "cad_probs_5000000", "cad_probs_10000000", "cads_avg", "cads_med", "cads_std"]

not_important = ['freq1_lambda','freq_amplitude_ratio_31', 'freq_amplitude_ratio_21']
drop_feats = nan_cols + cads + not_important + ['target', 'object_id', 'diff_flux', 'flux_min', 'flux_max', 'flux_mean', 'check']
print(len(drop_feats))

#%%
X = train_md.drop(drop_feats, axis=1) 
y = train_md['target']

#%%

# standard scaler on features

cols = list(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=cols)


#%%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state = 43)

#%%
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

predict = logreg.predict(X_test)
print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict, normalize=True, sample_weight=None))

#%%
# Hyperparameter Tuning for RANDOM FOREST


parameter_grid = {'max_depth': [10, 20, 50, 100, None], #30, 70
                      'max_features': ['log2', 'sqrt', 5], 
                      'n_estimators': [200, 500, 800, 1000]} # 400, 600, 1200, 1400, 1600, 1800, 2000,
#'min_samples_leaf': [1, 2, 4],
# 'min_samples_split': [2, 5, 10],
clf = RandomForestClassifier()
cross_validation = StratifiedKFold(n_splits=3)

grid_search = GridSearchCV(clf, scoring='accuracy', param_grid=parameter_grid, cv=cross_validation, verbose=1)

grid_search.fit(X_train, y_train)
model = grid_search
parameters = grid_search.best_params_

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
    
predict = grid_search.predict(X_test)
print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict, normalize=True, sample_weight=None))

#%%
 
rfc = RandomForestClassifier(max_depth= None, max_features= 'sqrt', n_estimators= 500)
rfc.fit(X_train, y_train)
predict = rfc.predict(X_test)

print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict, normalize=True, sample_weight=None))
feat_importance = rfc.feature_importances_

#%%
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = list(feat_importance)
features.sort_values(by=['importance'], ascending = False, inplace=True)

#%%
features2 = pd.DataFrame()
features2['feature'] = X.columns
features2['importance'] = list(feat_importance)
features2.sort_values(by=['importance'], ascending = False, inplace=True)

#%% 
print(features2)


