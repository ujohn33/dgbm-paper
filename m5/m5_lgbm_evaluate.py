FINAL_BASE = ['d_1941', 'd_1913'][0]

IMPORT = True 
CACHED_FEATURES = False  # Changed to False to build features from scratch
CACHE_FEATURES = False

from m5_config import *
import argparse
import sys

# Command line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run M5 LGBM model for specific level')
    parser.add_argument('--level', type=int, default=13, 
                        help='Level to run (13=HOBBIES, 14=HOUSEHOLD, 15=FOODS)')
    parser.add_argument('--max_level', type=int, default=None, 
                        help='Maximum level to include (optional)')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
LEVEL = args.level  # Level 13 is HOBBIES; Level 14 is HOUSEHOLD; Level 15 is FOODS (there is no "Level 12")
MAX_LEVEL = args.max_level

import numpy as np  
import pandas as pd 
import psutil
import os
import torch
import pickle
from collections import Counter
import datetime as datetime
from scipy.stats.mstats import gmean
import random
import gc
import gzip
import bz2
import matplotlib.pyplot as plt
from pylab import rcParams

from sklearn.model_selection import RandomizedSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer
import lightgbm as lgb
rcParams['figure.figsize'] = (17,5.5)
rcParams['figure.max_open_warning'] = 0
# %config InlineBackend.figure_format='retina'
import seaborn as sns
from m5_lgbm_winner import parse_args, getCal, loadTrain, getPricePivot, aggTrain, getSubsample, getXYG, clean_df, clean_features


start = datetime.datetime.now()

if TIME_SEED:
    np.random.seed(datetime.datetime.now().microsecond)

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

# Add after major operations to free memory
def clear_memory():
    """Release unused memory more aggressively"""
    gc.collect()
    torch.cuda.empty_cache()  # If using GPU
    # Force Python to release memory to OS (Linux)
    if 'linux' in sys.platform:
        os.system('sync')

def memCheck():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

def ramCheck():
    print("{:.1f} GB used".format(psutil.virtual_memory().used/1e9 - 0.7))

def show_FI(model, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fis = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1][:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")

def avg_FI(all_clfs, featNames, featCount, title = "Feature Importances"):
    # 1. Sum
    clfs = []
    for clf_set in all_clfs:
        for clf in clf_set:
            clfs.append(clf);
    fi = np.zeros( (len(clfs), len(clfs[0].feature_importances_)) )
    for idx, clf in enumerate(clfs):
        fi[idx, :] = clf.feature_importances_
    avg_fi = np.mean(fi, axis = 0)

    # 2. Plot
    fis = avg_fi
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1]#[:featCount]
    #print(indices)
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title(title + ' - {} classifiers'.format(len(clfs)))
    
    return pd.Series(fis[indices], featNames[indices])


def linear_FI_plot(fi, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(np.absolute(fi))[::-1]#[:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fi[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    return pd.Series(fi[indices], featNames[indices])

def avg(arr, axis = 0):
    return np.median(arr, axis = axis)

def predictSet(X, y, groups, scalers, clf_set):
    start_time = datetime.datetime.now(); 
    
    group_list = [*dict.fromkeys(groups)]   
    group_list.sort()
#     print(group_list)
    
    y_unscaled = y * scalers.scaler
    
    all_preds = []; ys=[]; gs = []; xs = []; scaler_stack = []
    if SINGLE_FOLD: group_list = group_list[-1:]
    for group_idx, group in enumerate(group_list):
        g = gc.collect()
        x_holdout = X[groups == group]
        y_holdout = y_unscaled[groups == group] 
        scalers_holdout = scalers[groups == group]
        groups_holdout = groups[groups == group]
        
        preds = np.zeros( (len(QUANTILES), len(y_holdout)), dtype=np.half)
        for q_idx, quantile in enumerate(QUANTILES):            
            q_preds = np.zeros( ( len(clf_set), len(y_holdout) ) )
            for bag_idx, clf in enumerate(clf_set):
                x_clean = x_holdout.drop(columns = [c for c in x_holdout.columns if c=='d' or c=='series'])
                if group_idx >= len(clf_set[bag_idx]): # if out of sample year, blend all years
                    qs_preds = np.zeros( (group_idx, len(x_clean)) )
                    for gidx in range(group_idx):
                        qs_preds[gidx, :] = clf_set[bag_idx][gidx][q_idx].predict(x_clean)
                    q_preds[bag_idx, :] = np.mean(qs_preds, axis = 0)
                else:
                    q_preds[bag_idx, :] = clf_set[bag_idx][group_idx][q_idx].predict(x_clean)
                
            q_preds = avg(q_preds) * scalers_holdout.scaler

            preds[q_idx, :] = q_preds
            
#             print(u"{} \u03BC={:.3f}: {:.4f}".format(group, quantile, quantile_loss(y_holdout, q_preds, quantile) ) )
        
        all_preds.append(preds)
        xs.append(x_holdout)
        ys.append(y_holdout)
        gs.append(groups_holdout)
        scaler_stack.append(scalers_holdout)
        print()
    y_pred = np.hstack(all_preds)
    scaler_stack = pd.concat(scaler_stack)
    y_true = pd.concat(ys)
    groups = pd.concat(gs)
    X = pd.concat(xs)
    
    end_time = datetime.datetime.now(); 
    print("Bag Prediction Time: {}".format(str(end_time - start_time).split('.', 2)[0] ))
    return y_pred, y_true, groups, scaler_stack, X

def predictOOS(X, scalers, clf_set, QUANTILES, validation = False):
    start_time = datetime.datetime.now(); 
    
    group_list = [1 + i for i in range(0, len(clf_set[0]))]   
    if validation:
        group_list = np.zeros(len(clf_set[0]))
        group_list[-1] = 1
    
    
    divisor = sum(group_list)
    print(np.round([g / divisor for g in group_list], 3)); print()
    
    x_holdout = X
    scalers_holdout = scalers 

    preds = np.zeros( (len(clf_set[0][0]), len(x_holdout)), dtype=np.float32)
    for q_idx in range( len(clf_set[0][0])): # loop over quantiles
        print(u'Predicting for \u03BC={}'.format( QUANTILES[q_idx]) )
        
        q_preds = np.zeros( ( len(clf_set), len(x_holdout) ), dtype = np.float32 )
        for bag_idx, clf in enumerate(clf_set):
            x_clean = x_holdout # .drop(columns = [c for c in x_holdout.columns if c=='d' or c=='series'])
            qs_preds = np.zeros( (len(group_list), len(x_clean)), dtype = np.float32 )
            if SINGLE_FOLD: group_list = group_list[-1:]
            for gidx in range(len(group_list)):
                if group_list[gidx] > 0: 
                    qs_preds[gidx, :] = clf_set[bag_idx][gidx][q_idx].predict(x_clean) * group_list[gidx] / divisor
            q_preds[bag_idx, :] = np.sum(qs_preds, axis = 0)

        q_preds = np.mean(q_preds, axis = 0) * scalers_holdout.scaler

        preds[q_idx, :] = q_preds
 
    end_time = datetime.datetime.now(); 
    print("Bag Prediction Time: {}".format(str(end_time - start_time).split('.', 2)[0] ))
    return preds

def wspl(true, pred, weights, trailing_vol, quantile=0.5):
    """
    Compute the weighted scaled pinball loss (WSPL) for a given quantile.
    """
    loss = np.maximum(quantile * (true - pred), (1 - quantile) * (pred - true))
    if weights is not None:
        loss = loss * weights
    if trailing_vol is not None:
        loss = loss / trailing_vol
    return np.mean(loss)
 


# if main
if __name__ == "__main__":
    print("Running M5 LGBM evaluation for level {}...".format(LEVEL))
    if MAX_LEVEL is not None:
        print("Maximum level set to {}.".format(MAX_LEVEL))
    else:
        print("No maximum level set, will include all levels up to {}.".format(LEVEL))
    
    print("Current memory usage:")
    memCheck()
    ramCheck()
    print('IMPORT:', IMPORT)

    if IMPORT:
        clf_sets = []  # ***
        model_path = 'm5/models/'

        print("Loading models from path: {}".format(model_path))

        # if LEVEL != 12: 
        files = [f for f in os.listdir(model_path) if '.pkl' in f]
        if LEVEL == 13 and MAX_LEVEL is None: files = [f for f in files if '13_' in f or 'hobbies' in f]
        if LEVEL == 14 and MAX_LEVEL is None: files = [f for f in files if '14_' in f or 'household' in f]
        if LEVEL == 15 and MAX_LEVEL is None: files = [f for f in files if '15_' in f or 'foods' in f]      
            
        #  else:
        #      files = [f for f in os.listdir(path) if '.pkl' in f and ID_FILTER.lower() in f]
            
        for file in files:
            clf_sets.append(pickle.load(open(model_path + file,'rb')))
    
        clf_df = []; pairs = []
        for clf_set in clf_sets:
            for level, level_clfs in clf_set.items():
                for clf_bag_idx, clf_bag in enumerate(level_clfs):
                    for group_idx, clf_group in enumerate(clf_bag):
                        for quantile_idx, clf in enumerate(clf_group):
                            clf_df.append((level, clf.alpha, group_idx, clf))


        clf_df = pd.DataFrame(clf_df, columns = ['level', 'alpha', 'group', 'clf'])
        
        if LEVEL > 12 and MAX_LEVEL == None:
            clf_df.loc[clf_df.level==12, 'level'] = LEVEL

        
        LEVEL_QUANTILES = {}; clf_set = {}
        for level in sorted(clf_df.level.unique()):

            level_df = clf_df[clf_df.level == level]

            level_list = []
            for group in sorted(level_df.group.unique()):
                group_df = level_df[level_df.group == group].sort_values('alpha')
                if level in LEVEL_QUANTILES:
                    assert LEVEL_QUANTILES[level] == list(group_df.alpha)
                else:
                    LEVEL_QUANTILES[level] = list(group_df.alpha)
                level_list.append(list(group_df.clf))
            if len(level_df.group.unique()) > 1:
                SINGLE_FOLD = False
            clf_set[level] = [level_list]
            print(level, ": ", LEVEL_QUANTILES[level]); 
            print("  Number of quantiles: ", len(LEVEL_QUANTILES[level]))

    for level in sorted(clf_set.keys()):
        print("Level {}:".format(level))
        
        for idx, q in enumerate(LEVEL_QUANTILES[level]):
            print(u'\n\n      Regressors for \u03BC={}:\n'.format(q))
            for clf in [q_clfs[idx] for clfs in clf_set[level] for q_clfs in clfs]:
                print(clf)
        
        print(); print()

    LEVELS = [(12, ['item_id', 'store_id']),
            (11, ['state_id', 'item_id']),
            (10, ['item_id']),
            (9, ['store_id', 'dept_id']),
            (8, ['store_id', 'cat_id']),
            (7, ['state_id', 'dept_id']),
            (6, ['state_id', 'cat_id']),
            (5, ['dept_id']),
            (4, ['cat_id']),
            (3, ['store_id']),
            (2, ['state_id']),
            (1, []) ]

    DOWNSTREAM = {'item_id': ['dept_id', 'cat_id'],
                'dept_id': ['cat_id'],
                'store_id': ['state_id']}

    print(ramCheck())


    cal = getCal()
    cal.date = pd.to_datetime(cal.date)

    day_to_cal_index = dict([(col, idx) for idx, col in enumerate(cal.index)])
    cal_index_to_day = dict([(idx, col) for idx, col in enumerate(cal.index)])

    cal_index_to_wm_yr_wk = dict([(idx, col) for idx, col in enumerate(cal.wm_yr_wk)])
    day_to_wm_yr_wk = dict([(idx, col) for idx, col in cal.wm_yr_wk.items()])

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    print(ramCheck())


    # Load
    train = loadTrain()
    price_pivot = getPricePivot()
    clear_memory()

    # combine
    assert (train.id == price_pivot.columns).all()
    daily_sales = pd.concat((train.iloc[:, :6], 
                            train.iloc[:, 6:] * price_pivot.loc[train.columns[6:].fillna(0)\
                                                                    .map(day_to_wm_yr_wk)].transpose().values ), 
                                axis = 'columns')
    # Aggregate
    train, levels = aggTrain(train)
    daily_sales = aggTrain(daily_sales)[0]
    clear_memory() 

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    # Rescale each level to avoid hitting np.half ceiling and keep similar ranges
    level_multiplier = dict([ (c, (levels==c).sum() / (levels==12).sum()) for c in sorted(levels.unique())])

    # split up level 12
    for row in LEVEL_SPLITS:
        level_multiplier[row[0]] = level_multiplier[12]
        levels.loc[(levels == 12) & (train.cat_id == row[1])] = row[0]

    Counter(levels)
    level_os = dict([(idx, 1/val) for (idx,val) in level_multiplier.items()])
    print('Level multipliers:', level_multiplier)
    print('Level multipliers (1/level):', level_os)

    # Rescale by number of series at each level
    train = pd.concat((train.iloc[:, :6], 
                            train.iloc[:, 6:].multiply( levels.map(level_multiplier), axis = 'index').astype(np.float32) ), 
                                axis = 'columns')

    daily_sales = pd.concat((daily_sales.iloc[:, :6], 
                            daily_sales.iloc[:, 6:].multiply( levels.map(level_multiplier), axis = 'index').astype(np.float32) ), 
                                axis = 'columns')
    clear_memory()

    def loadSampleSub():
        return pd.read_csv(path+ '/' + 'sample_submission.csv').astype(np.int8, errors = 'ignore')

    sample_sub = loadSampleSub()

    assert set(train.id) == set(sample_sub.id.str.split('_').str[:-2].str.join('_'))

    print(len(train))

    print(ramCheck())

    train_filter = (   
                ( ( MAX_LEVEL is not None )   & (levels <= MAX_LEVEL) )  | 
                (  ( MAX_LEVEL is None )  &  (levels == LEVEL) )
                    )
    train = train[train_filter].reset_index(drop=True)
    train_head = train.iloc[:, :6]  
    daily_sales = daily_sales[train_filter].reset_index(drop=True)
    levels = levels[train_filter].reset_index(drop=True).astype(np.int8)

    print('Train data loaded and filtered.')
    print('Train shape:', train.shape)
    print(train.head())

    # replace leading zeros with nan
    train['d_1'].replace(0, np.nan, inplace=True)

    for i in range(train.columns.get_loc('d_1') + 1, train.shape[1]):
        train.loc[:, train.columns[i]].where( ~ ((train.iloc[:,i]==0) & (train.iloc[:,i-1].isnull())),
                                            np.nan, inplace=True)

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    train_flipped = train.set_index('id', drop = True).iloc[:, 5:].transpose()

    print('Train flipped shape:', train_flipped.shape)
    print(train_flipped.dtypes)
    print(train_flipped.max().sort_values(ascending=False)[::3000])

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    ramCheck()

    features = []

    store_avg_qs = train_flipped[train_flipped.columns[levels >= 12]].transpose()\
                .groupby(train_head.iloc[(levels >= 12).values].store_id.values).mean().fillna(1)
    store_dept_avg_qs = train_flipped[train_flipped.columns[levels >= 12]].transpose()\
                .groupby(  ( train_head.iloc[(levels >= 12).values].store_id.astype(str) + '_'
                            + train_head.iloc[(levels >= 12).values].dept_id.astype(str)).values
                        ).mean().fillna(1)

    scaled_sales = train_flipped / (store_avg_qs.loc[train.store_id].transpose().values); 
    del store_avg_qs, store_dept_avg_qs,

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    arrs = [train_flipped, scaled_sales, ] # sales_over_all]
    labels = ['qs', 'qs_divbystore', ] #'qs_divbyall']

    if REDUCED_FEATURES: arrs = arrs[0:1]

    # Build features from scratch, similar to m5_lgbm_winner.py
    print("Building features from scratch...")
    
    # basic lag features
    for lag in range(1, 10+1):
        if REDUCED_FEATURES: continue;
        features.append( ('qs_lag_{}d'.format(lag),
                            train_flipped.shift(lag).fillna(0).astype(np.half) ) )
        
    # means and medians -- by week to avoid day of week effects
    for idx in range(0, len(arrs)):
        arr = arrs[idx]
        label = labels[idx]

        for window in [7, 14, 21, 28, 28*2, 28*4, ]:  ## ** mean and median
            if REDUCED_FEATURES and window != 28: continue;
            features.append( ('{}_mean_{}d'.format(label, window), 
                        arr.rolling(window).mean().astype(np.half) )  )

            features.append( ('{}_median_{}d'.format(label, window), 
                        arr.rolling(window).median().astype(np.half) )  )
            
            print('{}: {}'.format(label,window))
        clear_memory()
        del arr
    clear_memory()

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    # stdev, skewness, and kurtosis
    # ideally kurtosis and skewness should NOT be labeled qs_ as they are scale-invariant
    for idx in range(0, len(arrs)):
        arr = arrs[idx]
        label = labels[idx]
        for window in [7, 14, 28, 28*3, 28*6]:
            if REDUCED_FEATURES and window != 28: continue;
            print('{}: {}'.format(label,window))

            features.append( ('{}_stdev_{}d'.format(label, window), 
                                arr.rolling(window).std().astype(np.half) )  )

            if window >= 10:
                if REDUCED_FEATURES: continue;
                features.append( ('{}_skew_{}d'.format(label, window), 
                                    arr.rolling(window).skew().astype(np.half) )  )

                features.append( ('{}_kurt_{}d'.format(label, window), 
                                    arr.rolling(window).kurt().astype(np.half) )  )
        clear_memory()
        del arr
    clear_memory()

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    # high and low quantiles (adding more seemed to hurt performance)
    for idx in range(0, len(arrs)):
        arr = arrs[idx]
        label = labels[idx]
        for window in [14, 28, 56]:
            if REDUCED_FEATURES and window != 28: continue;

            features.append( ('{}_qtile10_{}d'.format(label, window), 
                        arr.rolling(window).quantile(0.1).astype(np.half) )  )

            features.append( ('{}_qtile90_{}d'.format(label, window), 
                        arr.rolling(window).quantile(0.9).astype(np.half) )  )

            print('{}: {}'.format(label,window))
            clear_memory()
        del arr
    clear_memory()

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    del arrs; del scaled_sales

    # Clean features after building
    clean_features(features, cal)

    print('Shape of features:', len(features), 'features')
        
    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    ramCheck()
    cal_features = pd.DataFrame()

    cal_features['dayofweek'] =  cal.date.dt.dayofweek.astype(np.int8)
    cal_features['dayofmonth'] =  cal.date.dt.day.astype(np.int8)
    cal_features['season'] =  cal.date.dt.month.astype(np.half)

    state_cal_features = []
    snap_cols = [c for c in cal.columns if 'snap' in c]
    state_cal_features.append( ( 'snap_day' , 
                                    cal[snap_cols].astype(np.int8) ) )
    state_cal_features.append( ( 'snap_day_lag_1' , 
                                    cal[snap_cols].shift(1).fillna(0).astype(np.int8) ) )
    state_cal_features.append( ( 'snap_day_lag_2' , 
                                    cal[snap_cols].shift(2).fillna(0).astype(np.int8) ) )
    state_cal_features.append( ( 'nth_snap_day',
                (cal[snap_cols].rolling(15, min_periods = 1).sum() * cal[snap_cols] ).astype(np.int8)  ) )

    for window in [2, 5, 10, 30, 60]:
        state_cal_features.append( ('snap_{}d_ewm'.format(window),
                                        cal[snap_cols].ewm(span = window, adjust=False).mean().astype(np.half) ) )
        
    # strip columns to match state_id
    def snapRename(x):
        return x.replace('snap_', '')

    for f in range(0, len(state_cal_features)):
        state_cal_features[f] = (state_cal_features[f][0],
                                    state_cal_features[f][1].rename(snapRename, axis = 'columns')) 
        

    for etype in [c for c in cal.event_type_1.dropna().unique()]:
        cal[etype.lower() + '_holiday'] = np.where(cal.event_type_1 == etype,
                                        cal.event_name_1,
                                                np.where(cal.event_type_2 == etype,
                                                        cal.event_name_2, 'None'))

    for etype in [c for c in cal.event_type_1.dropna().unique()]:
        cal[etype.lower() + '_holiday'] = cal[etype.lower() + '_holiday'].astype('category')

    print(ramCheck())

    series_to_series_id = dict([(col, idx) for idx, col in enumerate(train_flipped.columns)])
    series_id_to_series = dict([(idx, col) for idx, col in enumerate(train_flipped.columns)])
    series_id_level = dict([(idx, col) for idx, col in enumerate(levels)])
    series_level = dict(zip(train_flipped.columns, levels))

    series_to_item_id = dict([(x[1].id, x[1].item_id) for x in train_head[['id', 'item_id']].iterrows()])

    for feature in features:
        assert feature[1].shape == features[0][1].shape

    fstack = features[0][1].stack(dropna = False)
    series_features = pd.DataFrame({'d': fstack.index.get_level_values(0) \
                                                    .map(day_to_cal_index).values.astype(np.int16),
                        'series': fstack.index.get_level_values(1) \
                                                    .map(series_to_series_id).values.astype(np.int16)  })
    del fstack

    for idx, feature in enumerate(features):
        if feature is not None:
            series_features[feature[0]] = feature[1].stack(dropna=False).values
            
    del features 

    print(ramCheck())

    for feature in state_cal_features:
        assert feature[1].shape == state_cal_features[0][1].shape

    fstack = state_cal_features[0][1].stack(dropna = False)

    state_cal_series_features = pd.DataFrame({'d': fstack.index.get_level_values(0) \
                                                    .map(day_to_cal_index).values.astype(np.int16),
                        'state': fstack.index.get_level_values(1)  })
    del fstack

    for idx, feature in enumerate(state_cal_features):
        if feature is not None:
            state_cal_series_features[feature[0]] = feature[1].stack(dropna=False).values
            
    series_features.isnull().sum().sum()
    series_features.fillna(-10, inplace=True)


    CATEGORICALS = ['dept_id', 'cat_id', 'store_id', 'state_id', ] # 'item_id'] # never item_id; wrecks higher layers;

            
    for col in CATEGORICALS:
        series_features[col] = series_features.series.map(series_id_to_series).map(
                    train_head.set_index('id')[col]) #.astype('category')

    print(ramCheck())
    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    trailing_28d_sales = daily_sales.iloc[:,6:].transpose().rolling(28, min_periods = 1).sum().astype(np.float32)

    fstack = train_flipped.stack(dropna = False)
    weight_stack = pd.DataFrame({'d': fstack.index.get_level_values(0) \
                                                    .map(day_to_cal_index).values.astype(np.int16),
                        'series': fstack.index.get_level_values(1) \
                                                    .map(series_to_series_id).values.astype(np.int16),
                        'days_since_first': (~train_flipped.isnull()).expanding().sum().stack(dropna = False).values\
                                                .astype(np.int16),
                        'trailing_vol': ( (train_flipped.diff().abs()).expanding().mean() ).astype(np.float16)\
                                                    .stack(dropna = False).values,
                        'weights': (trailing_28d_sales / 
                                        trailing_28d_sales.transpose().groupby(levels).sum().loc[levels].transpose().values)
                                        .astype(np.float32)\
                                                .stack(dropna = False).values,
                                })

    del fstack
    del trailing_28d_sales
    print(weight_stack.dtypes)

    new_items = weight_stack.days_since_first < 30
    weight_stack[new_items].weights.sum() / weight_stack[weight_stack.days_since_first >= 0].weights.sum()
    weight_stack.loc[new_items, 'weights'] = 0

    print(ramCheck())
    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    print("Before cleaning:")
    print(f"- weight_stack.shape: {weight_stack.shape}")
    print(f"- series_features.shape: {series_features.shape}")

    # Apply the cleaning
    weight_stack = clean_df(weight_stack, cal, day_to_cal_index)

    print("After cleaning:")
    print(f"- weight_stack.shape: {weight_stack.shape}")  
    print(f"- series_features.shape: {series_features.shape}")

    print(f'Weight Stack index: {weight_stack.index}')
    print(f'Series Features index: {series_features.index}')
    print(f'Weight Stack columns: {weight_stack.columns}')
    print(f'Series Features columns: {series_features.columns}')

    # # Replace the merge operation with index-based filtering
    # if len(weight_stack) != len(series_features):
    #     print(f"Aligning dataframes with different lengths")
        
    #     # Create a composite key for fast lookup
    #     weight_stack['key'] = weight_stack['d'].astype(str) + '_' + weight_stack['series'].astype(str)
    #     series_features['key'] = series_features['d'].astype(str) + '_' + series_features['series'].astype(str)
        
    #     # Find common keys
    #     common_keys = set(weight_stack['key']).intersection(set(series_features['key']))
    #     print(f"Found {len(common_keys)} common keys")
        
    #     # Filter both dataframes to the same set of keys
    #     weight_stack = weight_stack[weight_stack['key'].isin(common_keys)]
    #     series_features = series_features[series_features['key'].isin(common_keys)]
        
    #     # Drop the temporary key column
    #     weight_stack.drop('key', axis=1, inplace=True)
    #     series_features.drop('key', axis=1, inplace=True)
        
    #     # Reset indexes to ensure they're aligned
    #     weight_stack.reset_index(drop=True, inplace=True)
    #     series_features.reset_index(drop=True, inplace=True)
        
    #     print(f"After alignment:")
    #     print(f"- weight_stack.shape: {weight_stack.shape}")
    #     print(f"- series_features.shape: {series_features.shape}")
        

    assert len(weight_stack) == len(series_features)
    assert (weight_stack.d.values == series_features.d).all()
    assert (weight_stack.series.values == series_features.series).all()

    series_features = pd.concat( (series_features, 
                    weight_stack.reset_index(drop=True).iloc[:, -2:]), axis = 1,)
    weight_stack = weight_stack.iloc[:10, :]

    fstack = train_flipped.stack(dropna = False)
    y_full = pd.DataFrame({'d': fstack.index.get_level_values(0) \
                                                    .map(day_to_cal_index).values.astype(np.int16),
                        'series': fstack.index.get_level_values(1) \
                                                    .map(series_to_series_id).values.astype(np.int16),
                        'y': fstack.values})
    del fstack

    print(ramCheck())



    for level in sorted(clf_set.keys()):
        X = getSubsample(0.0001, level, 0.1, series_features=series_features, 
                series_id_level=series_id_level, cal=cal, cal_features=cal_features, 
                state_cal_feats=state_cal_series_features, train_flipped=train_flipped,
                y_full=y_full, weight_stack=weight_stack, validation=VALIDATION)[0]
        print("Level {}:".format(level))
        for idx, q in enumerate(LEVEL_QUANTILES[level]):
            f = avg_FI([[q_clfs[idx] for clfs in clf_set[level] for q_clfs in clfs]], X.columns, 25, 
                        title = "Level {} \u03BC={} Feature Importances".format(level, q))
        print(); print()


    qls = {}; all_predictions = {}
    for level in sorted(set(clf_set.keys()) & set(levels)):
        print("\n\n\nLevel {}\n\n\n".format(level))
        QUANTILES = LEVEL_QUANTILES[level]
        
        SS_FRAC, SCALE_RANGE = P_DICT[level] #  if level < 12 else ID_FILTER]; 
        SS_FRAC = SS_FRAC * SS_SS 
        EVAL_FRAC = SS_FRAC * (1 if level < 11 else 1/2) 
        EVAL_PWR = 0.6
        SCALE_RANGE_TEST = SCALE_RANGE
        
        np.random.seed(RSEED)
        X, y, groups, scalers = getSubsample(EVAL_FRAC * level_os[level] ** EVAL_PWR, level, 
                                            SCALE_RANGE_TEST, 
                                            n_repeats = N_REPEATS if level < 15 else N_REPEATS//2, 
                                            drops=False, series_features=series_features, 
                                            series_id_level=series_id_level, cal=cal, cal_features=cal_features, 
                                            state_cal_feats=state_cal_series_features, train_flipped=train_flipped,
                                            y_full=y_full, weight_stack=weight_stack, validation=VALIDATION)
        if len(X) == 0:
            print("No Data for Level {}".format(level))
            continue;
            
        y_pred, y_true, groups, scaler_stack, X = predictSet(X, y, groups, scalers, clf_set[level]); 
        # assert (y_true == y.values * scalers.trailing_vol).all()

        predictions = pd.DataFrame(y_pred.T, index=y_true.index, columns = QUANTILES)
        predictions['y_true'] = y_true.values
        predictions = pd.concat((predictions, scaler_stack), axis = 'columns')
        predictions['group'] = groups.values
        predictions['series'] = X.series
        predictions['d'] = X.d
        predictions['days_fwd'] = X.days_fwd
        
        
        
        losses = pd.DataFrame(index=QUANTILES)
        for group in groups.unique():
            subpred = predictions[predictions.group == group]
            q_losses = []
            for quantile in QUANTILES:
                q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                                    1, subpred.trailing_vol, quantile)))
            losses[group] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values
        qls[level] = [losses]    
        
        ramCheck()
        
        # now combine them
        predictions = predictions.groupby(['series', 'd', 'days_fwd']).agg(
                    dict([(col, 'mean') for col in predictions.columns 
                            if col not in ['series', 'd', 'days_fwd']]\
                            + [('days_fwd', 'count')])  )\
                .rename(columns = {'days_fwd': 'ct'}).reset_index()
        predictions.head()
        predictions.sort_values('ct', ascending = False).head(5)
        print(len(predictions))
        
        all_predictions[level] = predictions
        
        for level in sorted(all_predictions.keys()):
            predictions = all_predictions[level]
            
            losses = pd.DataFrame(index=LEVEL_QUANTILES[level])
            for group in groups.unique():
                subpred = predictions[predictions.group == group]
                q_losses = []
                for quantile in QUANTILES:
                    q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                                        subpred.ct, subpred.trailing_vol, quantile)))
                losses[group] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values
                
                
            qls[level] = [losses]
            
            print("\n\n\nLevel {} Year-by-Year OOS Losses for Evaluation Bag {}:".format(level, 1))
            print(losses); #print(); print()

    
        for level in sorted(all_predictions.keys()):
            predictions = all_predictions[level]

            predictions['future_d'] = predictions.d + predictions.days_fwd

            for quantile in QUANTILES:
                true = predictions.y_true
                pred = predictions[quantile]
                trailing_vol= predictions.trailing_vol

                predictions['loss_{}'.format(quantile)] = \
                    np.where(true >= pred, 
                                    quantile*(true-pred),
                                    (1-quantile)*(pred - true) ) / trailing_vol

            predictions['loss'] = predictions[[c for c in predictions.columns if 'loss_' in str(c)]].sum(axis = 1)  
            predictions['wtg_loss'] = predictions.loss * predictions.ct / predictions.ct.mean()   

        print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


        all_predictions = {}
        for level in sorted(list(set(levels.unique()) & set(clf_set.keys()))):
            print('\n\nCreating Out-of-Sample Predictions for Level {}\n'.format(level))
            
            final_base = FINAL_BASE

            assert (final_base in ['d_1941', 'd_1913'])
            if final_base == 'd_1941':
                suffix = 'evaluation'
            elif final_base == 'd_1913':
                suffix = 'validation'
                
            print('   predicting 28 days forward from {}'.format(final_base))
            final_features = series_features[( series_features.d.map(cal_index_to_day) == final_base) & 
                                                (series_features.series.map(series_id_level) == level) ]

            print('    for {} series'.format(len(final_features)))
            
            SS_FRAC, SCALE_RANGE = P_DICT[level] # if level < 12 else ID_FILTER]; 
            SS_FRAC = SS_FRAC * 0.8
            print('   scale range of {}'.format(SCALE_RANGE))
            
            
            if level <= 9 or SPEED:
                X = []
                for df in range(0,28):
                    Xi = final_features.copy()
                    Xi['days_fwd'] = df + 1
                    X.append(Xi)
                X = pd.concat(X, ignore_index = True); del Xi; del final_features;

                Xn = np.power(X.weights, 2)
                Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
                Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
                
                print('   average repeats: {:.0f}'.format(Xn.mean()))
                print('   median repeats: {:.0f}'.format(Xn.median()))
                print('   max repeats: {:.0f}'.format(Xn.max()))

                X = X.loc[np.repeat(Xn.index, Xn)]

                X, y, groups, scalers = getXYG(X, scale_range = SCALE_RANGE, oos = True, y_full=y_full, weight_stack=weight_stack, 
                              cal=cal, cal_features=cal_features, 
                              state_cal_feats=state_cal_series_features,
                              train_flipped=train_flipped,
                              validation=VALIDATION)
                Xd = X.d;  Xseries = X.series
                X.drop(columns=['d', 'series'], inplace = True)

                print(X.shape)
                y_pred = predictOOS(X, scalers, clf_set[level], LEVEL_QUANTILES[level], suffix == 'validation'); print()

                predictions = pd.DataFrame(y_pred.T, index=X.index, columns = LEVEL_QUANTILES[level])
                predictions = pd.concat((predictions, scalers), axis = 'columns')
                predictions['series'] = Xseries
                predictions['d'] = Xd
                predictions['days_fwd'] = X.days_fwd.astype(np.int8)
                predictions['y_true'] = y * scalers.scaler
        #         break;
                ramCheck()

                predictions = predictions.groupby(['series', 'd', 'days_fwd']).agg(
                                dict([(col, 'mean') for col in predictions.columns 
                                        if col not in ['series', 'd', 'days_fwd']]\
                                        + [('days_fwd', 'count')])  )\
                            .rename(columns = {'days_fwd': 'ct'}).reset_index()
                predictions.days_fwd = predictions.days_fwd.astype(np.int8)

            else: # levels 10, 11, 12
                
                predictions_full = []
                
                for df in range(0,28):
                    print( '\n Predicting {} days forward from {}'.format(df + 1, final_base))
                    X = final_features.copy()
                    X['days_fwd'] = df + 1

                    Xn = np.power(X.weights, 1.5)
                    Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
                    Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
                    
                    print('   average repeats: {:.0f}'.format(Xn.mean()))
                    print('   median repeats: {:.0f}'.format(Xn.median()))
                    print('   max repeats: {:.0f}'.format(Xn.max()))
                    
                    X = X.loc[np.repeat(Xn.index, Xn)]

                    X, y, groups, scalers = getXYG(X, scale_range = SCALE_RANGE, oos = True, y_full=y_full, weight_stack=weight_stack, 
                              cal=cal, cal_features=cal_features, 
                              state_cal_feats=state_cal_series_features,
                              train_flipped=train_flipped,
                              validation=VALIDATION)
                    Xd = X.d;  Xseries = X.series
                    X.drop(columns=['d', 'series'], inplace = True)

                    print(X.shape)
                    y_pred = predictOOS(X, scalers, clf_set[level], LEVEL_QUANTILES[level], suffix == 'validation'); print()

                    predictions = pd.DataFrame(y_pred.T, index=X.index, columns = LEVEL_QUANTILES[level])
                    predictions = pd.concat((predictions, scalers), axis = 'columns')
                    predictions['series'] = Xseries
                    predictions['d'] = Xd
                    predictions['days_fwd'] = X.days_fwd.astype(np.int8)
                    predictions['y_true'] = y * scalers.scaler

                    ramCheck()

                    predictions = predictions.groupby(['series', 'd', 'days_fwd']).agg(
                                    dict([(col, 'mean') for col in predictions.columns 
                                            if col not in ['series', 'd', 'days_fwd']]\
                                            + [('days_fwd', 'count')])  )\
                                .rename(columns = {'days_fwd': 'ct'}).reset_index()
                    predictions.days_fwd = predictions.days_fwd.astype(np.int8)
                    predictions_full.append(predictions)
                    
                predictions = pd.concat(predictions_full); del predictions_full
        
            all_predictions[level] = predictions; del predictions

            with open('all_predictions_raw.pkl', 'wb') as handle:
                pickle.dump(all_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
            losses = pd.DataFrame(index=LEVEL_QUANTILES[levels.min()])
            for level in sorted(all_predictions.keys()):
                predictions = all_predictions[level]
                subpred = predictions
                q_losses = []
                for quantile in LEVEL_QUANTILES[level]:
                    q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                                        subpred.weights, subpred.trailing_vol, quantile)))

            #         print(np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values)
                losses[level] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 7).values


            #         print("\n\n\nLevel {} Year-by-Year OOS Losses for Evaluation Bag {}:".format(level, 1))
            print(losses); print(); print()
            print(losses.mean())
            print(losses.mean().mean())