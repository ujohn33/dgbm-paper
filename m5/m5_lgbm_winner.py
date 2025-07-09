FINAL_BASE = ['d_1941', 'd_1913'][0]
IMPORT = False 
CACHED_FEATURES = False
CACHE_FEATURES = True

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
import pickle
from collections import Counter
import datetime as datetime
from scipy.stats.mstats import gmean
import random
import gc
import gzip
import torch
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

def aggTrain(train):
    tcd = dict([(col, 'first') for col in train.columns[1:6]])
    tcd.update( dict([(col, 'sum') for col in train.columns[6:]]))

    tadds =[]; tadd_levels= [ [12 for i in range(0, len(train))] ] 
    for idx, lvl in enumerate(LEVELS[1:]):
        level = lvl[0]
        lvls = lvl[1]

        if len(lvls) == 0:  # group all if no list provided
            lvls = [1 for i in range(0, len(train))]

        tadd = train.groupby(lvls).agg(tcd)

        # name it
        if len(lvls) == 2:
            tadd.index = ['_'.join(map(str,i)) for i in tadd.index.tolist()]
        elif len(lvls) == 1:
            tadd.index = tadd.index + '_X'
        else:
            tadd.index = ['Total_X']
        tadd.index.name = 'id'

        # fill in categorical features
        tadd.reset_index(inplace=True)
        for col in [c for c in train.columns[1:6] if c not in lvls and not  
                            any(c in z for z in[DOWNSTREAM[lvl] for lvl in lvls if lvl in DOWNSTREAM])]:
            tadd[col] = 'All'
        tadds.append(tadd)

        #levels
        tadd_levels.append([level for i in range(0, len(tadd))])

    train = pd.concat((train,*tadds), sort=False, ignore_index=True); del tadds, tadd
    levels = pd.Series(data = [x for sub_list in tadd_levels for x in sub_list], index = train.index); del tadd_levels
    for col in train.columns[1:6]:
        train[col] = train[col].astype('category')
        
    return train, levels

def loadTrain(path='data/m5-forecasting-uncertainty'):
    train_cols =  pd.read_csv(path+ '/' + 'sales_train_evaluation.csv', nrows=1)

    c_dict = {}
    for col in [c for c in train_cols if 'd_' in c]:
        c_dict[col] = np.float32

    train = pd.read_csv(path+ '/' + 'sales_train_evaluation.csv', dtype=c_dict)#.astype(np.int16, errors='ignore')

    train.id = train.id.str.split('_').str[:-1].str.join('_')
    
    train.sort_values('id', inplace=True)
    return train.reset_index(drop=True)

def getPricePivot(path='data/m5-forecasting-uncertainty'):
    prices = pd.read_csv(path+ '/' + 'sell_prices.csv',
                    dtype = {'wm_yr_wk': np.int16, 'sell_price': np.float32})
    prices['id'] = prices.item_id + "_" + prices.store_id
    price_pivot =  prices.pivot(columns = 'id' , index='wm_yr_wk', values = 'sell_price')
    price_pivot = price_pivot.reindex(sorted(price_pivot.columns), axis=1)
    return price_pivot

# start after one year, remove anything with proximity to holiday months (given mid-year LB targets)
# also saves a lot of RAM/processing time 

def clean_df(fr, cal=None, day_to_cal_index=None):
    if cal is None:
        # Try to load cal if not provided
        try:
            cal = getCal()
        except:
            print("Error: 'cal' variable is required but not provided")
            raise
    
    # Create day_to_cal_index if not provided
    if day_to_cal_index is None:
        day_to_cal_index = dict([(col, idx) for idx, col in enumerate(cal.index)])
            
    early_rows = cal[cal.year == cal.year.min()].index.to_list()
    holiday_rows = cal[cal.month.isin([10, 11, 12, 1])].index.to_list()
    delete_rows = early_rows + holiday_rows
    
    MIN_DAY = 'd_{}'.format(300)
    
    if 'd' in fr.columns: # d, series stack:
        fr = fr[fr.d >= day_to_cal_index[MIN_DAY]]
        fr = fr[~fr.d.isin([day_to_cal_index[d] for d in delete_rows])]
    else:  # pivot table
        if MIN_DAY in fr.index:
            fr = fr.iloc[fr.index.get_loc(MIN_DAY):, :]

        if len(delete_rows) > 0:
            fr = fr[~fr.index.isin(delete_rows)]

    return fr

def clean_features(features, cal=None):
    if cal is None:
        # Try to load cal if not provided
        try:
            cal = getCal()
        except:
            print("Error: 'cal' variable is required but not provided")
            raise
            
    for idx, feat_row in enumerate(features):
        fr = feat_row[1]
        fr = clean_df(fr, cal)

        if len(fr) < len(feat_row[1]):
            features[idx] = (features[idx][0], fr)

def getCal(path='data/m5-forecasting-uncertainty'):
    return pd.read_csv(path+ '/' + 'calendar.csv').set_index('d')

def addMAcrosses(X):
    EWMS = [c for c in X.columns if 'ewm' in c and 'qs_' in c and len(c) < 12]
    for idx1, col1 in enumerate(EWMS):
        for idx2, col2 in enumerate(EWMS):
            if not idx1 < idx2:
                continue;
            
            X['qs_{}_{}_ewm_diff'.format(col1.split('_')[1], col2.split('_')[1])] = X[col1] - X[col2]
            X['qs_{}_{}_ewm_ratio'.format(col1.split('_')[1], col2.split('_')[1])] = X[col1] / X[col2]
                
    return X

def addCalFeatures(X, cal, cal_features):  # large block of code; easy;
    cal.date = pd.to_datetime(cal.date)

    day_to_cal_index = dict([(col, idx) for idx, col in enumerate(cal.index)])
    cal_index_to_day = dict([(idx, col) for idx, col in enumerate(cal.index)]) 
    # day of week, month, season of year
    X['dayofweek'] = ( X.d + X.days_fwd).map(cal_index_to_day).map(cal_features.dayofweek)
    X['dayofmonth'] = ( X.d + X.days_fwd).map(cal_index_to_day).map(cal_features.dayofmonth)
 
    X['basedayofweek'] = X.d.map(cal_index_to_day).map(cal_features.dayofweek)
    X['dayofweekchg'] = (X.days_fwd % 7).astype(np.int8)

    X['basedayofmonth'] = X.d.map(cal_index_to_day).map(cal_features.dayofmonth)
    X['season'] =  ( ( X.d + X.days_fwd).map(cal_index_to_day).map(cal_features.season) \
                             + np.random.normal( 0, 1, len(X)) ).astype(np.half)
                        # with a full month SD of noise to not overfit to specific days;

    # holidays
    holiday_cols = [c for c in cal.columns if '_holiday' in c]
    for col in holiday_cols:
        X['base_' + col] = X.d.map(cal_index_to_day).map(cal[col])
        X[col] = ( X.d + X.days_fwd).map(cal_index_to_day).map(cal[col])

    return X

def convertToLinearFeatures(X):
    X = X.copy()
    for s in X.dayofweek.unique():
        X['dayofweek_{}'.format(s)] = (X.dayofweek == s).astype(np.int8)
    X.drop( columns = X.columns[X.dtypes == 'category'], inplace=True)
    X['daysfwd_sqrt'] = (X.days_fwd ** 0.5).astype(np.half)
    
    return X

def addStateCalFeatures(X, state_cal_series_features):  
    if (X.state_id == 'All').mean() > 0:
        print('No State Ids')
        return X;
    
    def rename_scf(c, name = 'basedate'):
        return c if (c=='d' or c == 'state') else name + '_' + c
    
    X['future_d'] = ( X.d + X.days_fwd)
    X['state'] = X.state_id.astype('object')
    
    nX = X.merge(state_cal_series_features[['state', 'd', 'snap_day', 'nth_snap_day']]
                 .rename(rename_scf, axis = 'columns'),
                                         on = ['d', 'state'],  
             validate='m:1', how = 'inner', suffixes = (False, False)) 
    
    
    nX = nX.merge(state_cal_series_features[['state', 'd', 'snap_day', 'nth_snap_day']]
                 .rename(columns = {'d': 'future_d'}), 
                                         on = ['future_d', 'state'],  
             validate='m:1', how = 'inner', suffixes = (False, False)) 
    
    nX.drop(columns = ['state', 'future_d'], inplace=True)
    
    assert len(nX) == len(X)
    
    return nX

def add_item_features(X):  
    return X

def getXYG(X, scale_range = None, oos = False, y_full=None, weight_stack=None, 
           cal=None, cal_features=None, state_cal_feats=None, train_flipped=None,
           validation=VALIDATION):  # Use the imported VALIDATION as default
    start_time = datetime.datetime.now(); 

    # ensure it's in the train set, and days_forward is actually *forward*
    X.drop( X.index[ (X.days_fwd < 1) |
           (  ~oos  &  ( X.d + X.days_fwd > cal.index.get_loc(train_flipped.index[-1])  )    ) ], inplace=True)
    g = gc.collect()
    
    
    X = addMAcrosses(X)

    X = addCalFeatures(X, cal, cal_features)
    X = addStateCalFeatures(X, state_cal_feats)
    
    # noise to time-static features
    for col in [c for c in X.columns if 'store' in c and 'ratio' in c]:
        X[col] = X[col] + np.random.normal(0, 0.1, len(X))
        print('adding noise to {}'.format(col))
    

    # match with Y
    if 'y' not in X.columns:
        st = datetime.datetime.now(); 
        X['future_d'] = X.d + X.days_fwd
        if oos:  
            X = X.merge(y_full.rename(columns = {'d': 'future_d'}), on = ['future_d', 'series'], 
                             how = 'left')
            X.y = X.y.fillna(-1)
            
        else:  
            X = X.merge(y_full.rename(columns = {'d': 'future_d'}), on = ['future_d', 'series'],
                       )#    suffixes = (None, None), validate = 'm:1')
#     X['yo'] = X.y.copy()
    g = gc.collect()
    
    scaler_columns = [c for c in X.columns if c in weight_stack.columns[2:]]
    scalers = X[scaler_columns].copy()
    y = X.y
    
    groups = pd.Series(cal.iloc[(X.d + X.days_fwd)].year.values, X.index).astype(np.int16)
    
    
    # feature drops
    if REDUCED_FEATURES:
        feat_drops = [c for c in X.columns if c not in (sparse_features + ['d', 'series', 'days_fwd'])]
    
    elif len(FEATURE_DROPS) > 0:
        feat_drops = [c for c in X.columns if any(z in c for z in FEATURE_DROPS )]
        print('dropping {} features; anything containing {}'.format(len(feat_drops), FEATURE_DROPS))
        print('   -- {}'.format(feat_drops))
    else:
        feat_drops = []
        
    # final drops
    X.drop(columns = scaler_columns + (['future_d'] if 'future_d' in X.columns else []) + ['y'] + feat_drops , inplace=True)

    scalers['scaler'] = scalers.trailing_vol.copy()
    
    # randomize scaling
    if scale_range > 0:
        scalers.scaler = scalers.scaler * np.exp( scale_range * ( np.random.normal(0, 0.5, len(X))) )
#         scalers.scaler = scalers.scaler * np.exp( scale_range * ( np.random.rand(len(X)) - 0.5) )
    
    # now rescale y and  'scaled variable' in X by its vol
    for col in [c for c in X.columns if 'qs_' in c and 'ratio' not in c]:
        X[col] = np.where( X[col] == -10, X[col], (X[col] / scalers.scaler).astype(np.half)) 
    y = y / scalers.scaler
    
    
    yn = (oos == False) & (y.isnull() | (groups==VALIDATION)) 

    
    print("\nXYG Pull Time: {}".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
    
    return (X[~yn], y[~yn], groups[~yn], scalers[~yn])

def getSubsample(frac, level = 12, scale_range = 0.1, n_repeats = 1, drops = True, post_process_X = None, series_features=None, series_id_level=None, 
                cal=None, cal_features=None, state_cal_feats=None, train_flipped=None, y_full=None, weight_stack=None, validation=VALIDATION):
    start_time = datetime.datetime.now();

    wtg_mean = series_features.weights[(series_features.series.map(series_id_level) == level)].mean()
    ss = series_features.weights / wtg_mean * frac
    
    X = series_features[  (ss > np.random.rand(len(ss)) ) 
                              & (series_features.series.map(series_id_level) == level) ]
    ss =  X.weights / wtg_mean   * frac 
      
    print('{} series that seek oversampling'.format( (ss > 1). sum() ) )
    print( ss[ss>1].sort_values()[-5:])
    
    extras = []
    
    while ss.max() > 1:
        ss = ss - 1
        extras.append( X[ ss > np.random.rand(len(ss))] )
        
    # Replace lines 377-379 with:
    if len(extras) > 0:
        print(' scaled EWMS of extras:')
        if 'qs_30d_ewm' in extras[-1].columns:
            print((extras[-1].qs_30d_ewm / extras[-1].trailing_vol)[-5:])
        else:
            print("Column 'qs_30d_ewm' not found. Available columns:", 
                [col for col in extras[-1].columns if 'qs_' in col])
            # Use a different column as fallback
            available_cols = [col for col in extras[-1].columns if 'qs_' in col and '_mean_' in col]
            if available_cols:
                print(f"Using {available_cols[0]} as fallback")
                print((extras[-1][available_cols[0]] / extras[-1].trailing_vol)[-5:])

    if len(extras) > 0:
        X = pd.concat((X, *extras))
    else:
        X = X.copy()
    
    
    X['days_fwd'] = (np.random.randint(0, 28, size = len(X)) + 1).astype(np.int8)
    
    if n_repeats > 1:
         X = pd.concat([X] * n_repeats)

    g = gc.collect()

    X, y, groups, scalers = getXYG(X, scale_range, y_full=y_full, weight_stack=weight_stack, cal=cal, cal_features=cal_features, state_cal_feats=state_cal_feats, train_flipped=train_flipped, validation=VALIDATION)
    ramCheck()
    g = gc.collect()
    if drops:
        X.drop(columns = ['d', 'series'], inplace=True)
    
    if post_process_X is not None:
        X = post_process_X(X)
    
    print(X.shape)
    print("\nSubsample Time: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))

    return X, y, groups, scalers

def quantile_loss(true, pred, quantile = 0.5):
    loss = np.where(true >= pred, 
                        quantile*(true-pred),
                        (1-quantile)*(pred - true) )
    return np.mean(loss)   
 
def quantile_scorer(quantile = 0.5):
    return make_scorer(quantile_loss, greater_is_better=False, quantile = quantile)

def trainLGBquantile(x, y, groups, cv = 0, n_jobs = -1, alpha = 0.5, **kwargs):
    clfargs = kwargs.copy(); clfargs.pop('n_iter', None)
    clf = lgb.LGBMRegressor(verbosity=-1, hist_pool_size = 1000,  objective = 'quantile', alpha = alpha,
                            importance_type = 'gain', num_threads=4,
                            seed = datetime.datetime.now().microsecond if TIME_SEED else None,
                             **clfargs,
                      )
    print('\n\n Running Quantile Regression for \u03BC={}\n'.format(alpha))
    params = lgb_quantile_params
    
    return trainModel(x, y, groups, clf, params, quantile_scorer(alpha), n_jobs, **kwargs)

def trainModel(x, y, groups, clf, params, cv = 0, n_jobs = None, 
                   verbose=0, splits=None, **kwargs):
    if n_jobs is None:
        n_jobs = 2
    folds = LeaveOneGroupOut()
    clf = RandomizedSearchCV(clf, params, cv=  folds, 
                             n_iter= ( kwargs['n_iter'] if len(kwargs) > 0 and 'n_iter' in kwargs else 4), 
                            verbose = 0, n_jobs = n_jobs, scoring = cv)
    f = clf.fit(x, y, groups=groups)
    print(pd.DataFrame(clf.cv_results_['mean_test_score'])); print();  

    best = clf.best_estimator_;  print(best)
    print("\nBest In-Sample CV: {}\n".format(np.round(clf.best_score_,4)))

    return best

def runQBags(n_bags = 3, model_type = trainLGBquantile, data = None, quantiles = [0.5], **kwargs):
    start_time = datetime.datetime.now(); 
    
    clf_set = []; loss_set = []
    for bag in range(0, n_bags):
        print('\n\n  Running Bag {} of {}\n\n'.format(bag+1, n_bags))
        if data is None:
            X, y, groups, scalers = getSubsample()
        else:
            X, y, groups, scalers = data

        group_list = [*dict.fromkeys(groups)]   
        group_list.sort()
        print("Groups: {}".format(group_list))

        clfs = []; preds = []; ys=[]; datestack = []; losses = pd.DataFrame(index=QUANTILES)
        if SINGLE_FOLD: group_list = group_list[-1:]
        for group in group_list:
            print('\n\n   Running Models with {} Out-of-Fold\n\n'.format(group))
            x_holdout = X[groups == group]
            y_holdout = y[groups == group]
            
            ramCheck()
            model = model_type 
            
            q_clfs = []; q_losses = []
            for quantile in quantiles:
                set_filter = (groups != group) \
                        & (np.random.rand(len(groups)) < 
                                 quantile_wts[quantile] ** (0.35 if LEVEL >=11 else 0.25) )
                clf = model(X[set_filter], y[set_filter], groups[set_filter], 
                                alpha = quantile, **kwargs) 
                q_clfs.append(clf)

                predicted = clf.predict(x_holdout)

                q_losses.append((quantile, quantile_loss(y_holdout, predicted, quantile)))
                print(u"{} \u03BC={:.3f}: {:.4f}".format(group, quantile, q_losses[-1][1] ) )
                
                preds.append(predicted)
                ys.append(y_holdout)
            
            clfs.append(q_clfs)
            print("\nLevel {} OOS Losses for Bag {} in {}:".format(level, bag+1, group))
            print(np.round(pd.DataFrame(q_losses).set_index(0)[1], 4))
            losses[group] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values
            print("\nElapsed Time So Far This Bag: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
            
        
        clf_set.append(clfs)
        print("\nLevel {} Year-by-Year OOS Losses for Bag {}:".format(level, bag, group))
        print(losses)
        
        loss_set.append(losses)
        print("\nModel Bag Time: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
    return clf_set, loss_set


if __name__ == "__main__":
    path = 'data/m5-forecasting-uncertainty'

    print(ramCheck())


    cal = getCal(path)
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

    # basic lag features
    if not CACHED_FEATURES:
        for lag in range(1, 10+1):
            if REDUCED_FEATURES: continue;
            features.append( ('qs_lag_{}d'.format(lag),
                                train_flipped.shift(lag).fillna(0).astype(np.half) ) )
            
    # means and medians -- by week to avoid day of week effects
    if not CACHED_FEATURES:
        for idx in range(0, len(arrs)):
            arr = arrs[idx]
            label = labels[idx]

            for window in [7, 14, 21, 28, 28*2, 28*4,  ]:  ## ** mean and median
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

    if not CACHED_FEATURES:
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

    if not CACHED_FEATURES:
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



    # if CACHED_FEATURES:
    #     if 'features.pbz2' in os.listdir(pickle_dir):
    #         with bz2.BZ2File(pickle_dir + 'features.pbz2', 'r') as handle:
    #             features = pickle.load(handle)
    #     elif 'features.pgz' in os.listdir(pickle_dir):
    #         with gzip.GzipFile(pickle_dir + 'features.pgz', 'r') as handle:
    #             features = pickle.load(handle)
            
    clean_features(features)

    print('Shape of features:', len(features), 'features')

    if CACHE_FEATURES:
        with gzip.GzipFile('features.pgz', 'w') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.path.getsize('features.pgz') / 1e9

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

    weight_stack = clean_df(weight_stack)
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



    VALIDATION = -1; # 2016 # pure holdout from train and prediction sets;

    lgb_quantile_params = {     # fairly well tuned, with high runtimes 
                    'max_depth': [10, 20],
                    'n_estimators': [   200, 300, 350, 400, ],   
                    'min_split_gain': [0, 0, 0, 0, 1e-4, 1e-3, 1e-2, 0.1],
                    'min_child_samples': [ 2, 4, 7, 10, 14, 20, 30, 40, 60, 80, 100, 130, 170, 200, 300, 500, 700, 1000 ],
                    'min_child_weight': [0, 0, 0, 0, 1e-4, 1e-3, 1e-3, 1e-3, 5e-3, 2e-2, 0.1 ],
                    'num_leaves': [ 20, 30, 30, 30, 50, 70, 90, ],
                    'learning_rate': [  0.02, 0.03, 0.04, 0.04, 0.05, 0.05, 0.07, ],         
                    'colsample_bytree': [0.3, 0.5, 0.7, 0.8, 0.9, 0.9, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    'colsample_bynode':[0.1, 0.15, 0.2, 0.2, 0.2, 0.25, 0.3, 0.5, 0.65, 0.8, 0.9, 1],
                    'reg_lambda': [0, 0, 0, 0, 1e-5, 1e-5, 1e-5, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100   ],
                    'reg_alpha': [0, 1e-5, 3e-5, 1e-4, 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 1, 10, 10, 100, 1000,],
                    'subsample': [  0.9, 1],
                    'subsample_freq': [1],
                    'cat_smooth': [0.1, 0.2, 0.5, 1, 2, 5, 7, 10],
    }

    lgb_quantile_params["device"] = ['cuda'] if torch.cuda.is_available() else ['cpu']


    level_os = dict([(idx, 1/val) for (idx,val) in level_multiplier.items()])

    # these are to use less processing time on edge quantiles 
    QUANTILE_LEVELS = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
    QUANTILE_WTS = [0.1, 0.2, 0.6, 0.8, 1, 0.9, 0.7, 0.2, 0.1,]
        
    quantile_wts = dict(zip(QUANTILE_LEVELS, QUANTILE_WTS))

    print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')

    if not IMPORT:
        clf_set = {}; loss_set = {}; LEVEL_QUANTILES = {};
        for level in sorted(levels.unique()):
            print("\n\n\nRunning Models for Level {}\n\n\n".format(level))
            
            SS_FRAC, SCALE_RANGE = P_DICT[level] # if level < 12 else ID_FILTER]; 
            SS_FRAC = SS_FRAC * SS_SS
            print('{}/{}'.format(SS_FRAC, SCALE_RANGE))
            
            # much higher iteration counts for low levels
            clf_set[level], loss_set[level] = runQBags(n_bags = int(BAGS * level_os[level] ** BAGS_PWR), 
                                                    model_type = trainLGBquantile, 
                                                    data = getSubsample(SS_FRAC * level_os[level] ** SS_PWR, 
                                                                        level, SCALE_RANGE, n_repeats = N_REPEATS,
                                                                        drops = not CACHED_FEATURES,
                                                                        post_process_X = add_item_features,
                                                                        series_features=series_features,
                                                                        series_id_level=series_id_level,
                                                                        cal=cal, cal_features=cal_features,
                                                                        state_cal_feats=state_cal_series_features,
                                                                        train_flipped=train_flipped,
                                                                        y_full=y_full,
                                                                        weight_stack=weight_stack,
                                                                        validation=VALIDATION),
                                                            n_iter =  int( 
                                                                    (2.2 if level <= 9 else 1.66) 
                                                                    * (16 - (level if level <=12 else 12) ) 
                                                                        * (1/4 if SUPER_SPEED else (1/2 if SPEED else 1))   
                                                                        ) ,
                        quantiles = QUANTILES,
                        n_jobs = N_JOBS) 
            
            LEVEL_QUANTILES[level] = QUANTILES

    for level in sorted(clf_set.keys()):
        print("Level {}:".format(level))
        
        for idx, q in enumerate(LEVEL_QUANTILES[level]):
            print(u'\n\n      Regressors for \u03BC={}:\n'.format(q))
            for clf in [q_clfs[idx] for clfs in clf_set[level] for q_clfs in clfs]:
                print(clf)
        
        print(); print()

    # save classifiers
    clf_file = ('clf_set.pkl' if IMPORT 
                            else ('lvl_{}_clfs.pkl'.format(LEVEL) if MAX_LEVEL == None 
                                                                else 'lvls_lt_{}_clfs.pkl'.format(MAX_LEVEL)))
    with open(clf_file, 'wb') as handle:
        pickle.dump(clf_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


