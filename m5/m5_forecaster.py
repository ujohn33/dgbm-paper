#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object-oriented implementation for M5 forecasting models
Specialized for levels 13 (HOBBIES), 14 (HOUSEHOLD), 15 (FOODS) with SPEED and SUPER_SPEED options.
"""

import os
import gc
import bz2
import gzip
import pickle
import psutil
import random
import numpy as np
import pandas as pd
import datetime as dt
import lightgbm as lgb
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats.mstats import gmean
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer

class M5Config:
    """Configuration class for M5 forecasting models"""
    
    def __init__(self, level=13, super_speed=True):
        """
        Initialize configuration
        
        Args:
            level (int): Which level to process (13=HOBBIES, 14=HOUSEHOLD, 15=FOODS)
            super_speed (bool): Whether to use super speed mode
        """
        self.level = level
        self.max_level = None
        self.import_models = False
        self.final_base = 'd_1941'  # 'd_1913' for validation
        self.single_fold = True
        self.speed = True if not super_speed else False
        self.super_speed = super_speed
        self.reduced_features = False
        self.time_seed = True
        self.bags = 1
        self.n_jobs = -1
        self.ss_pwr = 0.6
        self.bags_pwr = 0
        self.cached_features = False
        self.cache_features = False
        self.ss_ss = 0.8  # 0.8 was production version
        
        # Adjust SS_SS based on speed and feature settings
        if self.speed or self.super_speed or self.reduced_features:
            self.ss_ss /= (5 if self.super_speed else (2 if self.speed else 1)) * (5 if self.reduced_features else 1)
        
        # Setting level-specific parameters
        self.level_splits = [(13, 'HOBBIES'), (14, 'HOUSEHOLD'), (15, 'FOODS')]
        self.quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
        
        # Parameters dictionary for different levels
        self.p_dict = {
            13: (0.12, 2),
            14: (0.065, 2),
            15: (0.03, 0.5)
        }
        
        # Core sparse features
        self.sparse_features = [
            'dayofweek', 'dayofmonth', 
            'qs_30d_ewm', 'qs_100d_ewm',
            'qs_median_28d', 'qs_mean_28d',
            'qs_qtile90_28d',
            'pct_nonzero_days_28d',
            'days_fwd'
        ]
        
        # Features to drop
        self.feature_drops = [
            'item_id', '_abs_diff', 'squared_diff',
            '336', '300d'
        ]
        
        # Set LightGBM parameters based on speed mode
        self.lgb_params = self._get_lgb_params()
        
    def _get_lgb_params(self):
        """Get LightGBM parameters based on speed settings"""
        if self.speed or self.super_speed or self.reduced_features:
            return {
                'max_depth': [10, 20],
                'n_estimators': [150, 200, 200],
                'min_split_gain': [0, 0, 0, 0, 1e-4, 1e-3, 1e-2, 0.1],
                'min_child_samples': [2, 4, 7, 10, 14, 20, 30, 40, 60, 80, 100, 100, 100, 
                                      130, 170, 200, 300, 500, 700, 1000],
                'min_child_weight': [0, 0, 0, 0, 1e-4, 1e-3, 1e-3, 1e-3, 5e-3, 2e-2, 0.1],
                'num_leaves': [20, 30, 50, 50],
                'learning_rate': [0.04, 0.05, 0.07, 0.07, 0.07, 0.1, 0.1, 0.1],
                'colsample_bytree': [0.3, 0.5, 0.7, 0.8, 0.9, 0.9, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'colsample_bynode': [0.1, 0.15, 0.2, 0.2, 0.2, 0.25, 0.3, 0.5, 0.65, 0.8, 0.9, 1],
                'reg_lambda': [0, 0, 0, 0, 1e-5, 1e-5, 1e-5, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                'reg_alpha': [0, 1e-5, 3e-5, 1e-4, 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 1, 10, 10, 100, 1000],
                'subsample': [0.9, 1],
                'subsample_freq': [1],
                'cat_smooth': [0.1, 0.2, 0.5, 1, 2, 5, 7, 10],
            }
        else:
            # Full parameters for standard speed
            return {
                'max_depth': [10, 20],
                'n_estimators': [200, 300, 350, 400],
                'min_split_gain': [0, 0, 0, 0, 1e-4, 1e-3, 1e-2, 0.1],
                'min_child_samples': [2, 4, 7, 10, 14, 20, 30, 40, 60, 80, 100, 130, 170, 200, 300, 500, 700, 1000],
                'min_child_weight': [0, 0, 0, 0, 1e-4, 1e-3, 1e-3, 1e-3, 5e-3, 2e-2, 0.1],
                'num_leaves': [20, 30, 30, 30, 50, 70, 90],
                'learning_rate': [0.02, 0.03, 0.04, 0.04, 0.05, 0.05, 0.07],
                'colsample_bytree': [0.3, 0.5, 0.7, 0.8, 0.9, 0.9, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'colsample_bynode': [0.1, 0.15, 0.2, 0.2, 0.2, 0.25, 0.3, 0.5, 0.65, 0.8, 0.9, 1],
                'reg_lambda': [0, 0, 0, 0, 1e-5, 1e-5, 1e-5, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                'reg_alpha': [0, 1e-5, 3e-5, 1e-4, 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 1, 10, 10, 100, 1000],
                'subsample': [0.9, 1],
                'subsample_freq': [1],
                'cat_smooth': [0.1, 0.2, 0.5, 1, 2, 5, 7, 10],
            }


class DataLoader:
    """Class for loading and preprocessing M5 data"""
    
    def __init__(self, data_path):
        """
        Initialize data loader
        
        Args:
            data_path (str): Path to M5 data directory
        """
        self.path = data_path
        self.train = None
        self.levels = None
        self.cal = None
        self.price_pivot = None
        self.daily_sales = None
        self.sample_sub = None
        self.level_multiplier = None
        self.day_to_cal_index = None
        self.cal_index_to_day = None
        self.cal_index_to_wm_yr_wk = None
        self.day_to_wm_yr_wk = None
        
    def load_data(self, config):
        """
        Load and preprocess all necessary data
        
        Args:
            config (M5Config): Configuration object
            
        Returns:
            tuple: Processed data components
        """
        print("Loading data...")
        start = dt.datetime.now()
        
        # Load calendar and create mappings
        self.cal = self._load_calendar()
        self.day_to_cal_index = dict([(col, idx) for idx, col in enumerate(self.cal.index)])
        self.cal_index_to_day = dict([(idx, col) for idx, col in enumerate(self.cal.index)])
        self.cal_index_to_wm_yr_wk = dict([(idx, col) for idx, col in enumerate(self.cal.wm_yr_wk)])
        self.day_to_wm_yr_wk = dict([(idx, col) for idx, col in self.cal.wm_yr_wk.items()])
        
        # Load training data
        self.train = self._load_train()
        self.price_pivot = self._get_price_pivot()
        
        # Create daily sales
        self._create_daily_sales()
        
        # Aggregate training data
        self.train, self.levels = self._aggregate_train(self.train)
        self.daily_sales = self._aggregate_train(self.daily_sales)[0]
        
        # Load sample submission
        self.sample_sub = self._load_sample_submission()
        
        # Set level multipliers
        self._set_level_multipliers()
        
        # Split levels according to configuration
        self._split_levels(config)
        
        # Rescale data by level
        self._rescale_data()
        
        # Filter data by level
        if config.max_level is not None:
            train_filter = ((self.levels <= config.max_level))
        else:
            train_filter = (self.levels == config.level)
            
        self.train = self.train[train_filter].reset_index(drop=True)
        self.daily_sales = self.daily_sales[train_filter].reset_index(drop=True)
        self.levels = self.levels[train_filter].reset_index(drop=True).astype(np.int8)
        
        # Clean data (replace leading zeros with NaN)
        self._clean_data()
        
        print(f"Data loading completed in {(dt.datetime.now() - start).seconds}s")
        return self.train, self.levels, self.cal, self.daily_sales
        
    def _load_calendar(self):
        """Load calendar data"""
        cal = pd.read_csv(f"{self.path}/calendar.csv").set_index('d')
        cal.date = pd.to_datetime(cal.date)
        return cal
        
    def _load_train(self):
        """Load training data"""
        train_cols = pd.read_csv(f"{self.path}/sales_train_evaluation.csv", nrows=1)
        c_dict = {}
        for col in [c for c in train_cols if 'd_' in c]:
            c_dict[col] = np.float32
        
        train = pd.read_csv(f"{self.path}/sales_train_evaluation.csv", dtype=c_dict)
        train.id = train.id.str.split('_').str[:-1].str.join('_')
        train.sort_values('id', inplace=True)
        
        return train.reset_index(drop=True)
        
    def _get_price_pivot(self):
        """Get price pivot table"""
        prices = pd.read_csv(f"{self.path}/sell_prices.csv",
                            dtype={'wm_yr_wk': np.int16, 'sell_price': np.float32})
        prices['id'] = prices.item_id + "_" + prices.store_id
        price_pivot = prices.pivot(columns='id', index='wm_yr_wk', values='sell_price')
        price_pivot = price_pivot.reindex(sorted(price_pivot.columns), axis=1)
        return price_pivot
        
    def _load_sample_submission(self):
        """Load sample submission file"""
        return pd.read_csv(f"{self.path}/sample_submission.csv").astype(np.int8, errors='ignore')
        
    def _create_daily_sales(self):
        """Create daily sales data from train and price data"""
        assert (self.train.id == self.price_pivot.columns).all()
        self.daily_sales = pd.concat((
            self.train.iloc[:, :6],
            self.train.iloc[:, 6:] * self.price_pivot.loc[
                self.train.columns[6:].fillna(0).map(self.day_to_wm_yr_wk)
            ].transpose().values
        ), axis='columns')
        
    def _aggregate_train(self, train):
        """Aggregate training data at different levels"""
        levels = [
            (12, ['item_id', 'store_id']),
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
            (1, [])
        ]
        
        downstream = {
            'item_id': ['dept_id', 'cat_id'],
            'dept_id': ['cat_id'],
            'store_id': ['state_id']
        }
        
        # Aggregation settings
        tcd = dict([(col, 'first') for col in train.columns[1:6]])
        tcd.update(dict([(col, 'sum') for col in train.columns[6:]]))
        
        tadds = []
        tadd_levels = [[12 for i in range(0, len(train))]]
        
        for idx, lvl in enumerate(levels[1:]):
            level = lvl[0]
            lvls = lvl[1]
            
            if len(lvls) == 0:  # group all if no list provided
                lvls = [1 for i in range(0, len(train))]
                
            tadd = train.groupby(lvls).agg(tcd)
            
            # Name it
            if len(lvls) == 2:
                tadd.index = ['_'.join(map(str, i)) for i in tadd.index.tolist()]
            elif len(lvls) == 1:
                tadd.index = tadd.index + '_X'
            else:
                tadd.index = ['Total_X']
            tadd.index.name = 'id'
            
            # Fill in categorical features
            tadd.reset_index(inplace=True)
            for col in [c for c in train.columns[1:6] if c not in lvls and not
                       any(c in z for z in [downstream[lvl] for lvl in lvls if lvl in downstream])]:
                tadd[col] = 'All'
            tadds.append(tadd)
            
            # Levels
            tadd_levels.append([level for i in range(0, len(tadd))])
            
        train = pd.concat((train, *tadds), sort=False, ignore_index=True)
        del tadds, tadd
        
        levels = pd.Series(data=[x for sub_list in tadd_levels for x in sub_list], index=train.index)
        del tadd_levels
        
        for col in train.columns[1:6]:
            train[col] = train[col].astype('category')
            
        return train, levels
    
    def _set_level_multipliers(self):
        """Set level multipliers for rescaling"""
        self.level_multiplier = dict([(c, (self.levels == c).sum() / (self.levels == 12).sum()) 
                                     for c in sorted(self.levels.unique())])
    
    def _split_levels(self, config):
        """Split levels according to configuration"""
        for row in config.level_splits:
            self.level_multiplier[row[0]] = self.level_multiplier[12]
            self.levels.loc[(self.levels == 12) & (self.train.cat_id == row[1])] = row[0]
            
    def _rescale_data(self):
        """Rescale data by level multipliers"""
        self.train = pd.concat((
            self.train.iloc[:, :6],
            self.train.iloc[:, 6:].multiply(self.levels.map(self.level_multiplier), axis='index').astype(np.float32)
        ), axis='columns')
        
        self.daily_sales = pd.concat((
            self.daily_sales.iloc[:, :6],
            self.daily_sales.iloc[:, 6:].multiply(self.levels.map(self.level_multiplier), axis='index').astype(np.float32)
        ), axis='columns')
    
    def _clean_data(self):
        """Replace leading zeros with NaN"""
        self.train['d_1'].replace(0, np.nan, inplace=True)
        
        for i in range(self.train.columns.get_loc('d_1') + 1, self.train.shape[1]):
            self.train.loc[:, self.train.columns[i]].where(
                ~((self.train.iloc[:, i] == 0) & (self.train.iloc[:, i-1].isnull())),
                np.nan, inplace=True
            )


class FeatureGenerator:
    """Class for generating features for M5 forecasting"""
    
    def __init__(self, train, levels, cal, daily_sales, config):
        """
        Initialize feature generator
        
        Args:
            train (pd.DataFrame): Training data
            levels (pd.Series): Series levels
            cal (pd.DataFrame): Calendar data
            daily_sales (pd.DataFrame): Daily sales data
            config (M5Config): Configuration object
        """
        self.train = train
        self.levels = levels
        self.cal = cal
        self.daily_sales = daily_sales
        self.config = config
        self.series_id_level = None
        self.train_flipped = None
        self.features = []
        self.series_features = None
        self.state_cal_features = []
        self.cal_features = None
        self.day_to_cal_index = dict([(col, idx) for idx, col in enumerate(cal.index)])
        self.cal_index_to_day = dict([(idx, col) for idx, col in enumerate(cal.index)])
        
    def generate_features(self):
        """
        Generate all features needed for training
        
        Returns:
            pd.DataFrame: Features dataframe
        """
        print("Generating features...")
        start = dt.datetime.now()
        
        # Flip train for time-series processing
        self.train_flipped = self.train.set_index('id', drop=True).iloc[:, 5:].transpose()
        
        # Generate basic features
        self._generate_basic_features()
        
        # Generate state calendar features
        self._generate_state_cal_features()
        
        # Generate calendar features
        self._generate_calendar_features()
        
        # Clean features
        self._clean_features()
        
        # Assemble series features
        self._assemble_series_features()
        
        print(f"Feature generation completed in {(dt.datetime.now() - start).seconds}s")
        return self.series_features
    
    def _generate_basic_features(self):
        """Generate basic time series features"""
        # Moving averages
        for window in [3, 7, 15, 30, 100]:
            if self.config.reduced_features and window < 15:
                continue
            self.features.append((
                f'qs_{window}d_ewm',
                self.train_flipped.ewm(
                    span=window,
                    min_periods=int(np.ceil(window ** 0.8))
                ).mean().astype(np.half)
            ))
        
        # Store average quantiles
        store_avg_qs = self.train_flipped[self.train_flipped.columns[self.levels >= 12]].transpose() \
            .groupby(self.train.iloc[(self.levels >= 12).values].store_id.values).mean().fillna(1)
            
        # Scale sales by store trends
        scaled_sales = self.train_flipped / (store_avg_qs.loc[self.train.store_id].transpose().values)
        
        # Percent non-zero days
        tff0ne0 = self.train_flipped.fillna(0).ne(0)
        for window in [7, 14, 28, 28*2, 28*4]:
            if self.config.reduced_features and window != 28:
                continue
            self.features.append((
                f'pct_nonzero_days_{window}d',
                tff0ne0.rolling(window).mean().astype(np.half)
            ))
        
        # Basic lag features
        if not self.config.reduced_features:
            for lag in range(1, 10+1):
                self.features.append((
                    f'qs_lag_{lag}d',
                    self.train_flipped.shift(lag).fillna(0).astype(np.half)
                ))
        
        # Means and medians
        arrs = [self.train_flipped, scaled_sales]
        labels = ['qs', 'qs_divbystore']
        
        if self.config.reduced_features:
            arrs = arrs[0:1]
        
        for idx in range(0, len(arrs)):
            arr = arrs[idx]
            label = labels[idx]
            
            for window in [7, 14, 21, 28, 28*2, 28*4]:
                if self.config.reduced_features and window != 28:
                    continue
                    
                self.features.append((
                    f'{label}_mean_{window}d',
                    arr.rolling(window).mean().astype(np.half)
                ))
                
                self.features.append((
                    f'{label}_median_{window}d',
                    arr.rolling(window).median().astype(np.half)
                ))
        
        # Standard deviation, skewness, and kurtosis
        for idx in range(0, len(arrs)):
            arr = arrs[idx]
            label = labels[idx]
            
            for window in [7, 14, 28, 28*3, 28*6]:
                if self.config.reduced_features and window != 28:
                    continue
                    
                self.features.append((
                    f'{label}_stdev_{window}d',
                    arr.rolling(window).std().astype(np.half)
                ))
                
                if window >= 10 and not self.config.reduced_features:
                    self.features.append((
                        f'{label}_skew_{window}d',
                        arr.rolling(window).skew().astype(np.half)
                    ))
                    
                    self.features.append((
                        f'{label}_kurt_{window}d',
                        arr.rolling(window).kurt().astype(np.half)
                    ))
        
        # High and low quantiles
        for idx in range(0, len(arrs)):
            arr = arrs[idx]
            label = labels[idx]
            
            for window in [14, 28, 56]:
                if self.config.reduced_features and window != 28:
                    continue
                    
                self.features.append((
                    f'{label}_qtile10_{window}d',
                    arr.rolling(window).quantile(0.1).astype(np.half)
                ))
                
                self.features.append((
                    f'{label}_qtile90_{window}d',
                    arr.rolling(window).quantile(0.9).astype(np.half)
                ))
    
    def _generate_state_cal_features(self):
        """Generate state calendar features"""
        # SNAP program features
        snap_cols = [c for c in self.cal.columns if 'snap' in c]
        
        self.state_cal_features.append((
            'snap_day',
            self.cal[snap_cols].astype(np.int8)
        ))
        
        self.state_cal_features.append((
            'snap_day_lag_1',
            self.cal[snap_cols].shift(1).fillna(0).astype(np.int8)
        ))
        
        self.state_cal_features.append((
            'snap_day_lag_2',
            self.cal[snap_cols].shift(2).fillna(0).astype(np.int8)
        ))
        
        self.state_cal_features.append((
            'nth_snap_day',
            (self.cal[snap_cols].rolling(15, min_periods=1).sum() * self.cal[snap_cols]).astype(np.int8)
        ))
        
        for window in [2, 5, 10, 30, 60]:
            self.state_cal_features.append((
                f'snap_{window}d_ewm',
                self.cal[snap_cols].ewm(span=window, adjust=False).mean().astype(np.half)
            ))
        
        # Strip columns to match state_id
        def snap_rename(x):
            return x.replace('snap_', '')
        
        for f in range(0, len(self.state_cal_features)):
            self.state_cal_features[f] = (
                self.state_cal_features[f][0],
                self.state_cal_features[f][1].rename(snap_rename, axis='columns')
            )
    
    def _generate_calendar_features(self):
        """Generate calendar features"""
        self.cal_features = pd.DataFrame()
        self.cal_features['dayofweek'] = self.cal.date.dt.dayofweek.astype(np.int8)
        self.cal_features['dayofmonth'] = self.cal.date.dt.day.astype(np.int8)
        self.cal_features['season'] = self.cal.date.dt.month.astype(np.half)
        
        # Holiday features
        for etype in [c for c in self.cal.event_type_1.dropna().unique()]:
            self.cal[etype.lower() + '_holiday'] = np.where(
                self.cal.event_type_1 == etype,
                self.cal.event_name_1,
                np.where(
                    self.cal.event_type_2 == etype,
                    self.cal.event_name_2,
                    'None'
                )
            )
            
            self.cal[etype.lower() + '_holiday'] = self.cal[etype.lower() + '_holiday'].astype('category')
    
    def _clean_df(self, df):
        early_rows = self.cal[self.cal.year == self.cal.year.min()].index.to_list()
        holiday_rows = self.cal[self.cal.month.isin([10, 11, 12, 1])].index.to_list()
        delete_rows = early_rows + holiday_rows
        
        min_day = 'd_300'
        
        if 'd' in df.columns:  # d, series stack
            df = df[df.d >= self.day_to_cal_index[min_day]]
            df = df[~df.d.isin([self.day_to_cal_index[d] for d in delete_rows])]
        else:  # pivot table
            if min_day in df.index:
                df = df.iloc[df.index.get_loc(min_day):, :]
                
            if len(delete_rows) > 0:
                df = df[~df.index.isin(delete_rows)]
                
        return df

    def _clean_features(self):
        """Clean features by removing early data and holiday months"""
        for idx, feat_row in enumerate(self.features):
            fr = feat_row[1]
            fr = self._clean_df(fr)
            
            if len(fr) < len(feat_row[1]):
                self.features[idx] = (self.features[idx][0], fr)
                
        for idx, feat_row in enumerate(self.state_cal_features):
            fr = feat_row[1]
            fr = self._clean_df(fr)
            
            if len(fr) < len(feat_row[1]):
                self.state_cal_features[idx] = (self.state_cal_features[idx][0], fr)
    
    def _assemble_series_features(self):
        """Assemble all features into a single dataframe"""
        # Create dictionaries for mapping
        series_to_series_id = dict([(col, idx) for idx, col in enumerate(self.train_flipped.columns)])
        series_id_to_series = dict([(idx, col) for idx, col in enumerate(self.train_flipped.columns)])
        series_id_level = dict([(idx, col) for idx, col in enumerate(self.levels)])
        
        # Stack features
        fstack = self.features[0][1].stack(dropna=False)
        self.series_features = pd.DataFrame({
            'd': fstack.index.get_level_values(0).map(self.day_to_cal_index).values.astype(np.int16),
            'series': fstack.index.get_level_values(1).map(series_to_series_id).values.astype(np.int16)
        })
        
        # Add each feature
        for idx, feature in enumerate(self.features):
            if feature is not None:
                self.series_features[feature[0]] = feature[1].stack(dropna=False).values
        
        # Stack state calendar features
        fstack = self.state_cal_features[0][1].stack(dropna=False)
        state_cal_series_features = pd.DataFrame({
            'd': fstack.index.get_level_values(0).map(self.day_to_cal_index).values.astype(np.int16),
            'state': fstack.index.get_level_values(1)
        })
        
        # Add each state feature
        for idx, feature in enumerate(self.state_cal_features):
            if feature is not None:
                state_cal_series_features[feature[0]] = feature[1].stack(dropna=False).values
        
        # Fill NAs
        self.series_features.fillna(-10, inplace=True)
        
        # Add categorical features
        categoricals = ['dept_id', 'cat_id', 'store_id', 'state_id']
        train_head = self.train.iloc[:, :6]
        
        for col in categoricals:
            self.series_features[col] = self.series_features.series.map(series_id_to_series).map(
                train_head.set_index('id')[col]
            )
        
        # Add weights and trailing volatility
        trailing_28d_sales = self.daily_sales.iloc[:, 6:].transpose().rolling(28, min_periods=1).sum().astype(np.float32)
        
        fstack = self.train_flipped.stack(dropna=False)
        weight_stack = pd.DataFrame({
            'd': fstack.index.get_level_values(0).map(self.day_to_cal_index).values.astype(np.int16),
            'series': fstack.index.get_level_values(1).map(series_to_series_id).values.astype(np.int16),
            'days_since_first': (~self.train_flipped.isnull()).expanding().sum().stack(dropna=False).values.astype(np.int16),
            'trailing_vol': ((self.train_flipped.diff().abs()).expanding().mean()).astype(np.float16).stack(dropna=False).values,
            'weights': np.ones(len(fstack)).astype(np.float16),  # Replace with uniform weights
        })
        
        # Set weights for new items to 0
        new_items = weight_stack.days_since_first < 30
        weight_stack.loc[new_items, 'weights'] = 0
        
        # Clean weight stack
        weight_stack = self._clean_df(weight_stack)
        
        # Merge weight stack with series features
        assert len(weight_stack) == len(self.series_features)
        assert (weight_stack.d.values == self.series_features.d).all()
        assert (weight_stack.series.values == self.series_features.series).all()
        
        self.series_features = pd.concat(
            (self.series_features, weight_stack.reset_index(drop=True).iloc[:, -2:]),
            axis=1
        )
        
        # Add y values
        fstack = self.train_flipped.stack(dropna=False)
        y_full = pd.DataFrame({
            'd': fstack.index.get_level_values(0).map(self.day_to_cal_index).values.astype(np.int16),
            'series': fstack.index.get_level_values(1).map(series_to_series_id).values.astype(np.int16),
            'y': fstack.values
        })
        
        # Store mapping dictionaries as attributes
        self.series_to_series_id = series_to_series_id
        self.series_id_to_series = series_id_to_series
        self.series_id_level = series_id_level
        self.y_full = y_full
        self.state_cal_series_features = state_cal_series_features


class ModelTrainer:
    """Class for training M5 forecasting models"""
    
    def __init__(self, series_features, y_full, cal_features, state_cal_features, config, series_id_level, cal_index_to_day, cal):
        """
        Initialize model trainer
        
        Args:
            series_features (pd.DataFrame): Features dataframe
            y_full (pd.DataFrame): Target values
            cal_features (pd.DataFrame): Calendar features
            state_cal_features (pd.DataFrame): State calendar features
            config (M5Config): Configuration object
        """
        self.series_features = series_features
        self.y_full = y_full
        self.cal_features = cal_features
        self.state_cal_features = state_cal_features
        self.config = config
        self.series_id_level = series_id_level
        self.cal_index_to_day = cal_index_to_day
        self.cal = cal
        self.day_to_cal_index = dict([(col, idx) for idx, col in enumerate(cal.index)])  # Add this line
        self.quantile_wts = self._get_quantile_weights()
        self.clf_set = {}
        
    def _get_quantile_weights(self):
        """Get weights for each quantile to optimize training resources"""
        quantile_levels = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
        quantile_wts = [0.1, 0.2, 0.6, 0.8, 1, 0.9, 0.7, 0.2, 0.1]
        return dict(zip(quantile_levels, quantile_wts))
    
    def train_models(self):
        """
        Train models for specified level
        
        Returns:
            dict: Dictionary of trained classifier sets
        """
        print(f"Training models for level {self.config.level}...")
        start = dt.datetime.now()
        
        level = self.config.level
        ss_frac, scale_range = self.config.p_dict[level]
        ss_frac = ss_frac * self.config.ss_ss
        
        print(f"Using fraction {ss_frac} and scale range {scale_range}")
        
        # Get level_os values
        level_os = self._get_level_os()
        
        # Train a single model instead of using bagging
        self.clf_set[level], loss_set = self._run_q_bags(
            n_bags=1,  # Set to 1 to train a single model
            data=self._get_subsample(
                ss_frac * level_os[level] ** self.config.ss_pwr,
                level,
                scale_range
            ),
            n_iter=int(
                (2.2 if level <= 9 else 1.66)
                * (16 - (level if level <= 12 else 12))
                * (1/4 if self.config.super_speed else (1/2 if self.config.speed else 1))
            ),
            quantiles=self.config.quantiles
        )
        
        print(f"Model training completed in {(dt.datetime.now() - start).seconds}s")
        return self.clf_set
    
    def _get_level_os(self):
        """Get level OS values for levels 13, 14, 15 - no weighting"""
        return {13: 1.0, 14: 1.0, 15: 1.0}
    
    def _get_subsample(self, frac, level, scale_range=0.1, n_repeats=1):
        """Get subsample of data for training"""
        start_time = dt.datetime.now()
        
        # Use series_id_level dictionary directly instead of looking for a 'level' column
        level_filter = self.series_features.series.map(lambda x: self.series_id_level.get(x) == level)
        
        wtg_mean = self.series_features.weights[level_filter].mean()
        
        ss = self.series_features.weights / wtg_mean * frac
        
        X = self.series_features[
            (ss > np.random.rand(len(ss))) & level_filter
        ]
        
        ss = X.weights / wtg_mean * frac
        
        print(f"{(ss > 1).sum()} series that seek oversampling")
        
        extras = []
        while ss.max() > 1:
            ss = ss - 1
            extras.append(X[ss > np.random.rand(len(ss))])
            
        if len(extras) > 0:
            X = pd.concat((X, *extras))
        else:
            X = X.copy()
            
        X['days_fwd'] = (np.random.randint(0, 28, size=len(X)) + 1).astype(np.int8)
        
        if n_repeats > 1:
            X = pd.concat([X] * n_repeats)
            
        gc.collect()
        
        X, y, groups, scalers = self._get_x_y_g(X, scale_range)
        
        print(f"\nSubsample Time: {(dt.datetime.now() - start_time).seconds}s\n")
        return X, y, groups, scalers
    
    def _get_x_y_g(self, X, scale_range=None, oos=False):
        """Get X, y and groups for model training"""
        start_time = dt.datetime.now()
        
        # Ensure it's in the train set, and days_forward is actually *forward*
        X.drop(X.index[(X.days_fwd < 1) |
              (~oos & (X.d + X.days_fwd > max(self.y_full.d.values)))], inplace=True)
        gc.collect()
        
        # Add feature interactions
        X = self._add_ma_crosses(X)
        
        # Add calendar features
        X = self._add_cal_features(X)
        X = self._add_state_cal_features(X)
        
        # Add noise to time-static features
        for col in [c for c in X.columns if 'store' in c and 'ratio' in c]:
            X[col] = X[col] + np.random.normal(0, 0.1, len(X))
            
        # Match with Y
        if 'y' not in X.columns:
            X['future_d'] = X.d + X.days_fwd
            if oos:
                X = X.merge(self.y_full.rename(columns={'d': 'future_d'}), 
                           on=['future_d', 'series'], how='left')
                X.y = X.y.fillna(-1)
            else:
                X = X.merge(self.y_full.rename(columns={'d': 'future_d'}),
                           on=['future_d', 'series'])
        
        gc.collect()
        
        # Extract scaler columns - only using trailing_vol, not weights
        scalers = X[['trailing_vol']].copy()
        y = X.y
        
        # Define groups for CV
        groups = pd.Series(self.cal.iloc[(X.d + X.days_fwd).values].year.values, X.index).astype(np.int16)
        
        # Determine features to drop
        if self.config.reduced_features:
            feat_drops = [c for c in X.columns if c not in (self.config.sparse_features + ['d', 'series', 'days_fwd'])]
        elif len(self.config.feature_drops) > 0:
            feat_drops = [c for c in X.columns if any(z in c for z in self.config.feature_drops)]
        else:
            feat_drops = []
            
        # Final drops
        X.drop(columns=(
            ['trailing_vol'] + 
            (['future_d'] if 'future_d' in X.columns else []) + 
            ['y'] + 
            feat_drops
        ), inplace=True)
        
        scalers['scaler'] = scalers.trailing_vol.copy()
        
        # Randomize scaling
        if scale_range > 0:
            scalers.scaler = scalers.scaler * np.exp(scale_range * (np.random.normal(0, 0.5, len(X))))
            
        # Rescale y and scaled variables in X by its vol
        for col in [c for c in X.columns if 'qs_' in c and 'ratio' not in c]:
            X[col] = np.where(X[col] == -10, X[col], (X[col] / scalers.scaler).astype(np.half))
        
        y = y / scalers.scaler
        
        # Remove null or validation rows
        validation = -1
        yn = (oos == False) & (y.isnull() | (groups == validation))
        
        print(f"\nXYG Pull Time: {(dt.datetime.now() - start_time).seconds}s")
        
        return X[~yn].reset_index(drop=True), y[~yn].reset_index(drop=True), groups[~yn].reset_index(drop=True), scalers[~yn].reset_index(drop=True)
    
    def _add_ma_crosses(self, X):
        """Add moving average crosses features"""
        ewms = [c for c in X.columns if 'ewm' in c and 'qs_' in c and len(c) < 12]
        for idx1, col1 in enumerate(ewms):
            for idx2, col2 in enumerate(ewms):
                if not idx1 < idx2:
                    continue
                    
                X[f'qs_{col1.split("_")[1]}_{col2.split("_")[1]}_ewm_diff'] = X[col1] - X[col2]
                X[f'qs_{col1.split("_")[1]}_{col2.split("_")[1]}_ewm_ratio'] = X[col1] / X[col2]
                
        return X
    
    def _add_cal_features(self, X):
        """Add calendar features"""
        X['dayofweek'] = (X.d + X.days_fwd).map(self.cal_index_to_day).map(self.cal_features.dayofweek)
        X['dayofmonth'] = (X.d + X.days_fwd).map(self.cal_index_to_day).map(self.cal_features.dayofmonth)
        
        X['basedayofweek'] = X.d.map(self.cal_index_to_day).map(self.cal_features.dayofweek)
        X['dayofweekchg'] = (X.days_fwd % 7).astype(np.int8)
        
        X['basedayofmonth'] = X.d.map(self.cal_index_to_day).map(self.cal_features.dayofmonth)
        X['season'] = ((X.d + X.days_fwd).map(self.cal_index_to_day).map(self.cal_features.season) 
                     + np.random.normal(0, 1, len(X))).astype(np.half)
        
        # Add holiday features
        holiday_cols = [c for c in self.cal.columns if '_holiday' in c]
        for col in holiday_cols:
            X['base_' + col] = X.d.map(self.cal_index_to_day).map(self.cal[col])
            X[col] = (X.d + X.days_fwd).map(self.cal_index_to_day).map(self.cal[col])
            
        return X
    
    def _add_state_cal_features(self, X):
        """Add state calendar features"""
        if (X.state_id == 'All').mean() > 0:
            print('No State Ids')
            return X
            
        def rename_scf(c, name='basedate'):
            return c if (c == 'd' or c == 'state') else name + '_' + c
            
        X['future_d'] = X.d + X.days_fwd
        X['state'] = X.state_id.astype('object')
        
        # Use the state_cal_features DataFrame directly as in the original
        nX = X.merge(
            self.state_cal_features[['state', 'd', 'snap_day', 'nth_snap_day']]
            .rename(rename_scf, axis='columns'),
            on=['d', 'state'],
            validate='m:1', how='inner', suffixes=(False, False)
        )
        
        nX = nX.merge(
            self.state_cal_features[['state', 'd', 'snap_day', 'nth_snap_day']]
            .rename(columns={'d': 'future_d'}),
            on=['future_d', 'state'],
            validate='m:1', how='inner', suffixes=(False, False)
        )
            
        nX.drop(columns=['state', 'future_d'], inplace=True)
        
        assert len(nX) == len(X)
        return nX
    
    def _run_q_bags(self, n_bags=1, data=None, quantiles=None, **kwargs):
        """Run multiple quantile bags"""
        start_time = dt.datetime.now()
        
        clf_set = []
        loss_set = []
        
        # Single bag since we're not using bagging anymore
        print('\n\n  Running single model (no bagging)\n\n')
        
        if data is None:
            X, y, groups, scalers = self._get_subsample()
        else:
            X, y, groups, scalers = data
            
        group_list = [*dict.fromkeys(groups)]
        group_list.sort()
        print(f"Groups: {group_list}")
        
        clfs = []
        preds = []
        ys = []
        datestack = []
        losses = pd.DataFrame(index=quantiles)
        
        if self.config.single_fold:
            group_list = group_list[-1:]
            
        for group in group_list:
            print(f'\n\n   Running Models with {group} Out-of-Fold\n\n')
            x_holdout = X[groups == group]
            y_holdout = y[groups == group]
            
            model = self._train_lgb_quantile
            
            q_clfs = []
            q_losses = []
            
            for quantile in quantiles:
                set_filter = (
                    (groups != group)
                    & (np.random.rand(len(groups)) < self.quantile_wts[quantile] ** (0.35 if self.config.level >= 11 else 0.25))
                )
                
                clf = model(
                    X[set_filter],
                    y[set_filter],
                    groups[set_filter],
                    alpha=quantile,
                    **kwargs
                )
                
                q_clfs.append(clf)
                
                predicted = clf.predict(x_holdout)
                
                q_losses.append((quantile, self._quantile_loss(y_holdout, predicted, quantile)))
                print(f"{group} μ={quantile:.3f}: {q_losses[-1][1]:.4f}")
                
                preds.append(predicted)
                ys.append(y_holdout)
                
            clfs.append(q_clfs)
            print(f"\nLevel {self.config.level} OOS Losses in {group}:")
            print(np.round(pd.DataFrame(q_losses).set_index(0)[1], 4))
            losses[group] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values
            print(f"\nElapsed Time: {(dt.datetime.now() - start_time).seconds}s\n")
            
        clf_set.append(clfs)
        print(f"\nLevel {self.config.level} Year-by-Year OOS Losses:")
        print(losses)
        
        loss_set.append(losses)
        print(f"\nModel Training Time: {(dt.datetime.now() - start_time).seconds}s\n")
            
        return clf_set, loss_set
    
    def _quantile_loss(self, true, pred, quantile=0.5):
        """Compute quantile loss"""
        loss = np.where(
            true >= pred,
            quantile * (true - pred),
            (1 - quantile) * (pred - true)
        )
        return np.mean(loss)
    
    def _train_lgb_quantile(self, x, y, groups, cv=0, n_jobs=None, alpha=0.5, **kwargs):
        """Train LightGBM quantile regression model"""
        clfargs = kwargs.copy()
        clfargs.pop('n_iter', None)
        
        clf = lgb.LGBMRegressor(
            verbosity=-1,
            hist_pool_size=1000,
            objective='quantile',
            alpha=alpha,
            importance_type='gain',
            seed=dt.datetime.now().microsecond if self.config.time_seed else None,
            **clfargs
        )
        
        print(f'\n\n Running Quantile Regression for μ={alpha}\n')
        
        return self._train_model(x, y, groups, clf, self.config.lgb_params, self._quantile_scorer(alpha), n_jobs, **kwargs)
    
    def _quantile_scorer(self, quantile=0.5):
        """Create a scorer function for quantile regression"""
        return make_scorer(self._quantile_loss, False, quantile=quantile)
    
    def _train_model(self, x, y, groups, clf, params, cv=0, n_jobs=None, verbose=0, splits=None, **kwargs):
        """Train a model using RandomizedSearchCV"""
        if n_jobs is None:
            n_jobs = -1
            
        folds = LeaveOneGroupOut()
        clf = RandomizedSearchCV(
            clf,
            params,
            cv=folds,
            n_iter=(kwargs['n_iter'] if len(kwargs) > 0 and 'n_iter' in kwargs else 4),
            verbose=0,
            n_jobs=n_jobs,
            scoring=cv
        )
        
        f = clf.fit(x, y, groups)
        print(pd.DataFrame(clf.cv_results_['mean_test_score']))
        print()
        
        best = clf.best_estimator_
        print(best)
        print(f"\nBest In-Sample CV: {np.round(clf.best_score_, 4)}\n")
        
        return best
        
    def save_models(self, clf_set, filename=None):
        """Save trained models to file"""
        if filename is None:
            filename = f'lvl_{self.config.level}_clfs.pkl'
            
        with open(filename, 'wb') as handle:
            pickle.dump(clf_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Models saved to {filename}")


class M5Forecaster:
    """Main class for M5 forecasting"""
    
    def __init__(self, data_path="m5-data", level=13, super_speed=True):
        """
        Initialize M5 forecaster
        
        Args:
            data_path (str): Path to M5 data directory
            level (int): Which level to process (13=HOBBIES, 14=HOUSEHOLD, 15=FOODS)
            super_speed (bool): Whether to use super speed mode
        """
        self.data_path = data_path
        self.config = M5Config(level, super_speed)
        self.data_loader = None
        self.feature_generator = None
        self.model_trainer = None
    
    def run(self):
        """Run the full forecasting pipeline"""
        start = dt.datetime.now()
        print(f"Starting M5 forecaster for Level {self.config.level} with SUPER_SPEED={self.config.super_speed}")
        
        # Initialize components
        self.data_loader = DataLoader(self.data_path)
        
        # Load and preprocess data
        train, levels, cal, daily_sales = self.data_loader.load_data(self.config)
        
        # Generate features
        self.feature_generator = FeatureGenerator(train, levels, cal, daily_sales, self.config)
        series_features = self.feature_generator.generate_features()
        
        # Train models - a single model per quantile will be trained (no bagging)
        self.model_trainer = ModelTrainer(
            series_features,
            self.feature_generator.y_full,
            self.feature_generator.cal_features,
            self.feature_generator.state_cal_series_features,
            self.config,
            self.feature_generator.series_id_level,
            # Pass these additional mapping dictionaries
            self.feature_generator.cal_index_to_day, 
            self.feature_generator.cal
        )
        
        # Override config bags to ensure a single model is trained
        self.config.bags = 1
        
        clf_set = self.model_trainer.train_models()
        
        # Save models
        self.model_trainer.save_models(clf_set)
        
        print(f"Total runtime: {(dt.datetime.now() - start).seconds}s")
        return clf_set


def main():
    """Main function to run forecaster"""
    # For training models at levels 13, 14, 15 with SUPER_SPEED
    for level in [13, 14, 15]:
        print(f"\n\n{'='*80}\nTraining model for level {level}\n{'='*80}\n")
        forecaster = M5Forecaster(data_path="m5-data", level=level, super_speed=True)
        forecaster.run()


if __name__ == "__main__":
    main()
