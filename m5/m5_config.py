"""
Shared configuration parameters for M5 LGBM scripts.
This module contains constants used by both m5_lgbm_winner.py and m5_lgbm_evaluate.py.
"""

# Common paths and base settings
FINAL_BASE = 'd_1941'
path = 'data/m5-forecasting-uncertainty'

# Runtime flags
SINGLE_FOLD = True
SPEED = True
SUPER_SPEED = False
REDUCED_FEATURES = False
TIME_SEED = True

# Model parameters
BAGS = 1
N_JOBS = -1
SS_PWR = 0.6
BAGS_PWR = 0
SS_SS = 0.8  # 0.8 was production version
RSEED = 11
N_REPEATS = 20
VALIDATION = -1

# Features and metrics
sparse_features = [
    'dayofweek', 'dayofmonth',
    'qs_30d_ewm', 'qs_100d_ewm',
    'qs_median_28d', 'qs_mean_28d',
    'state_id',
    'qs_qtile90_28d',
    'pct_nonzero_days_28d',
    'days_fwd'
]

FEATURE_DROPS = [
    'item_id', '_abs_diff', 'squared_diff',
    '336', '300d'
]

# Level definitions
LEVEL_SPLITS = [(13, 'HOBBIES'), (14, 'HOUSEHOLD'), (15, 'FOODS')]
QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

P_DICT = {
    1: (0.3, 0.7),   2: (0.1, 0.7),  3: (0.1, 0.5), 
    4: (0.3, 0.5),   5: (0.15, 1),   6: (0.2, 0.5),
    7: (0.1, 1),     8: (0.2, 0.5),  9: (0.1, 0.5),
   10: (0.05, 0.5), 11: (0.04, 1),  
   13: (0.12, 2),   14: (0.065, 2), 15: (0.03, 0.5)
}

LEVELS = [
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

DOWNSTREAM = {
    'item_id': ['dept_id', 'cat_id'],
    'dept_id': ['cat_id'],
    'store_id': ['state_id']
}

MEM_CAPACITY = 3e6  
MAX_RUNS = 2500 * (1/10 if SPEED or SUPER_SPEED else 1)
MIN_RUNS = 20 * (1/20 if SPEED or SUPER_SPEED else 1)

# Adjust SS_SS based on speed settings
if SPEED or SUPER_SPEED or REDUCED_FEATURES:
    SS_SS /= (5 if SUPER_SPEED else (2 if SPEED else 1)) * (5 if REDUCED_FEATURES else 1)