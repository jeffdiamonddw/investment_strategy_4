import functools
import joblib
import multiprocessing as mp
import numpy as np
import time
from pymoo.core.problem import Problem

import os
import numpy as np
import pandas as pd
import xarray as xr
import boto3
import time
import pickle
import fsspec
from pymoo.core.problem import Problem
import multiprocessing as mp

from triple_threat import *
from pareto_navigator import mean_annual_drawdown_integral, annualized_return, pct_change_quantile, worst_annual_return

# Reuse your existing classes and utilities from triple_threat.py
# (Assuming TripleThreatProblem and RobustParallelManager are available)

# --- CONFIGURATION FOR PERTURBATION ---
TICKER_FILE = "strategy/multi_dim_stock_list.csv"
PARETO_INPUT_FILE = 'analysis/star_17_year.csv' # The file you extracted
NUM_PERTURBATIONS = 9000
NOISE_SCALE = 0  # 5% COEFF VAR
NUM_WORKERS = 1
BATCH_REQUIREMENT = 1
TIMEOUT = 180
EVAL_FOLDER = "s3://jdinvestment/perturbation_evaluations_17_year"
HOLDINGS_FOLDER = "s3://jdinvestment/perturbation_holdings_17_year"
PERTURBATION_FOLDER = "s3://jdinvestment/perturbations_17_year"


# Indices: 0-7: PCA, 8: Threshold, 9: Beta, 10-11: Decay, 12-15: Macro Weights
XL_DEFAULT = np.array([
    -2, -2, -2, -2,  # Mom PCA
    -2, -2, -2, -2,  # Qual PCA
    -2.0,            # Threshold (Index 8: expanded from 0.1)
    0.5,             # Beta (Index 9)
    -1, -1,          # Decays
    -1, -1, -1, -1   # Macro Weights
])

XU_DEFAULT = np.array([
    2, 2, 2, 2,      # Mom PCA
    2, 2, 2, 2,      # Qual PCA
    2.0,             # Threshold (Index 8: expanded from 0.9)
    15.0,            # Beta (Index 9: expanded from 2.0)
    1, 1,            # Decays
    1, 1, 1, 1       # Macro Weights
])
VAR_COLS = [

            'dollar_ret_1p',
            'dollar_ret_6p',
            'dollar_ret_13p',
            'dollar_ret_26p',
            'avg_eps_1q',
            'avg_eps_2q',
            'avg_eps_4q',
            'avg_eps_8q', 
            'threshold', 
            'beta',
            'mom_decay',
            'qual_decay',
            'macro_weights_0',
            'macro_weights_1',
            'macro_weights_2',
            'macro_weights_3'
]




import logging


# Setup basic logging to stdout (which AWS Batch sends to CloudWatch)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TripleThreatMonitor")




if __name__ == "__main__":
    periods = {
        'test': {'train_start_date': pd.to_datetime('2022-12-01'), 'val_start_date': pd.to_datetime('2024-12-30'), 'end_date': pd.to_datetime('2026-05-01')},
    }
    periods = {
        'test': {'train_start_date': pd.to_datetime('Jan 1, 2006'), 'val_start_date': pd.to_datetime('Jan 1, 2008'), 'end_date': pd.to_datetime('May 1, 2026')},
    }

    weight_cols = [
        'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p',
       'dollar_ret_26p', 'avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q',
       'avg_eps_8q', 'threshold', 'beta', 'mom_decay', 'qual_decay',
       'macro_weights_0', 'macro_weights_1', 'macro_weights_2',
       'macro_weights_3'
    ]
    
    
    tickers = list(pd.read_csv(TICKER_FILE).symbol) + ['GIC']
    objective_functions_dict = {
        'worst_annual_return': functools.partial(worst_annual_return, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'drawdown_integral': functools.partial(mean_annual_drawdown_integral, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'annualized_return': functools.partial(annualized_return, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'worst_annual_4wk': functools.partial(pct_change_quantile, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers, 1/13),
    }
    objective_sense = {'drawdown_integral': 'min', 'annualized_return': 'max', 'worst_annual_4wk': 'max'}

    principal = [408000]
    #principal = [8652671.078100001]
    
    params = {
            'principal': principal, 'max_frac': .05, 'feature_horizon_weeks': 104,
            'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
            'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
        }
    # params = {
    #         'principal': [327000, 60000, 21000], 'max_frac': .05, 'feature_horizon_weeks': 104,
    #         'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
    #         'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
    #     }

    holdings = pd.read_csv('temp/holdings_2024_12_30.csv').transpose().iloc[1:-2]
    holdings = None

    problem_args = get_triple_threat_params(
        periods,
        weight_cols, 
        objective_functions_dict, 
        objective_sense,
        momentum_file = "simulation_data/momentum.nc", 
        quality_file = "simulation_data/quality.nc",
        gic_file = "simulation_data/gic_data.nc",
        macro_file = "simulation_data/macro_signals.csv",
        manifold_file = "sim_results/manifold_triple_threat.csv",
        holdings_folder = HOLDINGS_FOLDER,
        eval_folder = EVAL_FOLDER,
        params = params,
        holdings = holdings


    )
    with open('temp/problem_args1.joblib', 'wb') as fp:
        joblib.dump(problem_args, fp )

    problem = TripleThreatProblem(*problem_args)
    xl, xu = problem_args[-2:]

    df_pareto = pd.read_csv(PARETO_INPUT_FILE)
   
    
    # Generate 10,000 samples around Pareto seeds
    X_all = df_pareto[weight_cols].values
    
    problem._evaluate(X_all[0], dict())
    