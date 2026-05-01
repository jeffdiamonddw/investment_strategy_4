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

from manifold_dry_run_parallel import *
from pareto_navigator import *
from run_perturbations import *

# Reuse your existing classes and utilities from manifold_dry_run_parallel.py
# (Assuming TripleThreatProblem and RobustParallelManager are available)

# --- CONFIGURATION FOR PERTURBATION ---

NUM_PERTURBATIONS = 5000
NOISE_SCALE = 0.01  # 5% COEFF VAR
NUM_WORKERS = 94
TARGET_COMPLETIONS = 90
TIMEOUT = 180
EVAL_FOLDER = "s3://jdinvestment/perturbation_evaluations_nav_1"
HOLDINGS_FOLDER = "s3://jdinvestment/perturbation_holdings_nav_1"
PERTURBATION_FOLDER = "s3://jdinvestment/perturbations_nav_1"

PARETO_INPUT_FILE = "analysis/stars.csv"
STAR_FILE = 'temp/all_pareto_nav_stars.csv'
MOMENTUM_FILE = "simulation_data/momentum.nc"
QUALITY_FILE = "simulation_data/quality.nc"
GIC_FILE = "simulation_data/gic_data.nc"
MACRO_FILE = "simulation_data/macro_signals.csv"
MANIFOLD_FILE = "sim_results/manifold_triple_threat.csv"
CHECKPOINT_URI= "s3://jdinvestment/checkpoints/perturbations_nav_checkpoint_1"
VAR_COLS = ["p_{}".format(i) for i in range(17)]










if __name__ == "__main__":
    print("--- START ---")

    da_mom = xr.open_dataset(MOMENTUM_FILE).to_array().squeeze()
    da_qual = xr.open_dataset(QUALITY_FILE).to_array().squeeze()
    _data_features = xr.concat([da_mom, da_qual], dim='band')
    da_mom_gic = xr.open_dataarray(GIC_FILE)
    da_qual_gic = get_gic_eps(da_mom_gic)
    data_features = xr.concat([_data_features, xr.concat([da_mom_gic, da_qual_gic], dim='band')], dim='symbol').drop_sel(band = 'price_end')
    
    df_price = da_mom.sel(band='price_end').to_pandas()
    
    df_macro = pd.read_csv(MACRO_FILE, index_col=0)
    df_macro.index = pd.to_datetime(df_macro.index)
    
    mapping = {'VIX_RATIO_SMOOTH': 'vix_z', 'YIELD_SPREAD_SMOOTH': 'yield_spread_z', 'HY_SPREAD_SMOOTH': 'hy_spread_z', 'FED_RATE_SMOOTH': 'fed_z'}
    for csv_col, engine_key in mapping.items():
        if csv_col in df_macro.columns:
            rolling_mean = df_macro[csv_col].rolling(window=252, min_periods=1).mean()
            rolling_std = df_macro[csv_col].rolling(window=252, min_periods=1).std()
            df_macro[engine_key] = (df_macro[csv_col] - rolling_mean) / rolling_std.replace(0, 1)

    
    
    dates = sorted(list(set(df_macro.index).intersection(df_price.columns)))
    df_macro = df_macro.loc[dates, [col for col in df_macro.columns if col.endswith('z')]]
    
    params = {
        'principal': [327000, 60000, 21000], 'max_frac': .05, 'feature_horizon_weeks': 104,
        'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
        'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
    }
    
    training_periods = {
        'boom': {'train_start_date': pd.to_datetime('Jan 1, 2018'), 'end_date': pd.to_datetime('Jan 1, 2025')},
        'crash': {'train_start_date': pd.to_datetime('Nov 1, 2005'), 'end_date': pd.to_datetime('Nov 1, 2012')}
    }

    print("Initializing problem kits...")
    df_man = pd.read_csv(MANIFOLD_FILE)
    mom_cols = ['dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p']
    qual_cols = ['avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q']
    
    df_elite = df_man.nlargest(int(len(df_man) * 0.10), 'f4_terminal')
    mom_kit = build_kit(df_elite, mom_cols)
    qual_kit = build_kit(df_elite, qual_cols)

    tickers = data_features.symbol.values
    objective_functions_dict = {
        'f1_2008': functools.partial(drawdown_integral, pd.to_datetime('2007-11-26'), pd.to_datetime('2012-10-22'), tickers),
        'f2_2020': functools.partial(drawdown_integral, pd.to_datetime('2020-01-01'), pd.to_datetime('2020-12-31'), tickers),
        'f1_2022': functools.partial(drawdown_integral, pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'), tickers),
        'f4_terminal': functools.partial(terminal_value, pd.to_datetime('2020-01-01'), pd.to_datetime('2024-12-31'), tickers),
        'f5_worst_annual': functools.partial(pct_change_quantile, pd.to_datetime('2020-01-01'), pd.to_datetime('2024-12-31'), tickers, 1/13),
        'crash': functools.partial(annualized_return, pd.to_datetime('2007-11-26'), pd.to_datetime('2012-10-22'), tickers),
        'boom': functools.partial(annualized_return, pd.to_datetime('2020-01-01'), pd.to_datetime('2024-12-31'), tickers),
    }
    objective_sense = {'boom': 'max', 'crash': 'max'}
    weight_columns = mom_cols + qual_cols + [
            'threshold', 
            'beta',
            'mom_decay',
            'qual_decay',
            'macro_weights_0',
            'macro_weights_1',
            'macro_weights_2',
            'macro_weights_3'
    ]
    df_stars = pd.read_csv(PARETO_INPUT_FILE)


    #order stars along path in objective space
    #stars_path = get_greedy_path_indices(df_stars, list(objective_sense.keys()))
    #df_stars = df_stars.iloc[stars_path].reset_index()

    # 17 Pymoo Parameters (12 Temporal Encoder + 5 Spatial/Logic)
    xl = np.array([0.01]*4 + [-2.0]*4 + [-1.0]*4 + [0.001, 0.1, -2.0, -1.0, 0.0])
    xu = np.array([0.95]*4 + [2.0]*4 + [1.0]*4 + [1.5, 5.0, 2.0, 1.0, 0.5])


    # Pack the re-injection data for the problem
    # Note: Added HOLDINGS_FILE here to match your current __init__
    problem_args = (df_stars, weight_columns, df_macro, 
                 mom_kit, qual_kit, data_features, df_price, params, 
                 training_periods, HOLDINGS_FOLDER, EVAL_FOLDER, 
                 objective_functions_dict, objective_sense, xl, xu
    )

    var_count = 17
    obj_count = len(objective_sense)

    df_star = pd.read_csv(STAR_FILE)

    # Generate 10,000 samples around Pareto seeds
    X_all, df_perturb = generate_perturbations(df_star,   NUM_PERTURBATIONS, xl, xu, VAR_COLS, NOISE_SCALE)
    df_perturb.to_csv("{}/perturbations.csv".format(PERTURBATION_FOLDER))

    #*********************************************************************************************************************************
    manager = HighThroughputBatchManager(num_workers = NUM_WORKERS, timeout_sec =TIMEOUT, workhorse_cls = ParetoNavigator, workhorse_args = problem_args, target_completions = TARGET_COMPLETIONS)
    
    manager.run_batch(X_all)

    manager.cleanup()