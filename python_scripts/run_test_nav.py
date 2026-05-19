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

NUM_PERTURBATIONS = 100
NOISE_SCALE = 0.01  # 5% COEFF VAR
NUM_WORKERS = 1
TARGET_COMPLETIONS = 1
TIMEOUT = 180
EVAL_FOLDER = "s3://jdinvestment/perturbation_evaluations_nav_median"
HOLDINGS_FOLDER = "s3://jdinvestment/perturbation_holdings_nav_median"
PERTURBATION_FOLDER = "s3://jdinvestment/perturbations_nav_median"

PARETO_INPUT_FILE = "analysis/stars.csv"
STAR_FILE = 'analysis/best_pareto_nav_solution.csv'
MOMENTUM_FILE = "simulation_data/momentum.nc"
QUALITY_FILE = "simulation_data/quality.nc"
GIC_FILE = "simulation_data/gic_data.nc"
MACRO_FILE = "simulation_data/macro_signals.csv"
MANIFOLD_FILE = "sim_results/manifold_triple_threat.csv"
CHECKPOINT_URI= "s3://jdinvestment/checkpoints/perturbations_nav_median"
VAR_COLS = ["p_{}".format(i) for i in range(17)]










if __name__ == "__main__":
    print("--- START ---")

    periods = {
        '2013': {'train_start_date': pd.to_datetime('Jan 1, 2011'), 'end_date': pd.to_datetime('Jan 1, 2020')},
        '2025': {'train_start_date': pd.to_datetime('Jan 1, 2023'), 'end_date': pd.to_datetime('April 20, 2026')}
    }
    tickers = list(np.array(xr.open_dataarray(MOMENTUM_FILE).symbol)) + ['GIC']
    objective_functions_dict = {
        '2013_drawdown': functools.partial(drawdown_integral, pd.to_datetime('Jan 1, 2013'), pd.to_datetime('Jan 1, 2020'), tickers),
        '2025_drawdown': functools.partial(drawdown_integral, pd.to_datetime('Jan 1, 2025'), pd.to_datetime('April 20, 2026'), tickers),
        '2013_worst_annual': functools.partial(pct_change_quantile, pd.to_datetime('Jan 1, 2013'), pd.to_datetime('Jan 1, 2020'), tickers, 1/13),
        '2025_worst_annual': functools.partial(pct_change_quantile, pd.to_datetime('Jan 1, 2025'), pd.to_datetime('April 20, 2026'), tickers, 1/13),
        '2013_annualized': functools.partial(annualized_return, pd.to_datetime('Jan 1, 2013'), pd.to_datetime('Jan 1, 2020'), tickers),
        '2025_annuialized': functools.partial(annualized_return, pd.to_datetime('Jan 1, 2025'), pd.to_datetime('April 20, 2026'), tickers),
    }
    problem_args = get_pareto_nav_params(
        periods, 
        objective_functions_dict,
        star_file = PARETO_INPUT_FILE,
        holdings_folder = HOLDINGS_FOLDER,
        eval_folder = EVAL_FOLDER
    )
    xl, xu = problem_args[-2:]
    
    df_star = pd.read_csv(STAR_FILE)

    # Generate 10,000 samples around Pareto seeds
    X_all, df_perturb = generate_perturbations(df_star,   NUM_PERTURBATIONS, xl, xu, VAR_COLS, NOISE_SCALE)
    df_perturb.to_csv("{}/perturbations.csv".format(PERTURBATION_FOLDER))

    #*********************************************************************************************************************************
    manager = HighThroughputBatchManager(num_workers = NUM_WORKERS, timeout_sec =TIMEOUT, workhorse_cls = ParetoNavigator, workhorse_args = problem_args, target_completions = TARGET_COMPLETIONS)
    
    manager.run_batch(X_all)

    manager.cleanup()