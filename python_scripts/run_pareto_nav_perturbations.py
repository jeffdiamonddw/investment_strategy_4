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

NUM_PERTURBATIONS = 1000
NOISE_SCALE = 0.01  # 5% COEFF VAR
NUM_WORKERS = 94
TARGET_COMPLETIONS = 90
TIMEOUT = 180
EVAL_FOLDER = "s3://jdinvestment/perturbation_evaluations_nav_median"
HOLDINGS_FOLDER = "s3://jdinvestment/perturbation_holdings_nav_median"
PERTURBATION_FOLDER = "s3://jdinvestment/perturbations_nav_median"

PARETO_INPUT_FILE = "analysis/stars.csv"
STAR_FILE = 'temp/median_nav_pareto_0.csv'
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
        'boom': {'train_start_date': pd.to_datetime('Jan 1, 2018'), 'end_date': pd.to_datetime('Jan 1, 2025')},
        'crash': {'train_start_date': pd.to_datetime('Nov 1, 2005'), 'end_date': pd.to_datetime('Nov 1, 2012')}
    }
    problem_args = get_pareto_nav_params(
        periods, 
        star_file = STAR_FILE,
        holdings_folder = HOLDINGS_FOLDER,
        eval_folder = EVAL_FOLDER
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