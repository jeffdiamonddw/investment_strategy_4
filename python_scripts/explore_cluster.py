import functools
import itertools
import joblib
import logging
import math
import os
import subprocess
import sys
import time
from datetime import date
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core import *
import scipy.interpolate
import xarray as xr
from scipy.interpolate import interp1d

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

from pyomo.opt import TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds

import awswrangler as wr

# --- Utility Functions (Preserved from original) ---

def dataframe_to_dict(R):
    return {(symbol, date) : R.loc[symbol, date] for symbol, date in itertools.product(R.index, R.columns)}

def var_to_dataframe(var):
    records = []
    for index in var:
        val = value(var[index])
        idx_tuple = index if isinstance(index, tuple) else (index,)
        records.append((*idx_tuple, val))
    col_names = [f"dim_{i+1}" for i in range(len(records[0]) - 1)] + ["value"]
    return pd.DataFrame(records, columns=col_names)

def var_to_pivot_table(var, index_name = 'symbol', columns_name = 'account'):
    df = var_to_dataframe(var).rename(columns = {'dim_1': index_name, 'dim_2': columns_name})
    return pd.pivot_table(df, index = index_name, columns = columns_name, values = 'value')

class SuppressOutput:
    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = open(os.devnull, 'w'), open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close(); sys.stderr.close()
        sys.stdout, sys.stderr = self._stdout, self._stderr

# --- Core Logic Functions (Preserved from original) ---

def optimize(_params , df_features, current_price, holdings, budget, feature_weights):
    
    logging.getLogger('pyomo.util.infeasible').setLevel(logging.INFO)
    budget = np.maximum(0, budget)
    
  
    params = _params.copy()
    params['feature_values'] = df_features.fillna(0)
    params['current_price'] = current_price
    params['holdings'] = holdings
    params['budget'] = budget
    keep_stocks = (params['current_price'].values > params['min_price']).flatten() | (params['current_price'].index == 'GIC').flatten()
    drop_stock_list = params['current_price'].index[~keep_stocks]
    for key in ['current_price', 'feature_values', 'holdings']:
        params[key] = params[key].loc[keep_stocks]

    model = pyo.ConcreteModel()
    model.stock = pyo.Set(initialize = list(params['current_price'].index))
    model.feature = pyo.Set(initialize = list(params['feature_values'].columns))
    model.account = pyo.Set(initialize = range(len(params['principal'])))
    model.feature_values = pyo.Param(model.stock, model.feature, initialize = dataframe_to_dict(params['feature_values']), domain = pyo.Reals)
    model.current_price = pyo.Param(model.stock, initialize = dict(pd.DataFrame(params['current_price']).iloc[:,0]), domain = pyo.NonNegativeReals)
    model.M = pyo.Param(initialize = 1e7)
    model.holdings = pyo.Param(model.stock, model.account, initialize = dataframe_to_dict(params['holdings']))
    model.budget = pyo.Param(model.account, initialize = params['budget'])
    model.feature_weight = pyo.Param(model.feature, initialize = feature_weights)

    model.x = pyo.Var(model.stock, model.account, domain=pyo.NonNegativeIntegers)
    model.t = pyo.Var(model.stock, model.account, domain=pyo.Binary)

    def budget_constraint(model, a):
        return sum(model.x[s,a] * model.current_price[s] for s in model.stock) + sum(params['trade_fee'] * model.t[s,a] for s in model.stock) <= model.budget[a]
    model.budget_constraint = pyo.Constraint(model.account, rule = budget_constraint)

    def max_frac_constraint(model, stock):
       if stock == 'GIC': return Constraint.Feasible
       return sum(model.x[stock, a] * model.current_price[stock] for a in model.account) <= params['max_frac'] * sum(params['budget'])
    model.max_frac_constraint = pyo.Constraint(model.stock, rule = max_frac_constraint)

    def obj_expression(model):
        return - sum(model.feature_weight[w] * model.feature_values[s, w] * model.x[s,a] for w in model.feature for s in model.stock for a in model.account)
    model.OBJ = pyo.Objective(rule=obj_expression, sense = pyo.minimize)

    solver = pyo.SolverFactory('cbc')
    solver.options["seconds"] = 1
    solver.options["threads"] = 1  # Force CBC to use exactly one core
    solver.options["ratioGap"] = 1 - params['objective_sensitivity']

    with SuppressOutput():
        result = solver.solve(model)


    # if result.solver.termination_condition == TerminationCondition.infeasible:
    #     print("\n" + "!"*30)
    #     print("DIAGNOSING INFEASIBLE SOLUTION")
    #     print("!"*30)
        
    #     # 1. Check for basic variable bound violations (e.g., negative gauge)
    #     print("\n--- Checking Variable Bounds ---")
    #     log_infeasible_bounds(model, logger=logging.getLogger('pyomo.util.infeasible'))
        
    #     # 2. Check which specific constraints were violated
    #     # log_expression=True helps you see the actual math causing the conflict
    #     print("\n--- Checking Violated Constraints ---")
    #     log_infeasible_constraints(model, log_expression=True, log_variables=True)
        
    #     # 3. Print the current values of key variables to see where the solver gave up
    #     print("\n--- Final Model State (Current Variable Values) ---")
    #     print("\n--- Final Model State (Current Variable Values) ---")
    #     for v in model.component_objects(pyo.Var, active=True):
    #         if v.is_indexed():
    #             # If it's indexed (like weights[ticker]), loop through the indices
    #             for index in v:
    #                 val = pyo.value(v[index], exception=False)
    #                 print(f"Variable {v.name}[{index}]: {val}")
    #         else:
    #             # If it's a single variable (like total_risk)
    #             print(f"Variable {v.name}: {pyo.value(v, exception=False)}")
            
    #     print("!"*30 + "\n")
    
    obj_value = pyo.value(model.OBJ)
    if obj_value < _params['obj_threshold']:
        threshold_value = (1 - params['objective_sensitivity']) * obj_value
        def obj_near_optimal_constraint(model):
            return - sum(model.feature_weight[w] * model.feature_values[s, w] * model.x[s,a] for w in model.feature for s in model.stock for a in model.account) <= threshold_value
        model.near_optimal_constraint = pyo.Constraint(rule = obj_near_optimal_constraint)
        model.del_component(model.OBJ)
        model.num_trades_obj = pyo.Objective(rule=lambda m: sum(m.t[s,a] for s in m.stock for a in m.account), sense = pyo.minimize)
        with SuppressOutput(): solver.solve(model)
        df_sol = var_to_pivot_table(model.x).loc[params['current_price'].index, :]
    else:
        df_sol = pd.DataFrame(0, index = params['current_price'].index, columns = range(len(params['principal'])))

    _investment = np.matmul(pd.DataFrame(params['current_price']).loc[params['current_price'].index != 'GIC'].transpose().values, df_sol.loc[df_sol.index != 'GIC'].values)
    num_trades = (df_sol != params['holdings']).values.sum()
    df_sol = df_sol.astype('float')
    df_sol.loc['GIC', :] = (params['budget'] - _investment - float(params['trade_fee'] * num_trades))
    df_sol = pd.concat([df_sol, pd.DataFrame(0, index = drop_stock_list, columns = df_sol.columns)]).loc[holdings.index]
    return df_sol, num_trades * params['trade_fee'], obj_value

def save_result_agnostic(df, path, session = None):
    """
    Saves a single evaluation result (row) to a destination.
    
    Local Logic: 
        Appends the row to a single CSV file.
    S3 Logic: 
        Writes a unique CSV to the directory using the sim_id as the filename.
    """
    if path.startswith("s3://"):
        # Ensure we have a sim_id to name the file
        if 'sim_id' in df.columns and not df.empty:
            # We take the first sim_id (assuming one eval per call)
            sim_id = df['sim_id'].iloc[0]
            
            # Clean path to ensure it's treated as a directory
            base_dir = path if path.endswith('/') else path + '/'
            s3_path = f"{base_dir}sim_{sim_id}.csv"
            
            # Use awswrangler to write the individual file
            wr.s3.to_csv(df=df, path=s3_path, index=False, boto3_session=session)
        else:
            raise ValueError("S3 destination requires 'sim_id' in DataFrame for unique naming.")
            
    else:
        # Local Logic: Standard file append
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Append to the file (write header only if file doesn't exist)
        file_exists = os.path.isfile(path)
        df.to_csv(path, mode='a', index=False, header=not file_exists)
        
        
        
        
def simulate(df_price, _params, data_features, df_weights, training_period, data_gic, holdings_outfile, sim_id = None, session = None):
    params = _params.copy()
    params.update(training_period)
    train_start_dates = [d for d in df_price if params['train_start_date'] <= d <= params['end_date'] - pd.Timedelta(weeks = params['feature_horizon_weeks'])]
    train_end_dates = [d + pd.Timedelta(weeks = params['feature_horizon_weeks']) for d in train_start_dates]
    time_tups = list(zip(train_end_dates, [d + pd.Timedelta(weeks = 4) for d in train_end_dates]))
    
    holdings = pd.DataFrame(0.0, index = df_price.index, columns = range(len(params['principal'])))
    holdings.loc['GIC', :] = params['principal']
    history = []
    df_holdings_history = pd.DataFrame()

    for val_start_date, val_end_date in time_tups:
        current_price = df_price.loc[:, val_start_date].copy(); current_price.loc['GIC'] = 1
        budget = (holdings.values * pd.DataFrame(current_price).fillna(0).values).sum(axis = 0)
        df_features = data_features.sel(date = val_start_date).to_pandas().transpose()
       
        holdings, _, _ = optimize(params, df_features, current_price, holdings, budget, feature_weights)
        holdings_out = pd.DataFrame((holdings.sum(axis = 1).values * current_price)).transpose().reset_index().rename(columns = {'index': 'date'})
        holdings_out.loc[:, 'sim_id'] = sim_id

        df_holdings_history = pd.concat([df_holdings_history, holdings_out])
        
        

        gic_multiplier = 1 + np.array(data_features.sel(symbol = 'GIC', date = val_end_date, band = 'dollar_ret_1p'))
        stocks = sorted(list(set(holdings.index).difference(['GIC'])))
        v_start = (holdings.loc[stocks] * df_price.loc[stocks, [val_start_date]].values).sum(axis=0) + holdings.loc['GIC']
        v_end = (holdings.loc[stocks] * df_price.loc[stocks, [val_end_date]].values).sum(axis=0) + gic_multiplier * holdings.loc['GIC']
        history.append((val_start_date, val_end_date, v_start.sum(), v_end.sum()))
        print(sim_id, val_start_date, val_end_date, v_start.sum(), v_end.sum(), flush = True)  

    stagger_delay = (sim_id % 5000) / 1000.0
    time.sleep(stagger_delay)
    save_result_agnostic(df_holdings_history, holdings_outfile, session = session)
    

    return history, df_holdings_history

def get_gic_eps(data_gic):
    df_gic = data_gic.sel(symbol = 'GIC').to_pandas().transpose().iloc[:, 1:]
    df_gic['avg_eps_1q'] = (1 + df_gic.dollar_ret_1p)**((365/4)/(28))-1
    df_gic['avg_eps_2q'] = (1 + df_gic.dollar_ret_6p)**((365/2)/(6*28))-1
    df_gic['avg_eps_4q'] = (1 + df_gic.dollar_ret_13p)**(365/(13*28))-1
    df_gic['avg_eps_8q'] = (1 + df_gic.dollar_ret_26p)**((2*365)/(26*28))-1
    return df_gic[[c for c in df_gic.columns if 'eps' in c]].stack().to_xarray().expand_dims(symbol=['GIC']).transpose('band','symbol','date')

def interpolate_to_4week_grid(da, anchor_date):
    days = (pd.to_datetime(da.date) - pd.to_datetime(anchor_date)).days
    da_numeric = da.assign_coords(date=days)
    new_coords = np.arange(int(days.min()//28)*28, int(days.max()//28+1)*28, 28)
    da_interp = da_numeric.interp(date=new_coords, method="linear")
    return da_interp.assign_coords(date=pd.to_datetime(anchor_date) + pd.to_timedelta(da_interp.date.values, unit='D'))

# --- Main Cluster Exploration Script ---

