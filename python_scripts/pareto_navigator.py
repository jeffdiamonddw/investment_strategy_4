import functools 

import numpy as np
import pandas as pd
import xarray as xr
import time
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from scipy.spatial.distance import pdist, squareform
import networkx as nx
# Ensure these are imported from your local environment
from explore_cluster import save_result_agnostic, simulate, get_gic_eps
from manifold_dry_run_parallel import build_kit, RobustParallelManager, load_checkpoint, save_checkpoint  


STAR_FILE = 'analysis/stars.csv'
MOMENTUM_FILE = "simulation_data/momentum.nc"
QUALITY_FILE = "simulation_data/quality.nc"
GIC_FILE = "simulation_data/gic_data.nc"
MACRO_FILE = "simulation_data/macro_signals.csv"
MANIFOLD_FILE = "sim_results/manifold_triple_threat.csv"
HOLDINGS_FOLDER = "s3://jdinvestment/pareto_nav_holdings_1"
EVAL_FOLDER = "s3://jdinvestment/pareto_nav_eval_1"
CHECKPOINT_URI= "s3://jdinvestment/checkpoints/pareto_nav_checkpoint_2"

POP_SIZE = 3   # Minimal for testing
GEN_COUNT =  300
N_OFFSPRING = 3
NUM_WORKERS = 3
TIMEOUT = 180
TARGET_COMPLETIONS = 2





def get_greedy_path_indices(df, objective_cols):
    """
    Returns a list of integer indices representing a smooth path through 
    objective space, starting from an extreme point.
    """
    # 1. Normalize objectives to prevent scale bias
    objs = df[objective_cols].values
    objs_norm = (objs - objs.min(axis=0)) / (objs.max(axis=0) - objs.min(axis=0) + 1e-9)
    
    # 2. Calculate distance matrix
    dist_matrix = squareform(pdist(objs_norm, metric='euclidean'))
    
    # 3. Find the two most distant points (the 'anchors' of the frontier)
    # We'll pick the one that is "lowest" on the first objective to set a consistent start
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    start_node = i if objs[i, 0] < objs[j, 0] else j
    
    # 4. Greedy path construction
    path = [int(start_node)]
    remaining = set(range(len(df)))
    remaining.remove(start_node)
    
    while remaining:
        current_node = path[-1]
        remaining_list = list(remaining)
        
        # Find index of the closest remaining point
        distances = dist_matrix[current_node, remaining_list]
        closest_node = remaining_list[np.argmin(distances)]
        
        path.append(int(closest_node))
        remaining.remove(closest_node)
        
    return path





def drawdown_integral(start_date, end_date, ticker_cols, df):
    df1 = df.loc[(df.date >= start_date) & (df.date <= end_date), ticker_cols]
    result = np.maximum(0, df1.sum(axis=1).iloc[0] - df1.sum(axis=1)).sum() if not df1.empty else 9999999
    return result

def annualized_return(start_date, end_date, ticker_cols, df):
    df1 = df.loc[(df.date >= start_date) & (df.date <= end_date), ticker_cols]
    total_values = df1.sum(axis = 1)
    num_years = (end_date  - start_date).days/365
    result = (total_values.iloc[-1]/total_values.iloc[0])**(1/num_years) - 1
    return result

def terminal_value(start_date, end_date, ticker_cols, df):
    df1 = df.loc[(df.date >= start_date) & (df.date <= end_date), ticker_cols]
    total_values = df1.sum(axis = 1)
    result = total_values.iloc[-1]
    return result


def pct_change_quantile(start_date, end_date, ticker_cols, quantile, df):
    df1 = df.loc[(df.date >= start_date) & (df.date <= end_date), ticker_cols]
    result = df1.sum(axis = 1).pct_change().dropna().quantile(quantile)
    return result

def get_direction_sign(label):
    """Returns -1 for 'max' and +1 for 'min'."""
    return -1 if label.lower() == 'max' else 1 if label.lower() == 'min' else None

class ParetoNavigator(ElementwiseProblem):
    
    def __init__(self, df_stars, weight_columns, df_macro, 
                 mom_kit, qual_kit, data_features, df_price, params, 
                 training_periods, holdings_folder, eval_folder, 
                 objective_functions_dict, objective_sense, xl, xu):
        """
        17-Parameter Navigator [Jeff Diamond Analytics]
        
        Args:
            df_stars: Combined DF containing both objectives and star parameters.
            weight_columns: List of strings identifying the 16 parameter columns in df_stars.
            df_macro: 4 macro signals for 2D regime mapping.
            df_macro: High-freq macro signals for decay and sigmoid logic.
            objective_functions_dict: { 'name': func(df_sim) -> float }.
            active_objective_names: Subset of objective_functions_dict keys for Pymoo.
        """
        # 1. Unified Star Manifest
        self.df_stars = df_stars
        self.weight_columns = weight_columns
        
        # 2. Data and Environment
        self.df_macro = df_macro
        self.mom_kit = mom_kit
        self.qual_kit = qual_kit
        self.data_features = data_features
        self.df_price = df_price
        self.params = params
        self.training_periods = training_periods
        self.holdings_folder = holdings_folder
        self.eval_folder = eval_folder
        
        self.obj_funcs = objective_functions_dict
        self.objective_sense = objective_sense
        
        # 3. Internal State
        self.prev_target_pos = None

        

        super().__init__(n_var=17, n_obj=len(objective_sense), n_constr=0, xl=xl, xu=xu)

    def __getstate__(self):
        """Minimize pickle size for parallel worker distribution."""
        state = self.__dict__.copy()
        for key in ['df_macro',  'data_features', 'df_price', 'df_stars']:
            state[key] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_2d_coord(self, current_date, p_12):
        """Temporal encoder: Signal History -> RM/RP Coordinates."""
        start = current_date - pd.DateOffset(years=2)
        history = self.df_macro.loc[start:current_date]
        alphas = p_12[0:4]
        smoothed = np.array([history[col].ewm(alpha=alphas[i]).mean().iloc[-1] for i, col in enumerate(history.columns)])
        rm = np.dot(smoothed, p_12[4:8])
        
        history_3m = history.loc[current_date - pd.DateOffset(months=3):]
        rp_ratios = history_3m.mean() / (history.mean() + 1e-9)
        rp = np.dot(rp_ratios.values, p_12[8:12])
        return rm, rp





    def get_2d_coord_vectorized(self, dates, p_12):
        """
        Vectorized Temporal encoder: 
        Takes a list of dates and returns a (num_dates x 2) DataFrame of RM/RP.
        """
        # 1. Setup Parameters
        alphas = p_12[0:4]
        w_rm = p_12[4:8]
        w_rp = p_12[8:12]
        
        # 2. Vectorized RM Calculation
        # Calculate EWM for the entire history once
        # We use adjust=True to match the behavior of building up from history
        df_smoothed = pd.DataFrame(index=self.df_macro.index)
        for i, col in enumerate(self.df_macro.columns):
            df_smoothed[col] = self.df_macro[col].ewm(alpha=alphas[i], adjust=True).mean()
        
        # Dot product of smoothed columns with RM weights at every timestep
        # rm_series is (Total_History_Length,)
        rm_series = df_smoothed.values @ w_rm

        # 3. Vectorized RP Calculation (Rolling Windows)
        # We need a 3-month mean and a 2-year mean for every point in time
        # Assuming ~21 business days per month; 3m ~ 63 days, 2y ~ 504 days
        # Better: Use offset-based rolling for accuracy
        mean_3m = self.df_macro.rolling(window='90D').mean()
        mean_2y = self.df_macro.rolling(window='730D').mean()
        
        rp_ratios = mean_3m / (mean_2y + 1e-9)
        # rp_series is (Total_History_Length,)
        rp_series = rp_ratios.values @ w_rp

        # 4. Alignment & Extraction
        # Create a Master DF of all calculated signals
        df_signals = pd.DataFrame({
            'rm': rm_series,
            'rp': rp_series
        }, index=self.df_macro.index)

        # Reindex to the specific dates requested (using ffill to handle weekends/holidays)
        # This replaces the need to slice 'history' inside a loop
        result = df_signals.reindex(dates, method='ffill')
        
        return result

    def _calculate_star_xarray(self, period_key):
        """
        Vectorized weight calculation for all 13 stars.
        Outputs: xarray.DataArray with dims (star, date, feature)
        """
        period = self.training_periods[period_key]
        macro_slice = self.df_macro.loc[period['train_start_date']:period['end_date']]
        mom_horizons = np.array([int(c.split('_')[-1][:-1]) for c in self.mom_kit['columns']])
        qual_horizons = np.array([int(c.split('_')[-1][:-1]) for c in self.qual_kit['columns']])
        
        feature_names = list(self.mom_kit['columns']) + list(self.qual_kit['columns'])
        star_ids = self.df_stars.index.tolist()
        
        full_tensor = np.zeros((len(star_ids), len(macro_slice), 8))

        # Slicing df_stars by weight_columns to extract 16 parameters per expert
        for i, (_, star) in enumerate(self.df_stars[self.weight_columns].iterrows()):
            w_mom, w_qual = star.iloc[0:4].values, star.iloc[4:8].values
            threshold, beta, m_decay, q_decay = star.iloc[8:12]
            macro_weights = star.iloc[12:16].values
            
            ra = macro_slice.dot(macro_weights)
            s_qual = 1 / (1 + np.exp(-beta * (ra - threshold)))
            
            m_decay_mat = np.exp(-m_decay * (ra.mean() - ra).values[:, None] * mom_horizons)
            q_decay_mat = np.exp(-q_decay * (ra.mean() - ra).values[:, None] * qual_horizons)
            
            df_m = (m_decay_mat * w_mom) * (1 - s_qual.values[:, None])
            df_q = (q_decay_mat * w_qual) * s_qual.values[:, None]
            
            full_tensor[i] = np.hstack([df_m, df_q])

        return xr.DataArray(
            full_tensor,
            coords={'star': star_ids, 'date': macro_slice.index, 'feature': feature_names},
            dims=['star', 'date', 'feature']
        )

    def _evaluate(self, x, out, *args, **kwargs):
        t_start = time.time()
        x_num = x.X if hasattr(x, "X") else x
        sim_id = abs(hash(tuple(x_num))) % (10**10)
        self.prev_target_pos = None 
        
        # 1. Path Linearization (using objective columns in df_stars)
        objs = self.df_stars[list(self.objective_sense.keys())].values
        if objs.shape[1] > 2:
            objs_norm = (objs - objs.min(axis=0)) / (objs.max(axis=0) - objs.min(axis=0) + 1e-9)
            path_coords = np.cumsum(np.sqrt(np.sum(np.diff(objs_norm, axis=0)**2, axis=1)))
            path_coords = np.insert(path_coords, 0, 0)
        else:
            path_coords = objs[:,0]
        
        sim_results = []
        for period_key in self.training_periods:
            # 2. Get (13, days, 8) star weight tensor
            da_stars = self._calculate_star_xarray(period_key)
            df_2d_regime = self.get_2d_coord_vectorized(da_stars.date.values, x_num[:12])
            
           

            # 1. Pull parameters
            m_p = x_num[12:]
            sigma = max(0.01, m_p[0])
            hyst_threshold = m_p[4]

            # 2. Vectorized Raw Position Calculation
            # (rm * (1 + m_p[3] * rp) * m_p[1]) + m_p[2]
            rm = df_2d_regime['rm'].values
            rp = df_2d_regime['rp'].values
            raw_pos_vec = (rm * (1 + m_p[3] * rp) * m_p[1]) + m_p[2]

            # 3. Asymmetric Hysteresis (Sequential state tracking)
            # We calculate the target path coordinate for every day
            num_days = len(raw_pos_vec)
            targets = np.zeros(num_days)

            # Initialize state
            if self.prev_target_pos is None:
                self.prev_target_pos = raw_pos_vec[0]

            curr_target = self.prev_target_pos

            for i in range(num_days):
                rp_val = raw_pos_vec[i]
                # Asymmetric: immediate follow on increase, delayed follow on decrease
                if rp_val >= curr_target:
                    curr_target = rp_val
                else:
                    # Only drop if the move exceeds the threshold
                    if (curr_target - rp_val) >= hyst_threshold:
                        curr_target = rp_val
                targets[i] = curr_target

            self.prev_target_pos = curr_target  # Save for next period/batch

            # 4. Fully Vectorized Gaussian Blending (Outer Product)
            # targets is (days,), path_coords is (13,)
            # Resulting diff_sq is (days, 13)
            diff_sq = (targets[:, np.newaxis] - path_coords[np.newaxis, :])**2
            blending_data = np.exp(-diff_sq / (2 * sigma**2))

            # 5. Row-wise Normalization (Days x Stars)
            blending_data /= (blending_data.sum(axis=1, keepdims=True) + 1e-9)
            
            da_blending = xr.DataArray(
                blending_data, 
                coords={'date': da_stars.date, 'star': da_stars.star}, 
                dims=['date', 'star']
            )

            # 4. Vectorized Blending: (days, 13) dot (13, days, 8) -> (days, 8)
            da_final_weights = (da_blending * da_stars).sum(dim='star')
            df_weights = da_final_weights.to_pandas()

            # 5. Simulation execution
            _, df_h = simulate(self.df_price, self.params, self.data_features, df_weights, 
                               self.training_periods[period_key], self.holdings_folder, sim_id)
            sim_results.append(df_h)
            
        df_all = pd.concat(sim_results).reset_index()
        
        # 6. Performance Evaluation and Result Logging
        obj_results = {name: func(df_all) for name, func in self.obj_funcs.items()}
        out["F"] = [get_direction_sign(self.objective_sense[key]) * obj_results[key] for key in self.objective_sense.keys()] 
        
        save_result_agnostic(pd.DataFrame([{**{'sim_id': sim_id}, **obj_results, **{f'p_{i}': v for i, v in enumerate(x_num)}}]), self.eval_folder)
        print(f"ID: {sim_id} | Time: {time.time()-t_start:.2f}s | Objs: {out['F']}")


import numpy as np
from scipy.stats import truncnorm

def sample_bounded_vector(xl, xu, mean_factor=0.5, std_factor=0.25):
    """
    Samples a vector x within [xl, xu] with customizable mean and std.
    
    Args:
        xl (array-like): Lower bounds.
        xu (array-like): Upper bounds.
        mean_factor (float): Position of the mean relative to range (0.5 = center).
        std_factor (float): SD as a fraction of the total range (0.25 = 1/4 range).
    """
    xl = np.array(xl)
    xu = np.array(xu)
    delta = xu - xl
    
    # 1. Define mu and sigma based on factors
    mu = xl + (mean_factor * delta)
    sigma = std_factor * delta
    
    # 2. Calculate truncation bounds relative to the normal distribution
    # We add a tiny epsilon to sigma to prevent division by zero on fixed bounds
    a = (xl - mu) / (sigma + 1e-9)
    b = (xu - mu) / (sigma + 1e-9)
    
    # 3. Sample from truncated normal
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)




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
    df_stars = pd.read_csv(STAR_FILE)


    #order stars along path in objective space
    stars_path = get_greedy_path_indices(df_stars, list(objective_sense.keys()))
    df_stars = df_stars.iloc[stars_path].reset_index()

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
    
    
    
    
    # Attempt to load existing progress
    algorithm = load_checkpoint(CHECKPOINT_URI)
    
    if algorithm is None:
        print("Initial Startup: Building RANDOM population...")
        
        
        # 4. Initialize Manager and NSGA2 (Default sampling is FloatRandomSampling)
        problem = RobustParallelManager(NUM_WORKERS, TIMEOUT, ParetoNavigator, problem_args, var_count, obj_count, xl, xu, TARGET_COMPLETIONS)
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            n_offsprings=N_OFFSPRING,
            eliminate_duplicates=True
        )
        algorithm.setup(problem, termination=('n_gen', GEN_COUNT), seed=1)
    else:
        print(f"Resuming from Generation {algorithm.n_gen}...")
        
        # 1. Initialize the fresh manager
        problem = RobustParallelManager(NUM_WORKERS, TIMEOUT, ParetoNavigator, problem_args, var_count, obj_count, xl, xu, TARGET_COMPLETIONS)
        
        # Sync the generation counts
        problem.n_gen = algorithm.n_gen 
        algorithm.problem = problem

        # 2. FORCE OVERRIDE TERMINATION
        # We re-import the termination factory to ensure it's fresh
        from pymoo.termination import get_termination
        algorithm.termination = get_termination("n_gen", GEN_COUNT)
        
        # 3. CRITICAL: Reset the 'has_finished' flag if it exists
        algorithm.has_terminated = False
    

    # --- 4. EXECUTION LOOP ---
    while algorithm.has_next():
        infills = algorithm.ask()
        print(f"--- Gen {algorithm.n_gen} Evaluation Start ---")
        
        # In a parallel version, you'd use your input_queue logic here. 
        # For this dry run, it uses the standard evaluator.
        algorithm.evaluator.eval(algorithm.problem, infills)
        
        algorithm.tell(infills=infills)
        
        # Checkpoint at the end of every successful generation
        save_checkpoint(algorithm, CHECKPOINT_URI)
        print(f"Gen {algorithm.n_gen} Success. Checkpoint saved.")

    print("--- Optimization Complete ---")
    algorithm.problem.cleanup()
    # At the very end of your script
    
    print("Optimization Complete")
    #os.system("sudo shutdown -h now")
   