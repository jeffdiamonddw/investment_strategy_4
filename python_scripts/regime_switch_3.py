import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"




import numpy as np
import pandas as pd
import joblib
import os
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

# Import custom logic from explore_cluster.py
from explore_cluster import (
    simulate, 
    optimize, 
    interpolate_to_4week_grid, 
    get_gic_eps,
    SuppressOutput
)

# --- REGIME-SWITCHING PROBLEM CLASS ---

class RegimeSwitchingProblem(ElementwiseProblem):
    def __init__(self, mom_kit, qual_kit, df_macro, data_features, data_gic, df_price, params, training_periods, holdings_file, xl=None, xu=None):
        self.mom_kit = mom_kit
        self.qual_kit = qual_kit
        self.df_macro = df_macro
        self.data_features = data_features
        self.data_gic = data_gic
        self.df_price = df_price
        self.params = params
        self.training_periods = training_periods
        self.holdings_file = holdings_file
        
        # Output file paths
        self.eval_history_file = "sim_results/regime_eval_history.csv"
        self.detailed_sim_file = "sim_results/detailed_sim_history.csv"
        
        self.all_feature_names = data_features.band.values.tolist()
        
        
        super().__init__(n_var=16, n_obj=5, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x_numeric = x.X if hasattr(x, "X") else x
        sim_id = abs(hash(tuple(x_numeric))) % (10**10) 
        session = kwargs.get('session')
        # REPAIR: Use self.m_kit and self.q_kit to match __init__
        w_mom_vals = self.m_kit['scaler'].inverse_transform(
            self.m_kit['pca'].inverse_transform(x_numeric[0:4].reshape(1, -1))
        ).flatten()
        w_qual_vals = self.q_kit['scaler'].inverse_transform(
            self.q_kit['pca'].inverse_transform(x_numeric[4:8].reshape(1, -1))
        ).flatten()

        # REPAIR: Convert to dictionaries using the exact kit column names
        # This prevents the TypeError by ensuring numeric lookups in regime_switch_3.py
        w_mom = dict(zip(self.m_kit['columns'], np.clip(w_mom_vals, 0, 1)))
        w_qual = dict(zip(self.q_kit['columns'], np.clip(w_qual_vals, 0, 1)))

        opt_threshold = x_numeric[8]
        opt_beta = x_numeric[9]
        mom_decay = x_numeric[10]
        qual_decay = x_numeric[11]
        macro_weights = x_numeric[12:16]/sum(abs(x_numeric[12:16]))

        # Standard simulation calls
        df_h_crash = self.base.run_blended_sim(w_mom, w_qual, opt_threshold, opt_beta, mom_decay, qual_decay, macro_weights, 'crash', sim_id, session = session)
        df_h_boom = self.base.run_blended_sim(w_mom, w_qual, opt_threshold, opt_beta, mom_decay, qual_decay, macro_weights, 'boom', sim_id, session = session)
        
        df_sim = pd.concat([df_h_crash, df_h_boom]).reset_index(drop=True)
        if df_sim.empty:
            out["F"] = [9999999] * 5
            return

        ticker_cols = [c for c in df_sim.columns if c not in ['sim_id', 'date', 'total_value', 'pct_change']]
        df_sim['total_value'] = df_sim[ticker_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        
        # Objective Calculations
        f1 = np.maximum(0, df_h_crash[ticker_cols].sum(axis=1).iloc[0] - df_h_crash[ticker_cols].sum(axis=1)).sum() if not df_h_crash.empty else 9999999
        df_2020 = df_sim[(df_sim['date'] >= '2020-02-01') & (df_sim['date'] <= '2020-06-01')]
        f2 = np.maximum(0, df_2020['total_value'].iloc[0] - df_2020['total_value']).sum() if not df_2020.empty else 9999999
        df_2022 = df_sim[(df_sim['date'] >= '2022-01-01') & (df_sim['date'] <= '2022-12-31')]
        f3 = np.maximum(0, df_2022['total_value'].iloc[0] - df_2022['total_value']).sum() if not df_2022.empty else 9999999
        f4 = df_sim.iloc[-1]['total_value']
        f5 = df_sim['total_value'].pct_change().dropna().quantile(ANNUAL_RISK_PERCENTILE)

        # Restored Indexing for df_out
        columns = [
            'sim_id', 'f1_2008', 'f2_2020', 'f3_2022', 'f4_terminal', 'f5_annual_worst',
            'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p',
            'avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q', 
            'threshold', 'beta'
        ]
        
        # Explicit mapping of x_numeric indices to match the 16 required columns
        values = [
            sim_id, f1, f2, f3, f4, f5, 
            x_numeric[0], x_numeric[1], x_numeric[2], x_numeric[3], # Momentum PCA components
            x_numeric[4], x_numeric[5], x_numeric[6], x_numeric[7], # Quality PCA components
            x_numeric[8], x_numeric[9]                              # Threshold and Beta
        ]
        
        df_out = pd.DataFrame({columns[i]: [values[i]] for i in range(16)})
        save_result_agnostic(df_out, EVAL_FILE)

        out["F"] = [f1, f2, f3, -f4, -f5]

    def run_blended_sim(self, w_mom_vals, w_qual_vals, threshold, beta, mom_decay, qual_decay, macro_weights, period_key, sim_id, session = None):
        period = self.training_periods[period_key]
        
        
        s_risk_aversion = self.df_macro.loc[period['train_start_date']:period['end_date']].dot(macro_weights).rename("risk_aversion")   
        
        
        s_quality_weight = 1 / (1 + np.exp(-beta * (s_risk_aversion - threshold)))
        mom_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in self.mom_kit['columns']])
        qual_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in self.qual_kit['columns']])
        df_mom_decay = pd.DataFrame(np.exp(- mom_decay * (s_risk_aversion.mean() - s_risk_aversion).values[:, None] * mom_num_periods), index=s_risk_aversion.index, columns = self.mom_kit['columns'])
        df_qual_decay = pd.DataFrame(np.exp(- qual_decay * (s_risk_aversion.mean() - s_risk_aversion).values[:, None] * qual_num_periods), index=s_risk_aversion.index, columns = self.qual_kit['columns'])
        
        
        w_dict = {}
        df_mom = pd.DataFrame({self.mom_kit['columns'][j]: [w_mom_vals[j]] for j in range(len(w_mom_vals))})
        df_qual = pd.DataFrame({self.qual_kit['columns'][j]: [w_qual_vals[j]] for j in range(len(w_mom_vals))})
        
        df_mom_weights = df_mom_decay.mul(df_mom.iloc[0], axis=1).mul(1 - s_quality_weight, axis=0)
        df_qual_weights = df_qual_decay.mul(df_qual.iloc[0], axis=1).mul(s_quality_weight, axis=0)
        df_weights = pd.concat([df_mom_weights, df_qual_weights], axis = 1)
        h_raw, df_holdings = simulate(self.df_price, self.params, self.data_features, df_weights, period, self.data_gic, self.holdings_file, sim_id, session = session)
        
        # Log detailed output with sim_id for matching
        #df_h = pd.DataFrame(h_raw, columns=['date', 'symbol', 'price', 'capital'])
        #df_h['sim_id'], df_h['regime'], df_h['s_blend'] = sim_id, period_key, s_val
        #df_h.to_csv(self.detailed_sim_file, mode='a', index=False, header=not os.path.exists(self.detailed_sim_file))
        
        
        #boom_eval = h_raw[-1][3]
        #perc_10_diff_eval = np.percentile(np.diff(df_h.capital)/df_h.capital[:-1], 100/13)
        return df_holdings
    

    def simulate(self, w_mom_vals, w_qual_vals, threshold, beta, sim_id):
        """
        Standard interface to rerun a full backtest for both regimes.
        This bypasses PCA mapping and uses weight values directly.
        """
        # Run Crash Regime
        self.run_blended_sim(w_mom_vals, w_qual_vals, threshold, beta, 'crash', sim_id)
        # Run Boom Regime
        self.run_blended_sim(w_mom_vals, w_qual_vals, threshold, beta, 'boom', sim_id)
        

# --- MAIN EXECUTION ---

