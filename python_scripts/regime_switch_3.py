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
    def __init__(self, mom_kit, qual_kit, df_macro, data_features, data_gic, df_price, params, training_periods, holdings_file):
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
        
        # 10 Variables: 4 Mom PCs, 4 Qual PCs, Sigmoid Threshold, Sigmoid Beta
        xl = np.array([-4.0, -2.0, -2.0, -2.0, -4.0, -2.0, -2.0, -2.0, -2.0, 0.1])
        xu = np.array([ 8.0,  2.0,  2.0,  2.0,  8.0,  2.0,  2.0,  2.0,  4.0, 10.0])
        
        super().__init__(n_var=10, n_obj=3, xl=xl, xu=xu)

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

        # Standard simulation calls
        df_h_crash = self.base.run_blended_sim(w_mom, w_qual, opt_threshold, opt_beta, 'crash', sim_id, session = session)
        df_h_boom = self.base.run_blended_sim(w_mom, w_qual, opt_threshold, opt_beta, 'boom', sim_id, session = session)
        
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

    def run_blended_sim(self, w_mom_vals, w_qual_vals, threshold, beta, period_key, sim_id, session = None):
        period = self.training_periods[period_key]
        m = self.df_macro.loc[period['train_start_date']:period['end_date']]
        if m.empty: return 999999999
            
        fear = (m['VIX_z'] + m['FED_RATE_z'] + m['BOND_VOL_z'])
        s_val = 1 / (1 + np.exp(-beta * (fear.mean() - threshold)))
        
        w_dict = {feat: 0.0 for feat in self.all_feature_names}
        d_mom = dict(zip(self.mom_kit['columns'], w_mom_vals))
        d_qual = dict(zip(self.qual_kit['columns'], w_qual_vals))
        
        for f in self.mom_kit['columns']: w_dict[f] = (1 - s_val) * d_mom[f]
        for f in self.qual_kit['columns']: w_dict[f] += s_val * d_qual[f]
        
        h_raw, df_holdings = simulate(self.df_price, self.params, self.data_features, w_dict, period, self.data_gic, self.holdings_file, sim_id, session = session)
        
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

if __name__ == "__main__":
    # 1. Macro Data Setup
    df_macro = pd.read_csv('simulation_data/macro_indicators.csv', parse_dates=['Date'])
    df_macro = df_macro.set_index('Date').sort_index()
    for col in ['VIX', 'FED_RATE', 'BOND_VOL']:
        df_macro[f'{col}_z'] = (df_macro[col] - df_macro[col].mean()) / df_macro[col].std()

    # 2. Environment Setup
    stocks = sorted(list(set(pd.read_csv('strategy/stock_list_with_etfs.csv').symbol.values)))
    data_price_momentum = xr.open_dataarray('simulation_data/price_and_dollar_returns.nc')
    data_gic = xr.open_dataarray('simulation_data/gic_data.nc')
    data_momentum = xr.concat([data_price_momentum[1:], data_gic[1:]], dim='symbol')
    anchor = np.array(data_momentum.date).min()
    data_eps = interpolate_to_4week_grid(xr.open_dataarray('simulation_data/eps.nc'), anchor)
    data_etf_eps = interpolate_to_4week_grid(xr.open_dataarray('simulation_data/etf_eps.nc'), anchor)
    data_eps = xr.concat([data_eps, data_etf_eps, get_gic_eps(data_gic)], dim='symbol', join='outer')
    data_features = xr.concat([data_momentum, data_eps], dim='band', join='outer').sel(symbol=stocks + ['GIC'])
    df_price = data_price_momentum.sel(band='price_end').to_pandas().loc[stocks]

    params = {
        'principal': [327000, 60000, 21000], 'max_frac': .05, 'feature_horizon_weeks': 104,
        'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
        'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
    }
    training_periods = {
        'boom': {'train_start_date': pd.to_datetime('Jan 1, 2018'), 'end_date': pd.to_datetime('Jan 1, 2025')},
        'crash': {'train_start_date': pd.to_datetime('Sep 1, 2005'), 'end_date': pd.to_datetime('Sep 1, 2012')}
    }

    # 3. Load History and Prepare PCA Kits
    files = ['sim_results/mobo_results_1.csv', 'sim_results/mobo_results_2.csv', 
             'sim_results/pca_refined_results.csv', 'sim_results/pca_warm_start_results.csv']
    df_history = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)])
    mom_cols = ['dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p']
    qual_cols = ['avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q']
    df_top = df_history.nsmallest(int(len(df_history) * 0.10), 'crash_eval').copy()
    
    def get_kit(df, cols):
        s = StandardScaler().fit(df[cols])
        p = PCA(n_components=4).fit(s.transform(df[cols]))
        return {'scaler': s, 'pca': p, 'columns': cols}

    m_kit, q_kit = get_kit(df_top, mom_cols), get_kit(df_top, qual_cols)

    # 4. Warm Start Population
    df_ws = df_history.nsmallest(100, 'crash_eval')
    x_mom_pca = m_kit['pca'].transform(m_kit['scaler'].transform(df_ws[mom_cols]))
    x_qual_pca = q_kit['pca'].transform(q_kit['scaler'].transform(df_ws[qual_cols]))
    X_warm_start = np.hstack([x_mom_pca, x_qual_pca, np.tile([0.5, 1.0], (100, 1))])

    # 5. Execute Optimization
    problem = RegimeSwitchingProblem(m_kit, q_kit, df_macro, data_features, data_gic, df_price, params, training_periods)
    algorithm = NSGA2(pop_size=100, sampling=Population.new("X", X_warm_start))
    
    print("--- Starting Dual-PCA Regime-Switching Optimization ---")
    res = minimize(problem, algorithm, ('n_gen', 40), seed=1, verbose=True)

    # 6. Save Optimized Final Population
    final_mom = m_kit['scaler'].inverse_transform(m_kit['pca'].inverse_transform(res.X[:, 0:4]))
    final_qual = q_kit['scaler'].inverse_transform(q_kit['pca'].inverse_transform(res.X[:, 4:8]))
    
    res_df = pd.DataFrame(final_mom, columns=mom_cols)
    res_df[qual_cols] = final_qual
    res_df['threshold'], res_df['beta'] = res.X[:, 8], res.X[:, 9]
    res_df['crash_eval'], res_df['boom_eval'] = res.F[:, 0], -res.F[:, 1]
    
    res_df.to_csv("sim_results/regime_warm_start_final.csv", index=False)
    print("Optimization Complete.")