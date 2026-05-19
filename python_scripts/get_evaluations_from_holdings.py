import awswrangler as wr
from pareto_navigator import mean_annual_drawdown_integral, annualized_return, pct_change_quantile, worst_annual_return


holdings_folder = "s3://jdinvestment/perturbation_holdings_17_year"

# Returns a list of S3 paths (e.g., ['s3://bucket/folder/file1.csv', ...])
file_list = [f for f in wr.s3.list_objects(path=holdings_folder) if f.endswith('.csv')]
                                           
for file in file_list:
    df = pd.read_csv(file)
