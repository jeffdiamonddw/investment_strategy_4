import awswrangler as wr

s3_path = "s3://jdinvestment/new_evaluations_5"

# Read all parquet files in the folder into one DataFrame
df = wr.s3.read_parquet(path=s3_path, dataset=True)

print(df.head())