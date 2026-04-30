import awswrangler as wr
import pandas as pd
import time

# Configuration
bucket = "jdinvestment"
folder_path = "pareto_nav_eval_2"
database_name = folder_path
table_name = folder_path
s3_path = f"s3://{bucket}/{folder_path}"

t1 = time.time()

# 1. Read all CSVs from the S3 folder into a single DataFrame
# This automatically handles schema inference across all files
df = wr.s3.read_csv(path=s3_path)

# 2. Create the Glue Database (if it doesn't exist)
if database_name not in wr.catalog.databases().values:
    wr.catalog.create_database(name=database_name)

# 3. Write to Parquet and register in the Data Catalog
# This effectively 'constructs the database' by linking the schema to the S3 files
wr.s3.to_parquet(
    df=df,
    path=s3_path,  # Saving back to the same folder
    dataset=True,
    database=database_name,
    table=table_name,
    mode="overwrite"  # Use 'append' if you don't want to replace existing Parquet files
)

print(f"Successfully converted CSVs to Parquet in {s3_path} and registered table '{table_name}'.")
print("time: {}".format(time.time() - t1))

zzz=1