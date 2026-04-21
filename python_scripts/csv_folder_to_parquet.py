import awswrangler as wr
import pandas as pd

# Configuration
bucket = "jdinvestment"
folder_path = "new_holdings_history_5"
database_name = "new_holdings_history_5"
table_name = "new_holdings_history_5"
s3_path = f"s3://{bucket}/{folder_path}"

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