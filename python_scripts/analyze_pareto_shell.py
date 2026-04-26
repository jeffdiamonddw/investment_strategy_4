import awswrangler as wr

database_name = table_name = "new_evaluations_6"
# Read the table metadata and data using the Glue catalog
df = wr.s3.read_parquet_table(
    table=table_name,
    database=database_name
)

print(df.f5_annual_worst.head())