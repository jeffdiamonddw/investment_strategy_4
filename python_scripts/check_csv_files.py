import awswrangler as wr
import boto3

s3 = boto3.client("s3")
# List files in the bucket/prefix


bucket = "jdinvestment"
folder_path = "perturbation_evaluations_17_year"#sys.argv[1]
database_name = folder_path
table_name = folder_path
s3_path = f"s3://{bucket}/{folder_path}"

files = [f for f in wr.s3.list_objects(s3_path) if f.endswith('.csv')]

i =0
for file in files:
    if i % 100 == 0:
        print(i,len(files))
    try:
        print(f"Testing: {file}")
        df = wr.s3.read_csv(path=file)
    except UnicodeDecodeError:
        print(f"!!! ENCODING ERROR in file: {file}")
        raise
    except Exception as e:
        print(f"!!! OTHER ERROR in {file}: {e}")
        raise