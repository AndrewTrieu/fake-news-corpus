import random
import pandas as pd
import os
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq

parquet_path = "../data/processed_fakenews.parquet"
csv_path = "../data/processed_fakenews.csv"
sample_path = "../data/sampled_fakenews"
SAMPLE_FRACTION = 0.1
RANDOM_SEED = 42  # For reproducibility

def get_sample_size(total_rows, log=False):
    sample_size = int(total_rows * SAMPLE_FRACTION)
    if log:
        print(f"📉 Reducing dataset from {total_rows:,} to {sample_size:,} rows...")
    return sample_size

def sample_dataframe(df, total_rows):
    sample_size = get_sample_size(total_rows=total_rows, log=True)
    return df.sample(n=sample_size, random_state=RANDOM_SEED)

# Try to load from Parquet first, fall back to CSV if not available
if os.path.exists(parquet_path):
    print(f"🔍 Loading data from Parquet file at '{parquet_path}'")
    try:
        # Read metadata to get row count without loading entire file
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"🔍 Dataset contains {total_rows:,} rows.")
        
        # Read and sample the data
        df_sample = sample_dataframe(pd.read_parquet(parquet_path), total_rows)
        
    except Exception as e:
        print(f"❌ Error reading Parquet file: {e}")
        print("🔄 Falling back to CSV...")
        if not os.path.exists(csv_path):
            print(f"❌ Error: Neither Parquet nor CSV file found at {parquet_path} or {csv_path}")
            exit()
        
        # Get total rows from CSV (Unix-like systems only due to `wc`)
        total_rows = int(subprocess.check_output(["wc", "-l", csv_path]).split()[0]) - 1
        print(f"🔍 Dataset contains {total_rows:,} rows.")
        
        # Read and sample the data
        df_sample = sample_dataframe(
            pd.read_csv(csv_path, lineterminator="\n", on_bad_lines="skip"),
            total_rows
        )

elif os.path.exists(csv_path):
    print(f"🔍 Parquet file not found, loading from CSV at {csv_path}")
    # Get total rows from CSV (Unix-like systems only due to `wc`)
    total_rows = int(subprocess.check_output(["wc", "-l", csv_path]).split()[0]) - 1
    print(f"🔍 Dataset contains {total_rows:,} rows.")
    
    # Read and sample the data
    df_sample = sample_dataframe(
        pd.read_csv(csv_path, lineterminator="\n", on_bad_lines="skip"),
        total_rows
    )
else:
    print(f"❌ Error: Neither Parquet nor CSV file found at {parquet_path} or {csv_path}")
    exit()

# Verify the sample size
print(f"✅ Sample contains {len(df_sample):,} rows (expected {get_sample_size(total_rows=total_rows):,} rows)")

# Save the sample in both formats
df_sample.to_csv(f"{sample_path}.csv", index=False)
df_sample.to_parquet(f"{sample_path}.parquet", index=False)

print(f"💾 Sample saved to '{sample_path}.csv' and '{sample_path}.parquet'.")

# Split to 80/10/10 and save as both CSV and Parquet
train_size = int(len(df_sample) * 0.8)
valid_size = int(len(df_sample) * 0.1)
test_size = len(df_sample) - (train_size + valid_size)  # Ensure the sum is correct

df_train = df_sample.iloc[:train_size]
df_valid = df_sample.iloc[train_size:train_size + valid_size]
df_test = df_sample.iloc[train_size + valid_size:]

df_train.to_csv(f"{sample_path}_train.csv", index=False)
df_valid.to_csv(f"{sample_path}_valid.csv", index=False)
df_test.to_csv(f"{sample_path}_test.csv", index=False)

df_train.to_parquet(f"{sample_path}_train.parquet", index=False)
df_valid.to_parquet(f"{sample_path}_valid.parquet", index=False)
df_test.to_parquet(f"{sample_path}_test.parquet", index=False)

print(f"💾 Train/Valid/Test splits saved to '{sample_path}_train.csv', '{sample_path}_valid.csv', '{sample_path}_test.csv'.")

