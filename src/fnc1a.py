import random
import pandas as pd
import os
import subprocess

data_path = "./FNC/news_cleaned_2018_02_13.csv"
sample_path = "sampled_news"
SAMPLE_FRACTION = 0.001  # Use 0.001 for 0.1% of the dataset

if not os.path.exists(data_path):
    print(f"❌ Error: File not found at {data_path}")
    exit()

# Get total rows. Only works on Unix-like systems due to `wc` command
total_rows = int(subprocess.check_output(["wc", "-l", data_path]).split()[0]) - 1
print(f"🔍 Dataset contains {total_rows:,} rows.")

sample_size = int(total_rows * SAMPLE_FRACTION)
print(f"📉 Reducing dataset to {sample_size:,} rows...")

# Read only a sample
skip_rows = sorted(random.sample(range(1, total_rows + 1), total_rows - sample_size))
df_sample = pd.read_csv(data_path, skiprows=skip_rows, lineterminator="\n", on_bad_lines="skip")
df_sample.to_csv(f"{sample_path}.csv", index=False)
df_sample.to_parquet(f"{sample_path}.parquet", index=False)

print("✅ Sample saved to sampled_news.csv and sampled_news.parquet.")
