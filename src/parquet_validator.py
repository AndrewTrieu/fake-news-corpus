# Validate if a parquet file is valid or not, and print out some information about the file.
import pyarrow.parquet as pq

def validate_parquet_file(file_path):
    parquet_file = None
    try:
        parquet_file = pq.ParquetFile(file_path)
        print(f"✅ The file '{file_path}' is a valid Parquet file.")
    except Exception as e:
        print(f"❌ The file '{file_path}' is not a valid Parquet file.")
        print(f"Error: {e}")
    
    print(f"  - Column Names: {parquet_file.schema}")
    print(f"  - File Metadata: {parquet_file.metadata}")

# Example usage:
validate_parquet_file("../data/processed_fakenews.parquet")
validate_parquet_file("../data/sampled_fakenews.parquet")

