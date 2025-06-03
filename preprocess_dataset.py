import pandas as pd
import os
import argparse

def preprocess_dataset(input_file, output_file=None, n_rows=10000):
    """
    Preprocess a CSV dataset by cutting it to specified number of rows.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, will append '_preprocessed' to input filename
        n_rows (int): Number of rows to keep (default: 10000)
    """
    # Set default output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_preprocessed{ext}"
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    # Get total rows
    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows}")
    
    # Cut to specified number of rows
    if n_rows > 0 and total_rows > n_rows:
        print(f"Cutting dataset to {n_rows} rows...")
        df = df.head(n_rows)
    
    # Save preprocessed dataset
    print(f"Saving preprocessed dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Done! Preprocessed dataset has {len(df)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess a CSV dataset by cutting it to specified number of rows.')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output', '-o', help='Path to output CSV file (optional)')
    parser.add_argument('--rows', '-r', type=int, default=10000, help='Number of rows to keep (default: 10000)')
    
    args = parser.parse_args()
    
    preprocess_dataset(args.input_file, args.output, args.rows) 