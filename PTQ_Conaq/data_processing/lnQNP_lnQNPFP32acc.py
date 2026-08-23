import os
import argparse
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Calculate proxy features based on QNP and FP32 accuracy.")
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help="Directory containing the merged PTQ datasets (with FP32_acc)."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="Directory to save the datasets with the new proxy features."
    )
    parser.add_argument(
        '--multiply_fp32', 
        action='store_true', 
        help="If set, calculates ln(QNP) x FP32_acc. Otherwise, calculates ln(QNP+1)."
    )
    return parser.parse_args()

def main():
    args = get_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================================
    # 1. Configuration & Target Files
    # ============================================================================
    # Target files for all Q-Schemes
    target_files = [
        "final_dataset_Q1_8W8A_FP32Acc.csv",
        "final_dataset_Q2_4W4A_FP32Acc.csv",
        "final_dataset_Q3_2W8A_FP32Acc.csv",
        "final_dataset_Q4_2W2A_FP32Acc.csv"
    ]

    print("[*] STARTING PROXY FEATURE CALCULATION\n")
    
    # ============================================================================
    # 2. Main Processing Loop
    # ============================================================================
    for filename in target_files:
        input_csv = os.path.join(args.input_dir, filename)
        
        if not os.path.exists(input_csv):
            print(f"[!] Warning: Skipping {filename} -> File not found.")
            continue
            
        print(f"[*] Processing: {filename}")
        df = pd.read_csv(input_csv)
        
        # Ensure 'model_id' exists (fix 'arch_index' if necessary)
        if "arch_index" in df.columns and "model_id" not in df.columns:
            df = df.rename(columns={"arch_index": "model_id"})
            
        # Check for required columns
        required_cols = ["model_id", "QNP", "BitOps", "FP32_acc"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"    [-] Error: Missing columns {missing_cols}. Skipping file.\n")
            continue

        # ============================================================================
        # 3. Calculate Proxy Feature based on Toggle Flag
        # ============================================================================
        if args.multiply_fp32:
            new_col_name = "ln(QNP)xFP32_acc"
            # Using + 1e-12 to prevent log(0) which results in -inf
            df[new_col_name] = np.log(df["QNP"] + 1e-12) * df["FP32_acc"]
            out_suffix = "_with_LogQNPxFP32.csv"
            print(f"    [+] Mode: Multiplied -> Created '{new_col_name}'")
        else:
            new_col_name = "ln(QNP+1)"
            # np.log1p computes ln(1 + x) accurately even for very small x
            df[new_col_name] = np.log1p(df["QNP"])
            out_suffix = "_with_Log1pQNP.csv"
            print(f"    [+] Mode: Standard -> Created '{new_col_name}'")
            
        # Reorder columns: place the new feature immediately after 'FP32_acc'
        cols = list(df.columns)
        cols.insert(cols.index("FP32_acc") + 1, cols.pop(cols.index(new_col_name)))
        df = df[cols]
        
        # Save to the output directory
        output_filename = filename.replace(".csv", out_suffix)
        output_csv = os.path.join(args.output_dir, output_filename)
        
        df.to_csv(output_csv, index=False)
        print(f"    [+] Saved to: {output_filename}\n")
        
    print("-" * 65)
    print("[+] Process completed successfully!")

if __name__ == "__main__":
    main()