import os
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Merge top-1 accuracy data with QNP and BitOps datasets.")
    parser.add_argument(
        '--acc_dir', 
        type=str, 
        required=True, 
        help="Base directory containing the accuracy result folders (e.g., INT8a8w)."
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help="Directory containing the intermediate _Memory_QNP_BitOps.csv files."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="Directory to save the final COMPLETE datasets."
    )
    return parser.parse_args()

def main():
    args = get_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================================
    # 1. Configuration & Paths
    # ============================================================================
    # Re-aligned to strictly match the PTQ-NAS-Bench paper specifications
    Q_CONFIGS = {
        "Q1": {"w_bits": 8, "a_bits": 8, "acc_prefix": "INT8", "acc_folder": "INT8a8w"},
        "Q2": {"w_bits": 4, "a_bits": 4, "acc_prefix": "INT4", "acc_folder": "INT4a4w"},
        "Q3": {"w_bits": 2, "a_bits": 8, "acc_prefix": "INT8a2w", "acc_folder": "INT8a2w"},
        "Q4": {"w_bits": 2, "a_bits": 2, "acc_prefix": "INT2", "acc_folder": "INT2a2w"} 
    }

    # ============================================================================
    # 2. Merge Accuracy into Existing Datasets
    # ============================================================================
    for q_name, config in Q_CONFIGS.items():
        print(f"\n[*] Processing [{q_name}] - Extracting Accuracy from folder: {config['acc_folder']}...")
        
        # 1. Path to the file created in the previous step
        base_filename = f"final_dataset_{q_name}_Memory_QNP_BitOps.csv"
        base_file = os.path.join(args.input_dir, base_filename)
        
        if not os.path.exists(base_file):
            print(f"[!] Warning: Skipping {q_name}. Base file not found: {base_file}")
            continue
            
        df_base = pd.read_csv(base_file)
        
        # 2. On-the-fly merge of accuracy files (bucket0, bucket1, bucket2)
        acc_dfs = []
        prefix = config["acc_prefix"]
        folder = config["acc_folder"]
        
        for bucket in range(3):
            acc_filename = f"{prefix}_bucket{bucket}_server_results.csv"
            acc_file = os.path.join(args.acc_dir, folder, acc_filename)
            
            if os.path.exists(acc_file):
                # Read only necessary columns
                df_bucket = pd.read_csv(acc_file)[['model_id', 'top1_accuracy']]
                acc_dfs.append(df_bucket)
                print(f"    -> Loaded bucket {bucket}")
            else:
                print(f"    [!] Missing file: {acc_file}")
                
        # If accuracy files were found, merge them
        if acc_dfs:
            # Concatenate all buckets
            df_acc_merged = pd.concat(acc_dfs, ignore_index=True)
            
            # Rename 'top1_accuracy' column dynamically based on prefix
            acc_col_name = f"{prefix}_acc"
            df_acc_merged = df_acc_merged.rename(columns={'top1_accuracy': acc_col_name})
            
            # 3. Final merge with the base dataset
            df_final = pd.merge(df_base, df_acc_merged, on='model_id', how='inner')
            
            # 4. Save the fully complete dataset
            output_filename = f"final_dataset_{q_name}_COMPLETE.csv"
            output_file = os.path.join(args.output_dir, output_filename)
            
            # Ensure the desired column order
            final_columns = ['model_id', 'Memory', 'QNP', 'FP32_flops', 'BitOps', acc_col_name]
            df_final = df_final[final_columns]
            
            df_final.to_csv(output_file, index=False)
            
            print(f"[+] Success! Saved to: {output_file}")
            print(f"    -> Columns: {list(df_final.columns)}")
            print(f"    -> Total rows (models): {len(df_final)}")
        else:
            print(f"[-] No accuracy files found for {q_name}.")

    print("\n[+] All complete datasets generated successfully!")

if __name__ == "__main__":
    main()