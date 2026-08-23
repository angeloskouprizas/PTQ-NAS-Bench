import os
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Merge FP32 accuracy data into the PTQ datasets.")
    parser.add_argument(
        '--fp32_csv', 
        type=str, 
        required=True, 
        help="Path to the FP32 dataset CSV containing 'true_accuracy'."
    )
    parser.add_argument(
        '--target_dir', 
        type=str, 
        required=True, 
        help="Directory containing the target PTQ datasets to be updated."
    )
    return parser.parse_args()

def main():
    args = get_args()

    # ============================================================================
    # 1. Paths and Target Files
    # ============================================================================
    # The exact list of files we want to update, aligned with the paper's schemes
    target_files = [
        "final_dataset_Q1_8W8A_FP32Acc.csv",
        "final_dataset_Q2_4W4A_FP32Acc.csv",
        "final_dataset_Q3_2W8A_FP32Acc.csv",
        "final_dataset_Q4_2W2A_FP32Acc.csv",
    ]

    # Check if the source file exists
    if not os.path.exists(args.fp32_csv):
        print(f"[-] ERROR: Source file not found:\n{args.fp32_csv}")
        return

    print(f"[*] Loading source FP32 data from: {args.fp32_csv}")
    df_source = pd.read_csv(args.fp32_csv)

    # Validate required columns
    required_cols = {"model_id", "true_accuracy"}
    if not required_cols.issubset(df_source.columns):
        raise ValueError(f"Source file must contain the following columns: {required_cols}")

    # ============================================================================
    # 2. Create the mapping dataframe (avoiding duplicate IDs)
    # ============================================================================
    mapping_df = df_source[["model_id", "true_accuracy"]].drop_duplicates(subset=["model_id"])
    
    # Rename 'true_accuracy' to 'FP32_acc'
    mapping_df = mapping_df.rename(columns={"true_accuracy": "FP32_acc"})

    print(f"[*] Found {len(mapping_df)} unique 'model_id' entries in the source file.\n")

    # ============================================================================
    # 3. Iterate over the target PTQ files
    # ============================================================================
    for filename in target_files:
        target_file = os.path.join(args.target_dir, filename)

        print("-" * 60)
        print(f"[*] Processing: {filename}")

        if not os.path.exists(target_file):
            print(f"    [!] File not found in target directory, skipping...")
            continue

        # Read the target dataset
        df_target = pd.read_csv(target_file)

        # If the script was run previously, drop the 'FP32_acc' column for a clean re-merge
        if "FP32_acc" in df_target.columns:
            print("    [*] 'FP32_acc' column already exists. Replacing...")
            df_target = df_target.drop(columns=["FP32_acc"])

        # ============================================================================
        # 4. Merge (Left Join) based on model_id
        # ============================================================================
        df_merged = pd.merge(df_target, mapping_df, on="model_id", how="left")

        # Check for any missing values after the merge
        missing_count = df_merged["FP32_acc"].isna().sum()
        if missing_count > 0:
            print(f"    [!] WARNING: Found {missing_count} rows with missing FP32_acc (NaN).")
        else:
            print(f"    [+] All {len(df_merged)} model_ids matched successfully!")

        # ============================================================================
        # 5. Save the updated file
        # ============================================================================
        df_merged.to_csv(target_file, index=False)
        print(f"    [+] File saved successfully.")

    print("-" * 60)
    print("[+] Process completed successfully!")


if __name__ == "__main__":
    main()