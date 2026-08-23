import os
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Merge LHS sampled datasets with encoded FP32 architectures for MLP training.")
    parser.add_argument(
        '--lhs_csv', 
        type=str, 
        required=True, 
        help="Path to the LHS sampled CSV file containing the ranks."
    )
    parser.add_argument(
        '--encoded_csv', 
        type=str, 
        required=True, 
        help="Path to the dataset containing the FP32 encoded architectures."
    )
    parser.add_argument(
        '--output_csv', 
        type=str, 
        required=True, 
        help="Path where the final merged dataset will be saved."
    )
    return parser.parse_args()

def main():
    args = get_args()

    print("[*] STARTING ENCODED MLP DATASET PREPARATION\n")

    # ============================================================================
    # 1. File Existence Checks
    # ============================================================================
    if not os.path.exists(args.lhs_csv):
        print(f"[-] ERROR: LHS dataset not found at {args.lhs_csv}")
        return
    if not os.path.exists(args.encoded_csv):
        print(f"[-] ERROR: Encoded FP32 dataset not found at {args.encoded_csv}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # ============================================================================
    # 2. Load LHS Dataset
    # ============================================================================
    print(f"[*] Loading LHS Dataset from: {args.lhs_csv}...")
    df_lhs = pd.read_csv(args.lhs_csv)

    # Identify the rank column dynamically ('global_ranking' or 'rank')
    rank_col = "global_ranking" if "global_ranking" in df_lhs.columns else ("rank" if "rank" in df_lhs.columns else None)

    if not rank_col:
        raise KeyError(f"[-] ERROR: No rank column found in LHS file. Available columns: {list(df_lhs.columns)}")

    # Keep only the model_id and rank from the LHS dataset
    df_lhs_clean = df_lhs[["model_id", rank_col]].copy()

    # ============================================================================
    # 3. Load Encoded FP32 Dataset
    # ============================================================================
    print(f"[*] Loading Encoded Architectures from: {args.encoded_csv}...")
    # Load ONLY the necessary columns: 'arch_index' and 'concatenated'
    df_encoded = pd.read_csv(args.encoded_csv, usecols=["arch_index", "concatenated"])

    # ============================================================================
    # 4. Merge Datasets (Inner Merge)
    # ============================================================================
    print("[*] Matching architectures (Inner Merge)...")
    df_merged = pd.merge(
        df_lhs_clean,
        df_encoded,
        left_on="model_id",
        right_on="arch_index",
        how="inner",
    )

    if len(df_merged) == 0:
        print("[!] WARNING: The merge resulted in 0 rows! Check if model_id/arch_index match between the two files.")
        return

    # ============================================================================
    # 5. Rename Columns & Finalize Structure
    # ============================================================================
    df_merged = df_merged.rename(columns={"concatenated": "encoded_arch", rank_col: "rank"})

    # Strictly keep the structure: model_id, encoded_arch, rank
    # (arch_index is discarded as it is identical to model_id)
    final_df = df_merged[["model_id", "encoded_arch", "rank"]]

    # Sort by rank for aesthetic and structural consistency
    final_df = final_df.sort_values(by="rank").reset_index(drop=True)

    # ============================================================================
    # 6. Save Final Dataset
    # ============================================================================
    final_df.to_csv(args.output_csv, index=False)
    
    print(f"\n[+] Success! The new dataset has been saved to:\n    -> {args.output_csv}")
    print(f"[+] Total matched architectures: {len(final_df)} / {len(df_lhs)}")
    print("\n[Preview of first 3 rows]")
    print(final_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()