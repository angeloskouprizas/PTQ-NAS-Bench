import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# 1. Pareto Ranking Functions
# ============================================================================
def calculate_pareto_ranks_3d(accuracies, qnps, bitops):
    """
    Calculates Pareto Ranks based on 3 objectives:
    - Accuracy (Maximize)
    - ln(QNP+1) (Minimize)
    - BitOps (Minimize)
    """
    n = len(accuracies)
    ranks = np.zeros(n, dtype=int)
    remaining_indices = list(range(n))
    current_rank = 1

    acc = np.array(accuracies)
    qnp = np.array(qnps)
    bop = np.array(bitops)

    pbar = tqdm(total=n, desc="    -> 3D Pareto Sorting Progress", leave=False)

    while remaining_indices:
        current_acc = acc[remaining_indices]
        current_qnp = qnp[remaining_indices]
        current_bop = bop[remaining_indices]

        is_pareto = np.ones(len(remaining_indices), dtype=bool)

        for i in range(len(remaining_indices)):
            # Condition 1: Is there another network that is BETTER OR EQUAL in ALL objectives?
            domination_condition = (
                (current_acc >= current_acc[i])
                & (current_qnp <= current_qnp[i])
                & (current_bop <= current_bop[i])
            )

            # Condition 2: Is that network STRICTLY BETTER in AT LEAST ONE objective?
            strict_condition = (
                (current_acc > current_acc[i])
                | (current_qnp < current_qnp[i])
                | (current_bop < current_bop[i])
            )

            is_dominated = np.any(domination_condition & strict_condition)

            if is_dominated:
                is_pareto[i] = False

        pareto_indices = [remaining_indices[idx] for idx, p in enumerate(is_pareto) if p]

        for idx in pareto_indices:
            ranks[idx] = current_rank

        remaining_indices = [idx for idx in remaining_indices if idx not in pareto_indices]

        pbar.update(len(pareto_indices))
        current_rank += 1

    pbar.close()
    return ranks

def calculate_pareto_ranks_2d(proxies, bitops):
    """
    Calculates Pareto Ranks based on 2 objectives:
    - ln(QNP)xFP32_acc (Minimize)
    - BitOps (Minimize)
    """
    n = len(proxies)
    ranks = np.zeros(n, dtype=int)
    remaining_indices = list(range(n))
    current_rank = 1

    prx = np.array(proxies)
    bop = np.array(bitops)

    pbar = tqdm(total=n, desc="    -> 2D Pareto Sorting Progress", leave=False)

    while remaining_indices:
        current_prx = prx[remaining_indices]
        current_bop = bop[remaining_indices]

        is_pareto = np.ones(len(remaining_indices), dtype=bool)

        for i in range(len(remaining_indices)):
            # Condition 1: Is there another network that is BETTER OR EQUAL in ALL objectives?
            domination_condition = (
                (current_prx <= current_prx[i])
                & (current_bop <= current_bop[i])
            )

            # Condition 2: Is that network STRICTLY BETTER in AT LEAST ONE objective?
            strict_condition = (
                (current_prx < current_prx[i])
                | (current_bop < current_bop[i])
            )

            is_dominated = np.any(domination_condition & strict_condition)

            if is_dominated:
                is_pareto[i] = False

        pareto_indices = [remaining_indices[idx] for idx, p in enumerate(is_pareto) if p]

        for idx in pareto_indices:
            ranks[idx] = current_rank

        remaining_indices = [idx for idx in remaining_indices if idx not in pareto_indices]

        pbar.update(len(pareto_indices))
        current_rank += 1

    pbar.close()
    return ranks

# ============================================================================
# 2. Main Processing Loop
# ============================================================================
def get_args():
    parser = argparse.ArgumentParser(description="Calculate Global Pareto Rankings (2D or 3D).")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the input CSV files.")
    parser.add_argument('--multiply_fp32', action='store_true', help="If set, uses 2D targets: ln(QNP)xFP32_acc & BitOps. Else, uses 3D.")
    return parser.parse_args()

def main():
    args = get_args()
    
    print("\n=== Start Global Pareto Ranking Calculation ===\n")

    # Determine filenames based on the toggle (matching the output of the previous script)
    file_suffix = "_with_LogQNPxFP32.csv" if args.multiply_fp32 else "_with_Log1pQNP.csv"
    
    FILES_CONFIG = [
        f"final_dataset_Q1_8W8A_FP32Acc{file_suffix}",
        f"final_dataset_Q2_4W4A_FP32Acc{file_suffix}",
        f"final_dataset_Q3_2W8A_FP32Acc{file_suffix}",
        f"final_dataset_Q4_2W2A_FP32Acc{file_suffix}"
    ]

    for filename in FILES_CONFIG:
        file_path = os.path.join(args.input_dir, filename)

        print("-" * 65)
        print(f"📁 Processing: {filename}")
        
        if not os.path.exists(file_path):
            print(f"[-] ERROR: File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Set target columns dynamically based on the toggle
        if args.multiply_fp32:
            prx_col = "ln(QNP)xFP32_acc"
            bop_col = "BitOps"
            required_columns = [prx_col, bop_col]
            print(f"🎯 Targets [2D]: Minimize({prx_col}) | Minimize({bop_col})")
        else:
            acc_col = "FP32_acc"
            qnp_col = "ln(QNP+1)"
            bop_col = "BitOps"
            required_columns = [acc_col, qnp_col, bop_col]
            print(f"🎯 Targets [3D]: Maximize({acc_col}) | Minimize({qnp_col}) | Minimize({bop_col})")

        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"[-] ERROR: Missing columns: {missing_cols}. Skipping...")
            continue

        # Check for NaNs that might break Pareto comparisons
        nan_count = df[required_columns].isna().any(axis=1).sum()
        if nan_count > 0:
            print(f"[!] WARNING: Found {nan_count} rows with NaN in target objectives. Dropping them...")
            df = df.dropna(subset=required_columns).reset_index(drop=True)

        # Drop existing global_ranking if script is re-run
        if "global_ranking" in df.columns:
            df = df.drop(columns=["global_ranking"])

        # Calculate Ranks
        if args.multiply_fp32:
            proxies = df[prx_col].values
            bitops = df[bop_col].values
            ranks = calculate_pareto_ranks_2d(proxies, bitops)
        else:
            accuracies = df[acc_col].values
            qnps = df[qnp_col].values
            bitops = df[bop_col].values
            ranks = calculate_pareto_ranks_3d(accuracies, qnps, bitops)

        # Append new column & Sort
        df["global_ranking"] = ranks
        df = df.sort_values(by="global_ranking").reset_index(drop=True)

        # Save output (Rename to indicate it's ranked)
        out_filename = filename.replace(".csv", "_ranked.csv")
        out_path = os.path.join(args.input_dir, out_filename)
        df.to_csv(out_path, index=False)
        
        print(f"[+] SUCCESS: Ranked file saved as {out_filename} (Top Rank: 1, Worst Rank: {ranks.max()}).")

        # Preview Top 3 Models
        preview_cols = ["model_id"] + required_columns + ["global_ranking"]
        if "Memory" in df.columns:
            preview_cols.insert(1, "Memory")

        print("\n   🏆 Top 3 Models (Rank 1):")
        top_models = df[df["global_ranking"] == 1][preview_cols].head(3)
        print(top_models.to_string(index=False))
        print()

    print("-" * 65)
    print("🎉 All files have been successfully updated with 'global_ranking'!")

if __name__ == "__main__":
    main()