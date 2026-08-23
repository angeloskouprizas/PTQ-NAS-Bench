import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1. SCHEME CONFIGURATION & UNIQUE PALETTES
# ============================================================================
# Each Scheme has its own distinct palette (Background color & Algorithm color)
SCHEMES_CONFIG = {
    "Q1 (8W8A)": {
        "file": "final_dataset_Q1_8W8A_QuantAcc_ranked.csv", 
        "bg_color": "#aec7e8", "alg_color": "#1f77b4", "label": "INT8w8a"
    },
    "Q2 (4W4A)": {
        "file": "final_dataset_Q2_4W4A_QuantAcc_ranked.csv", 
        "bg_color": "#98ed98", "alg_color": "#2ca02c", "label": "INT4w4a"
    },
    "Q3 (2W8A)": {
        "file": "final_dataset_Q3_2W8A_QuantAcc_ranked.csv", 
        "bg_color": "#ffbb78", "alg_color": "#ff7f0e", "label": "INT2w8a"
    },
    "Q4 (2W2A)": {
        "file": "final_dataset_Q4_2W2A_QuantAcc_ranked.csv", 
        "bg_color": "#c5b0d5", "alg_color": "#9467bd", "label": "INT2w2a"
    }
}

def plot_all_50runs_noline(data_dir, runs_csv, output_figure):
    print("[*] Generating plot with all 50-run points (without the black line)...")
    
    if not os.path.exists(runs_csv):
        print(f"[-] ERROR: File not found: {runs_csv}")
        return

    # ------------------------------------------------------------------------
    # A. LOAD BACKGROUND SEARCH SPACE
    # ------------------------------------------------------------------------
    all_dfs = []
    for scheme_name, config in SCHEMES_CONFIG.items():
        filepath = os.path.join(data_dir, config["file"])
        if not os.path.exists(filepath): 
            print(f"[!] Warning: Missing background data for {scheme_name}")
            continue
            
        df = pd.read_csv(filepath)
        acc_col = next((col for col in df.columns if "acc" in col.lower() and "fp32" not in col.lower()), None)
        
        if df[acc_col].dtype == object:
            df[acc_col] = df[acc_col].astype(str).str.replace('tensor(', '').str.replace(')', '').astype(float)
        if df[acc_col].max() <= 1.5:
            df[acc_col] = df[acc_col] * 100.0
            
        df_sub = pd.DataFrame({
            "BitOps": df["BitOps"],
            "Accuracy": df[acc_col],
            "Scheme": scheme_name,
            "Label": config["label"],
            "BgColor": config["bg_color"]
        })
        all_dfs.append(df_sub)
        
    if not all_dfs:
        print("[-] ERROR: No background datasets found. Aborting.")
        return
        
    df_global = pd.concat(all_dfs, ignore_index=True)

    # ------------------------------------------------------------------------
    # B. LOAD ALL DATA FROM THE 50 RUNS
    # ------------------------------------------------------------------------
    df_runs = pd.read_csv(runs_csv)
    
    search_records = []
    for _, row in df_runs.iterrows():
        scheme = row["Scheme"]
        run_id = row["Run_ID"]
        
        rs_acc = row["RS_Best_Acc"] * 100.0 if row["RS_Best_Acc"] <= 1.5 else row["RS_Best_Acc"]
        moea_acc = row["MOEA_Best_Acc"] * 100.0 if row["MOEA_Best_Acc"] <= 1.5 else row["MOEA_Best_Acc"]
        
        # RS Point
        search_records.append({
            "Scheme": scheme, "Algorithm": "Random Search", 
            "BitOps": row["RS_Best_BitOps"], "Accuracy": rs_acc, "Run_ID": run_id
        })
        # MOEA Point
        search_records.append({
            "Scheme": scheme, "Algorithm": "MOEA (NSGA-II)", 
            "BitOps": row["MOEA_Best_BitOps"], "Accuracy": moea_acc, "Run_ID": run_id
        })
        
    df_search_all = pd.DataFrame(search_records)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # 1. Plot Background Search Space with each scheme's palette color
    for scheme_name, config in SCHEMES_CONFIG.items():
        sub_data = df_global[df_global["Scheme"] == scheme_name]
        ax.scatter(sub_data["BitOps"], sub_data["Accuracy"], color=config["bg_color"], alpha=0.2, s=15, zorder=1)

    # 2. Plot ALL points from the 50 runs for RS and MOEA (No Pareto line)
    for scheme_name, config in SCHEMES_CONFIG.items():
        s_data = df_search_all[df_search_all["Scheme"] == scheme_name]
        color = config["alg_color"]
        
        rs_data = s_data[s_data["Algorithm"] == "Random Search"]
        moea_data = s_data[s_data["Algorithm"] == "MOEA (NSGA-II)"]
        
        # Random Search -> Crosses (X) with the scheme's color
        ax.scatter(rs_data["BitOps"], rs_data["Accuracy"], color=color, marker="X", s=50, edgecolor="black", linewidth=0.5, zorder=5, alpha=0.7)
        # MOEA -> Stars (*) with the scheme's color
        ax.scatter(moea_data["BitOps"], moea_data["Accuracy"], color=color, marker="*", s=120, edgecolor="black", linewidth=0.6, zorder=6, alpha=0.9)

    # ------------------------------------------------------------------------
    # C. FORMATTING & LEGEND
    # ------------------------------------------------------------------------
    ax.set_title("Hardware-Aware NAS: All 50-Run Solutions Across Quantization Schemes", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Quantized GBitOps (log) / BitOps", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy after fine-tune (%)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='INT8w8a', markerfacecolor='#1f77b4', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='INT4w4a', markerfacecolor='#2ca02c', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='INT2w8a', markerfacecolor='#ff7f0e', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='INT2w2a', markerfacecolor='#9467bd', markersize=9),
        Line2D([0], [0], marker='*', color='w', label='MOEA (NSGA-II - All Runs)', markerfacecolor='gray', markeredgecolor='black', markersize=14),
        Line2D([0], [0], marker='X', color='w', label='Random Search (All Runs)', markerfacecolor='gray', markeredgecolor='black', markersize=10),
    ]
    
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    plt.savefig(output_figure, dpi=300)
    plt.close()
    
    print(f"\n[+] Success! The plot (without line) has been saved to:\n    -> {output_figure}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Unified Pareto Space scatter plot (RS vs MOEA).")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the ranked quantization datasets.")
    parser.add_argument('--runs_csv', type=str, required=True, help="Path to the CSV containing the 50-run search results.")
    parser.add_argument('--output_figure', type=str, required=True, help="Path to save the generated unified pareto plot (e.g., plot.png).")
    
    args = parser.parse_args()
    
    plot_all_50runs_noline(args.data_dir, args.runs_csv, args.output_figure)