import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. NAMING & ORDER CONFIGURATION
# ============================================================================
# Mapping Q-Schemes to BackUp folders (Ensure these exactly match your folder names)
PRE_QUANT_FOLDERS = {
    "Q1": "INT8a8w",
    "Q2": "INT4a4w",
    "Q3": "INT8a2w",
    "Q4": "INT2a2w"
}

CUSTOM_SCHEME_NAMES = {
    "Q1": "Q1 (8W8A)",
    "Q2": "Q2 (4W4A)",
    "Q3": "Q3 (2W8A)",
    "Q4": "Q4 (2W2A)"
}

PLOT_ORDER_KEYS = ["Q1", "Q2", "Q3", "Q4"]

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================
def normalize_accuracy(series):
    if series.max() > 1.5:
        return series / 100.0
    return series

def get_clean_id(df, id_col_candidates):
    for col in id_col_candidates:
        if col in df.columns:
            return df[col].astype(float).astype(int)
    raise ValueError(f"[-] ERROR: ID column not found! Candidates: {id_col_candidates}")

def get_acc_col(df):
    for col in df.columns:
        if 'acc' in col.lower() and 'val' not in col.lower():
            return col
    for col in df.columns:
        if 'acc' in col.lower():
            return col
    raise ValueError("[-] ERROR: Accuracy column not found in FP32 dataset!")

def load_fp32_baseline(fp32_csv):
    print(f"[*] Loading FP32 Baseline from {fp32_csv}...")
    df_fp32 = pd.read_csv(fp32_csv)
    df_fp32['model_id'] = get_clean_id(df_fp32, ['model_id', 'arch_index', 'id', 'arch_idx'])
    df_fp32['fp32_acc'] = normalize_accuracy(df_fp32[get_acc_col(df_fp32)])
    return df_fp32[['model_id', 'fp32_acc']]

# ============================================================================
# 3. MAIN ROUTINE
# ============================================================================
def generate_boxplot(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df_fp32 = load_fp32_baseline(args.fp32_csv)
    
    all_data = []
    
    print("[*] Collecting and merging data...")
    
    # 1. Read Q1 Ranked Dataset strictly to calculate Base MFLOPs
    q1_path = os.path.join(args.quant_dir, "final_dataset_Q1_8W8A_QuantAcc_ranked.csv")
    if not os.path.exists(q1_path):
        print(f"[-] ERROR: {q1_path} not found. Required for base complexity calculation.")
        return
        
    df_q1 = pd.read_csv(q1_path)
    df_q1['model_id'] = get_clean_id(df_q1, ['model_id', 'arch_index', 'id'])
    
    df_base_complexity = df_q1[['model_id', 'BitOps']].copy()
    df_base_complexity['Base_MFLOPs'] = (df_base_complexity['BitOps'] / 64) / 1e6
    df_base_complexity = df_base_complexity.drop(columns=['BitOps'])

    # 2. Divide the base into 4 Complexity Tiers
    _, bins = pd.qcut(df_base_complexity['Base_MFLOPs'], q=4, retbins=True, duplicates='drop')
    
    bin_labels = [
        f"Small\n({int(bins[0])}-{int(bins[1])} MFLOPs)", 
        f"Medium-Small\n({int(bins[1])}-{int(bins[2])} MFLOPs)", 
        f"Medium-Large\n({int(bins[2])}-{int(bins[3])} MFLOPs)", 
        f"Large\n({int(bins[3])}-{int(bins[4])} MFLOPs)"
    ]
    df_base_complexity['Complexity_Tier'] = pd.qcut(df_base_complexity['Base_MFLOPs'], q=4, labels=bin_labels, duplicates='drop')

    # 3. Load PRE-FINETUNE results from the BackUp folder
    for scheme_key in PLOT_ORDER_KEYS:
        scheme_name = CUSTOM_SCHEME_NAMES[scheme_key]
        folder_name = PRE_QUANT_FOLDERS[scheme_key]
        scheme_folder = os.path.join(args.backup_dir, folder_name)
        
        # Find ALL CSV buckets (e.g., bucket0, bucket1) in the folder
        csv_files = glob.glob(os.path.join(scheme_folder, "*.csv"))
        
        if not csv_files:
            print(f"[!] Warning: No files found in directory: {scheme_folder}")
            continue
            
        # Merge all buckets of the scheme into a single DataFrame
        df_list = []
        for f in csv_files:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        df_quant = pd.concat(df_list, ignore_index=True)
        
        # Clean ID and convert Accuracy
        df_quant['model_id'] = get_clean_id(df_quant, ['model_id', 'arch_index', 'id'])
        
        if 'top1_accuracy_pre' not in df_quant.columns:
            print(f"[!] WARNING: Column 'top1_accuracy_pre' is missing for scheme {scheme_name}")
            # Note: Script logic continues as originally written
            continue
            
        df_quant['quant_acc_pre'] = normalize_accuracy(df_quant['top1_accuracy'])
        
        # Merge with FP32 and Complexity Tiers
        df_merged = pd.merge(df_fp32, df_quant, on='model_id', how='inner')
        df_merged = pd.merge(df_merged, df_base_complexity[['model_id', 'Complexity_Tier']], on='model_id', how='inner')
        
        # Calculate Pre-Finetune Accuracy Drop (without clipping negative values)
        df_merged['acc_drop_pct'] = (df_merged['fp32_acc'] - df_merged['quant_acc_pre']) * 100.0
        df_merged['Scheme'] = scheme_name  
        
        all_data.append(df_merged[['model_id', 'acc_drop_pct', 'Scheme', 'Complexity_Tier']])

    if not all_data:
        print("[-] ERROR: No data successfully processed. Aborting plot generation.")
        return
        
    df_all = pd.concat(all_data, ignore_index=True)
    
    # ------------------- PLOT (GROUPED BOX PLOT) -------------------
    print("[*] Plotting the Grouped Box Plot...")
    
    sns.set_theme(style="whitegrid") # Whitegrid fits perfectly for boxplots
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    sns.boxplot(
        data=df_all,
        x='Complexity_Tier',
        y='acc_drop_pct',
        hue='Scheme',
        palette='Set2',          # Beautiful color palette
        linewidth=1.2,           # Line width
        fliersize=3.0,           # Outlier size (diamonds)
        flierprops=dict(marker='d', markerfacecolor='gray', alpha=0.5, markeredgecolor='none'),
        ax=ax
    )

    ax.set_title("Post-Finetune Accuracy Drop Distribution by Model Complexity", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Fundamental Network Size (Million FLOPs)", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel("Accuracy Drop (%) [FP32 - Post-Finetune Quant]", fontsize=14, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7) # Red line exactly at 0

    ax.legend(
        title='Quantization \n Schemes', 
        fontsize=9,                 # Smaller font size
        title_fontsize=10,          # Smaller title size
        loc='upper left',           # Back to the classic position
        frameon=True,
        edgecolor='black',
        markerscale=0.7,            # Smaller colored boxes
        labelspacing=0.3,           # Bring lines closer together
        borderpad=0.4               # Reduce white padding around legend
    )
    
    plt.tight_layout()
    
    output_path = os.path.join(args.output_dir, "Post_Finetune_Accuracy_Drop_BoxPlot.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"[+] Success! The Box Plot was saved to:\n    -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Accuracy Drop Distribution Boxplot by Model Complexity.")
    parser.add_argument('--fp32_csv', type=str, required=True, help="Path to the FP32 edgegpu dataset CSV.")
    parser.add_argument('--quant_dir', type=str, required=True, help="Directory containing the ranked Quant datasets (used for base MFLOPs).")
    parser.add_argument('--backup_dir', type=str, required=True, help="Directory containing the Pre-Finetune/PTQ backup buckets.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the resulting boxplot.")
    
    args = parser.parse_args()
    generate_boxplot(args)