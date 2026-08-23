import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# 1. Settings & Configurations
# ============================================================================
SCHEMES_CONFIG = [
    {
        "filename": "final_dataset_Q1_8W8A_QuantAcc.csv",
        "title": "Q1 Scheme (8W - 8A)",
        "color": "#1f77b4",
    },
    {
        "filename": "final_dataset_Q2_4W4A_QuantAcc.csv",
        "title": "Q2 Scheme (4W - 4A)",
        "color": "#ff7f0e",
    },
    {
        "filename": "final_dataset_Q3_2W8A_QuantAcc.csv",
        "title": "Q3 Scheme (2W - 8A)",
        "color": "#2ca02c",
    },
    {
        "filename": "final_dataset_Q4_2W2A_QuantAcc.csv",
        "title": "Q4 Scheme (2W - 2A)",
        "color": "#d62728",
    },
]

QNP_COL = "QNP"


# ============================================================================
# 2. Helper Function to Detect Accuracy Column
# ============================================================================
def detect_accuracy_column(df):
    common_names = [
        "INT8_acc",
        "INT4_acc",
        "INT2_acc",
        "QuantAcc",
        "accuracy",
        "val_acc",
        "top1",
    ]
    for col in common_names:
        if col in df.columns:
            return col

    for col in df.columns:
        if "acc" in col.lower() and "fp32" not in col.lower():
            return col

    return None


# ============================================================================
# 3. Main Plotting Function
# ============================================================================
def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    fig_grid, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.flatten()

    print("=== Start Analysis: Quantized Accuracy (X) vs QNP (Log Scale Y) ===\n")

    for idx, config in enumerate(SCHEMES_CONFIG):
        file_path = os.path.join(input_dir, config["filename"])
        title = config["title"]
        color = config["color"]
        ax = axes_flat[idx]

        print(f"--> Processing: {title}")

        if not os.path.exists(file_path):
            print(f"    [!] File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        acc_col = detect_accuracy_column(df)

        if not acc_col or QNP_COL not in df.columns:
            print(f"    [!] Column error. Available columns: {list(df.columns)}")
            continue

        # Clean NaNs
        df_clean = df[[acc_col, QNP_COL]].dropna()

        # Define: X = Accuracy, Y = QNP (Raw data, no modifications)
        x = df_clean[acc_col]
        y = df_clean[QNP_COL]

        # Calculate classical correlations on the original data
        pearson_r = x.corr(y, method="pearson")
        spearman_rho = x.corr(y, method="spearman")

        print(f"    [*] X: '{acc_col}' | Y: '{QNP_COL}' | Models: {len(x)}")
        print(f"    [*] Pearson r: {pearson_r:.3f} | Spearman rho: {spearman_rho:.3f}")

        # --- Dynamic Y Limits Calculation (QNP Range) ---
        # Find positive values to properly set log scale limits
        pos_qnp = y[y > 0]
        if not pos_qnp.empty:
            min_qnp = pos_qnp.min()
            max_qnp = y.max()
            print(f"    [*] Dynamic QNP limits -> Min: {min_qnp:.2e} | Max: {max_qnp:.2e}\n")
        else:
            min_qnp, max_qnp = None, None

        # --- 1. Plot on the 2x2 Grid ---
        sns.regplot(
            data=df_clean,
            x=acc_col,
            y=QNP_COL,
            ax=ax,
            color=color,
            scatter_kws={"alpha": 0.5, "s": 25},
            line_kws={"color": "black", "linewidth": 1.5, "linestyle": "--"},
        )

        # Change axis to Logarithmic Scale and apply Dynamic Limits
        ax.set_yscale("log")
        if min_qnp and max_qnp:
            # Visual margin: /5 at min and *5 at max for clean representation
            ax.set_ylim(bottom=min_qnp / 5, top=max_qnp * 5)

        ax.set_title(
            f"{title}\nPearson r = {pearson_r:.2f} | Spearman ρ = {spearman_rho:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel(f"Quantized Accuracy ({acc_col})", fontsize=10)
        ax.set_ylabel("QNP (Log Scale)", fontsize=10)

        # --- 2. Create and Save Independent Plot ---
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df_clean,
            x=acc_col,
            y=QNP_COL,
            color=color,
            scatter_kws={"alpha": 0.6, "s": 35},
            line_kws={"color": "black", "linewidth": 2, "linestyle": "--"},
        )

        plt.yscale("log")
        if min_qnp and max_qnp:
            plt.ylim(bottom=min_qnp / 5, top=max_qnp * 5)

        plt.title(
            f"Quantized Accuracy vs QNP\n{title}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel(f"Quantized Accuracy ({acc_col})", fontsize=12)
        plt.ylabel("QNP - Quantization Noise Power (Log Scale)", fontsize=12)

        stats_text = (
            f"Models: {len(x)}\nPearson r: {pearson_r:.2f}\nSpearman ρ: {spearman_rho:.2f}"
        )
        plt.gca().text(
            0.05,
            0.15,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        single_plot_path = os.path.join(
            output_dir, f"Accuracy_vs_QNP_{config['filename'].replace('.csv', '')}.png"
        )
        plt.savefig(single_plot_path, dpi=300)
        plt.close()

    # --- Save 2x2 Grid ---
    fig_grid.tight_layout()
    grid_plot_path = os.path.join(output_dir, "Combined_Accuracy_vs_QNP_Grid.png")
    fig_grid.savefig(grid_plot_path, dpi=300)
    plt.close(fig_grid)

    print("-" * 60)
    print(f"[+] Success! The new plots have been saved to:\n    {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Quantized Accuracy vs QNP.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the QuantAcc CSV files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated plots.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)