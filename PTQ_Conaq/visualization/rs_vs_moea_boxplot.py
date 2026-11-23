import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. VISUALIZATION FUNCTION
# ============================================================================
def plot_rs_vs_moea(input_csv, output_figure):
    print(f"[*] Reading data from:\n    -> {input_csv}")
    if not os.path.exists(input_csv):
        print("[-] ERROR: The CSV file was not found. Ensure the search algorithm script has completed successfully!")
        return

    df = pd.read_csv(input_csv)

    # ============================================================================
    # 2. CREATE BOXPLOT
    # ============================================================================
    print("\n[*] Generating Boxplot (RS vs MOEA)...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Define order and colors
    hue_order = ["Random Search", "MOEA"]
    palette = {
        "Random Search": "#1f77b4",    # Blue (baseline)
        "MOEA": "#2ca02c"    # Green (evolutionary algorithm)
    }

    sns.boxplot(
        data=df,
        x="Scheme",
        y="Global Actual Rank",
        hue="Algorithm",
        hue_order=hue_order,
        palette=palette,
        width=0.6,
        fliersize=3,
        ax=ax
    )

    # Horizontal line (Threshold)
    ax.axhline(50, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Rank = 50 Threshold (Elite)")

    ax.set_title("Search Algorithm Comparison: Random Search vs MOEA", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Quantization Scheme", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Global Actual Rank (Lower is Better)", fontsize=12, fontweight="bold", labelpad=10)
    
    # Move Legend to the top left
    ax.legend(title="Algorithm", loc="upper left", frameon=True, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    plt.savefig(output_figure, dpi=300)
    plt.close()
    
    print(f"[+] Success! The plot has been saved to:\n    -> {output_figure}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Boxplot comparing RS and MOEA search strategies.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file containing search results.")
    parser.add_argument('--output_figure', type=str, required=True, help="Path to save the generated boxplot image (e.g., plot.png).")
    
    args = parser.parse_args()
    
    plot_rs_vs_moea(args.input_csv, args.output_figure)
