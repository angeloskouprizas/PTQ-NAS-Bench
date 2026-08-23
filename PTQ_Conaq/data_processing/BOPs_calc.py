import pandas as pd
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Calculate and integrate BitOps for PTQ-NAS-Bench datasets.")
    parser.add_argument(
        '--fp32_csv', 
        type=str, 
        required=True, 
        help="Path to the FP32 dataset CSV containing raw FLOPs."
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help="Directory containing the PTQ Memory/QNP CSV files."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="Directory to save the updated CSV files with BitOps."
    )
    return parser.parse_args()

def main():
    args = get_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================================
    # 1. Configuration & Paths
    # ============================================================================
    # Bit configurations defined exactly as in the PTQ-NAS-Bench paper
    Q_CONFIGS = {
        "Q1": {"w_bits": 8, "a_bits": 8}, # INT8w8a
        "Q2": {"w_bits": 4, "a_bits": 4}, # INT4w4a
        "Q3": {"w_bits": 2, "a_bits": 8}, # INT2w8a
        "Q4": {"w_bits": 2, "a_bits": 2}  # INT2w2a
    }

    # ============================================================================
    # 2. Load FP32 Dataset (Extracting FLOPs)
    # ============================================================================
    print(f"[*] Loading FP32 Data from: {args.fp32_csv}")
    if not os.path.exists(args.fp32_csv):
        raise FileNotFoundError(f"FP32 CSV not found at {args.fp32_csv}")
        
    df_fp32 = pd.read_csv(args.fp32_csv)[['model_id', 'raw_flops']]
    df_fp32 = df_fp32.rename(columns={'raw_flops': 'FP32_flops'})

    # ============================================================================
    # 3. Integrate BitOps into PTQ Datasets
    # ============================================================================
    for q_name, bits in Q_CONFIGS.items():
        w_bits, a_bits = bits['w_bits'], bits['a_bits']
        print(f"\n[*] Processing [{q_name}] -> W: {w_bits}-bit, A: {a_bits}-bit")
        
        # Define input file path
        input_filename = f"final_dataset_{q_name}_Memory_QNP.csv"
        input_file = os.path.join(args.input_dir, input_filename)
        
        if not os.path.exists(input_file):
            print(f"[!] Warning: Skipping {q_name} - File not found: {input_file}")
            continue
            
        df_base = pd.read_csv(input_file)
        
        # Merge with FP32 FLOPs based on model_id
        df_merged = pd.merge(df_base, df_fp32, on='model_id', how='inner')
        
        # Calculate BitOps: FLOPs * w_bits * a_bits
        df_merged['BitOps'] = df_merged['FP32_flops'] * w_bits * a_bits
        
        # Reorder columns for consistency
        final_columns = ['model_id', 'Memory', 'QNP', 'FP32_flops', 'BitOps']
        df_merged = df_merged[final_columns]
        
        # Save to the output directory
        output_filename = f"final_dataset_{q_name}_Memory_QNP_BitOps.csv"
        output_file = os.path.join(args.output_dir, output_filename)
        df_merged.to_csv(output_file, index=False)
        
        print(f"[+] Success! Saved to: {output_file}")
        print(f"    -> Sample row:\n{df_merged.head(1).to_string(index=False)}")

    print("\n[+] All PTQ scheme datasets updated successfully!")

if __name__ == "__main__":
    main()