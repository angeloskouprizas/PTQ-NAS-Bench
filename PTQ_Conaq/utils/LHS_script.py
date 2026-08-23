import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

def get_args():
    parser = argparse.ArgumentParser(description="Latin Hypercube Sampling (LHS) for architecture selection.")
    parser.add_argument(
        '--input_csv', 
        type=str, 
        required=True, 
        help="Path to the input dataset containing the architectures and features."
    )
    parser.add_argument(
        '--output_csv', 
        type=str, 
        required=True, 
        help="Path to save the generated LHS sampled dataset."
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=4000, 
        help="Number of samples to extract (default: 4000 as per the paper)."
    )
    parser.add_argument(
        '--target_features', 
        nargs='+', 
        required=True, 
        help="List of column names to guide the LHS (e.g., 'ln(QNP)xFP32_acc' 'BitOps')."
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

def generate_lhs_samples_numpy(n_samples, n_dims, seed=42):
    """
    Pure Numpy implementation of Latin Hypercube Sampling (LHS).
    Divides each dimension into n_samples intervals and places exactly
    one random point in each interval to avoid overlaps.
    """
    np.random.seed(seed)
    result = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        # Create bins (0, 1, 2, ..., n-1)
        intervals = np.arange(n_samples)
        # Shuffle bins
        np.random.shuffle(intervals)
        # Place a random point inside the bin and normalize to [0, 1]
        result[:, d] = (intervals + np.random.uniform(size=n_samples)) / n_samples
    return result

def create_lhs_dataset(input_csv, output_csv, num_samples, target_features, seed):
    print(f"[*] Loading data from: {input_csv}...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[-] ERROR: Input file not found: {input_csv}")
        
    df = pd.read_csv(input_csv)
    
    if num_samples > len(df):
        raise ValueError(f"[-] ERROR: Requested {num_samples} samples, but dataset only has {len(df)} rows!")

    # 1. Isolate the target columns that will guide the LHS
    try:
        features = df[target_features].values
    except KeyError:
        raise KeyError(f"[-] ERROR: Columns {target_features} not found! Available columns: {list(df.columns)}")

    # 2. Normalize (MinMaxScaler) to [0, 1] so all objectives have equal weight in distance calculations
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    # 3. Generate ideal LHS points in the N-dimensional space
    n_dims = len(target_features)
    print(f"[*] Generating {num_samples} ideal LHS points in a {n_dims}D space based on {target_features}...")
    lhs_ideal_points = generate_lhs_samples_numpy(n_samples=num_samples, n_dims=n_dims, seed=seed)

    # 4. Find the "Nearest Neighbor" in the actual data using Euclidean distance
    print("[*] Matching LHS ideal points to actual architectures (Greedy Matching)...")
    distances = cdist(lhs_ideal_points, features_normalized, metric='euclidean')

    selected_indices = []
    available_indices = set(range(len(df)))  # To avoid picking the same network twice

    for i in range(num_samples):
        closest_idxs = np.argsort(distances[i])
        for idx in closest_idxs:
            if idx in available_indices:
                selected_indices.append(idx)
                available_indices.remove(idx)
                break

    # 5. Create and save the new dataset (copies ALL existing columns)
    sampled_df = df.iloc[selected_indices].copy()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    sampled_df.to_csv(output_csv, index=False)
    print(f"[+] Success! New dataset with {len(sampled_df)} samples saved to: {output_csv}")
    
    return sampled_df

def main():
    args = get_args()
    print("=== LATIN HYPERCUBE SAMPLING (LHS) ===")
    create_lhs_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        num_samples=args.num_samples,
        target_features=args.target_features,
        seed=args.seed
    )

if __name__ == "__main__":
    main()