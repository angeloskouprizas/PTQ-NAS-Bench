import torch
import os
import sys
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate FE Cache for NAS-Bench-201 architectures")
    parser.add_argument('--naszilla_dir', type=str, required=True, help="Path to the naszilla repository")
    parser.add_argument('--project_root', type=str, required=True, help="Path to the HW-PR-NAS project root")
    parser.add_argument('--dataset_name', type=str, default='cifar10', help="Dataset name (e.g., cifar10)")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the generated nas201_fe_cache.pt")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dynamically append paths
    sys.path.append(args.naszilla_dir)
    sys.path.append(args.project_root)

    # Imports must happen after sys.path is updated
    from naszilla.nas_benchmarks import Nasbench201
    from nas_benchmark.HW_NAS_Bench.hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net
    from Pareto_Rank_Predictor.MLP_Architecture.Encoding.fe import FE 

    device = 'cpu' # FE runs on CPU, no GPU required
    
    print(f"[*] Loading NAS-Bench-201 API for dataset: {args.dataset_name}...")
    nasbench = Nasbench201(dataset=args.dataset_name)
    api = nasbench.nasbench
    
    total_archs = len(api) # 15625
    
    # Tensor to store the 8 features for each index
    # Shape: (15625, 8)
    fe_cache_tensor = torch.zeros((total_archs, 8))
    
    print(f"[*] Starting feature extraction for {total_archs} architectures...")
    
    # Using tqdm for progress visualization
    for i in tqdm(range(total_archs)):
        try:
            # 1. Get config and model from the API
            config = api.get_net_config(i, args.dataset_name)
            model = get_cell_based_tiny_net(config)
            
            # 2. Use the FE module
            fe_extractor = FE(model)
            features = fe_extractor.extract_features(input_size=(1, 3, 32, 32))
            
            # 3. Save to tensor
            fe_cache_tensor[i] = features
            
        except Exception as e:
            print(f"[-] Error at index {i}: {e}")
            continue

    # 4. Save the final Tensor
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    torch.save(fe_cache_tensor, args.output_file)
    print(f"\n[+] Extraction completed! File saved to: {args.output_file}")

if __name__ == "__main__":
    main()
