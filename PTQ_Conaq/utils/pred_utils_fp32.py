import numpy as np
import sys
# --- Imports (Προσάρμοσε τα paths αν χρειάζεται) ---
sys.path.append('/home/lefterismyl/Downloads/HW_aware_NAS/HW-PR-NAS/naszilla')
sys.path.append('/home/lefterismyl/Downloads/HW_aware_NAS/HW-PR-NAS')

from naszilla.nas_bench_201.cell_201 import Cell201

from naszilla.nas_benchmarks import Nasbench201

def get_index_from_string(arch_str):
    nb201_api = Nasbench201(dataset='cifar10')
    """Convert architecture string to index using OFFICIAL API (matching Prepare_dataf.py)."""
    return nb201_api.nasbench.query_index_by_arch(arch_str)

def get_metrics(device,arch_str=None):

    csv_path = f'/home/lefterismyl/Downloads/HW_aware_NAS/HW-PR-NAS/Pareto_Rank_Predictor/MLP_Architecture/Dataset/FP32/edgegpu_dataset_with_flops.csv'
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    arch_index = data[:, 0].astype(int)  # Assuming architecture index is in the first column
    accuracies = data[:, 1]  # Assuming accuracy is in the second column
    raw_flops = data[:, 2]   # Assuming raw FLOPs is in the third column
    global_ranking = data[:, 3]        # Assuming rank is in the fourth column
    return arch_index, accuracies, raw_flops, global_ranking

if __name__ == "__main__":
    device = 'edgegpu'  # Example device
    arch_index, accuracies, raw_flops, true_ranks = get_metrics(device)
    num_archs = 5
    
    for i in range(num_archs):
        print(f"Arch Index: {arch_index[i]}, Accuracy: {accuracies[i]}, Raw FLOPs: {raw_flops[i]}, True Rank: {true_ranks[i]}")