import sys
import os
import time
import torch
import numpy as np
import csv
import json
import warnings
import pandas as pd
import argparse

# ============================================================================
# DATA MANAGEMENT & HELPER FUNCTIONS
# ============================================================================
def load_embeddings_from_fp32(csv_file):
    print(f"[*] Loading Embeddings from: {os.path.basename(csv_file)}...")
    embeddings_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                arch_idx = int(row['arch_index'])
                emb_list = json.loads(row['concatenated'])
                if len(emb_list) == 104:
                    embeddings_dict[arch_idx] = torch.tensor(emb_list, dtype=torch.float32)
            except Exception:
                continue
    return embeddings_dict

def load_scheme_lookup(quant_csv, embeddings_dict):
    lookup = {}
    with open(quant_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        id_col = 'model_id' if 'model_id' in headers else ('arch_index' if 'arch_index' in headers else 'id')
        
        # ADDED: Identify the True Rank column dynamically!
        rank_col = None
        for col in ["global_ranking", "rank", "pareto_rank"]:
            if col in headers:
                rank_col = col
                break
                
        for row in reader:
            try:
                arch_idx = int(float(row[id_col]))
                if arch_idx in embeddings_dict:
                    rank = int(float(row[rank_col])) if rank_col else -1
                    lookup[arch_idx] = {
                        'emb': embeddings_dict[arch_idx],
                        'true_rank': rank
                    }
            except Exception:
                continue
    return lookup

def convert_spec_to_str(spec):
    if isinstance(spec, str): return spec
    if isinstance(spec, dict) and 'string' in spec: return spec['string']
    if isinstance(spec, tuple) and len(spec) == 2:
        ops = spec[1]
        return f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
    raise ValueError("Cannot convert spec")

def get_idx_from_str(api, arch_str):
    try: return api.nasbench.query_index_by_arch(arch_str)
    except Exception: return -1

def get_mlp_scores_batch(arch_indices, predictor, lookup, stats, device):
    embeddings = [lookup[idx]['emb'] for idx in arch_indices]
    encoded_tensor = torch.stack(embeddings).to(device)
    with torch.no_grad():
        norm_emb = (encoded_tensor - stats['X_mean'].to(device)) / (stats['X_std'].to(device) + 1e-8)
        scores = predictor(norm_emb).squeeze().cpu().numpy()
    if scores.ndim == 0: scores = np.array([scores])
    return scores

# ============================================================================
# SEARCH ALGORITHMS (WITH STRICT BUDGET)
# ============================================================================
def run_random_search(search_space, predictor, lookup, stats, device, num_samples=100):
    valid_indices = []
    
    # Draw exactly 100 valid (unique) networks
    while len(valid_indices) < num_samples:
        batch = search_space.generate_random_dataset(num=num_samples)
        for arch in batch:
            a_str = convert_spec_to_str(arch['spec'])
            a_idx = get_idx_from_str(search_space, a_str)
            
            if a_idx != -1 and a_idx in lookup:
                if a_idx not in valid_indices:
                    valid_indices.append(a_idx)
            if len(valid_indices) == num_samples: break
            
    # Evaluate with the Predictor
    scores = get_mlp_scores_batch(valid_indices, predictor, lookup, stats, device)
    scored_indices = list(zip(valid_indices, scores))
    
    # Sort based on Predictor
    scored_indices.sort(key=lambda x: x[1], reverse=True) 
    
    # Return the IDs of the Top 10
    return [idx for idx, score in scored_indices[:10]]

def evolution_search_moea(
    search_space, pareto_mlp, stats, lookup, device,
    total_queries=80, num_init=20, population_size=20, tournament_size=5, mutation_rate=0.9
):
    init_data = []
    
    # 1. Initialization (Budget = 20)
    while len(init_data) < num_init:
        batch = search_space.generate_random_dataset(num=num_init)
        for arch in batch:
            a_str = convert_spec_to_str(arch['spec'])
            a_idx = get_idx_from_str(search_space, a_str)
            
            if a_idx != -1 and a_idx in lookup:
                existing_ids = [item['idx'] for item in init_data]
                if a_idx not in existing_ids:
                    score = get_mlp_scores_batch([a_idx], pareto_mlp, lookup, stats, device)[0]
                    init_data.append({'spec': arch['spec'], 'idx': a_idx, 'score': score})
            if len(init_data) == num_init: break

    population = list(range(len(init_data)))
    data = init_data

    # 2. Evolution Cycle (Budget = 80)
    for _ in range(total_queries):
        sample_indices = np.random.choice(population, tournament_size, replace=False)
        best_parent_idx = max(sample_indices, key=lambda i: data[i]['score'])
        parent_spec = data[best_parent_idx]['spec']
        
        try: mutated_spec = search_space.mutate_arch(parent_spec, mutation_rate=mutation_rate, mutate_encoding='adj', cutoff=0)
        except TypeError: mutated_spec = search_space.mutate_arch(parent_spec, mutation_rate=mutation_rate)

        m_str = convert_spec_to_str(mutated_spec)
        m_idx = get_idx_from_str(search_space, m_str)
        m_score = -100.0
        
        current_pop_ids = [data[i]['idx'] for i in population]
        
        if m_idx != -1 and m_idx in lookup:
            if m_idx not in current_pop_ids:
                if m_str.count('conv') >= 2:
                    m_score = get_mlp_scores_batch([m_idx], pareto_mlp, lookup, stats, device)[0]
                
        data.append({'spec': mutated_spec, 'idx': m_idx, 'score': m_score})
        population.append(len(data) - 1)
        
        # Kill the worst if population exceeds size
        if len(population) > population_size:
            worst_idx = min(population, key=lambda i: data[i]['score'])
            population.remove(worst_idx)
            
    final_population = [data[i] for i in population if data[i]['score'] != -100.0]
    final_population.sort(key=lambda x: x['score'], reverse=True)
    
    # Return the IDs of the Top 10
    return [item['idx'] for item in final_population[:10]]

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RS vs MOEA experiments on PTQ-NAS-Bench.")
    parser.add_argument('--project_root', type=str, required=True, help="Base directory of the project.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the CSV results for plotting.")
    parser.add_argument('--num_runs', type=int, default=200, help="Number of runs per scheme.")
    args = parser.parse_args()

    BASE_DIR = args.project_root
    NUM_RUNS = args.num_runs

    # --- Paths & Imports ---
    sys.path.append(os.path.join(BASE_DIR, 'naszilla'))
    sys.path.append(BASE_DIR)
    warnings.filterwarnings("ignore")

    # Import modules after appending to sys.path
    from naszilla.nas_benchmarks import Nasbench201
    from Pareto_Rank_Predictor.MLP_Architecture.Pareto_MLP.Pareto_mlp import ParetoRankPredictor

    # --- Experiment Settings (FAIR COMPARISON WITH BUDGET 100) ---
    BUDGET_RS = 100
    BUDGET_MOEA_INIT = 20
    BUDGET_MOEA_QUERIES = 80   # 20 + 80 = 100 Total Evaluations
    MOEA_POP_SIZE = 20         # Maintain a population of 20 networks 

    PATH_FP32_DATA = os.path.join(BASE_DIR, "Pareto_Rank_Predictor/MLP_Architecture/Dataset/FP32/encoded_fully_dataset_cifar10_pooling_edgegpu_flops_fp32.csv")

    QUANT_SCHEMES = {
        "Q1": {
            "name": "Q1",
            "csv": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Datasets/Final_eval_dataset/final_dataset_Q1_8W8A_QuantAcc_ranked.csv"),
            "weights": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q1/end2end_model_4000_lhs_cifar10_120ep_best_Q1_8W8A.pth"),
            "stats": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q1/end2end_stats_4000_lhs_cifar10_120ep_best_Q1_8W8A.pth")
        },
        "Q2": {
            "name": "Q2",
            "csv": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Datasets/Final_eval_dataset/final_dataset_Q2_4W4A_QuantAcc_ranked.csv"),
            "weights": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q2/end2end_model_4000_lhs_cifar10_120ep_best_Q2_4W4A.pth"),
            "stats": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q2/end2end_stats_4000_lhs_cifar10_120ep_best_Q2_4W4A.pth")
        },
        "Q3": {
            "name": "Q3",
            "csv": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Datasets/Final_eval_dataset/final_dataset_Q3_2W8A_QuantAcc_ranked.csv"),
            "weights": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q3/end2end_model_4000_lhs_cifar10_120ep_best_Q3_2W8A.pth"),
            "stats": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q3/end2end_stats_4000_lhs_cifar10_120ep_best_Q3_2W8A.pth")
        },
        "Q4": {
            "name": "Q4",
            "csv": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Datasets/Final_eval_dataset/final_dataset_Q4_2W2A_QuantAcc_ranked.csv"),
            "weights": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q4/end2end_model_4000_lhs_cifar10_120ep_best_Q4_2W2A.pth"),
            "stats": os.path.join(BASE_DIR, "BenchMark_Q_schemes/Predictors/Q4/end2end_stats_4000_lhs_cifar10_120ep_best_Q4_2W2A.pth")
        }
    }

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TARGET_DATASET = 'cifar10'
    
    print("\n" + "="*80)
    print(f"🚀 STARTING FAIR NAS EVALUATION (Budget: 100 Evals) - {NUM_RUNS} RUNS PER SCHEME")
    print("="*80)

    nas_api = Nasbench201(dataset=TARGET_DATASET)
    global_embeddings = load_embeddings_from_fp32(PATH_FP32_DATA)
    
    boxplot_data = [] # Data container for the plots

    for scheme_key, info in QUANT_SCHEMES.items():
        scheme_name = info["name"]
        
        print("\n" + "*"*60)
        print(f"⚙️ SCHEME: {scheme_name}")
        print("*"*60)
        
        lookup_dict = load_scheme_lookup(info["csv"], global_embeddings)
        stats = torch.load(info["stats"], map_location=device)
        
        pareto_model = ParetoRankPredictor(input_dim=104, hidden_dim=64).to(device)
        pareto_model.load_state_dict(torch.load(info["weights"], map_location=device))
        pareto_model.eval()
        
        for run in range(NUM_RUNS):
            # Ensure reproducibility
            np.random.seed(run)
            
            # --- 1. Random Search (Budget 100) ---
            rs_top10_ids = run_random_search(nas_api, pareto_model, lookup_dict, stats, device, num_samples=BUDGET_RS)
            for m_id in rs_top10_ids:
                true_rank = lookup_dict[m_id]['true_rank']
                boxplot_data.append({
                    "Scheme": scheme_name,
                    "Run": run,
                    "Algorithm": "Random Search",
                    "Global Actual Rank": true_rank
                })
            
            # --- 2. MOEA (Budget 100: Init 20 + 80 Queries) ---
            moea_top10_ids = evolution_search_moea(nas_api, pareto_model, stats, lookup_dict, device,
                                                   total_queries=BUDGET_MOEA_QUERIES, num_init=BUDGET_MOEA_INIT, population_size=MOEA_POP_SIZE)
            for m_id in moea_top10_ids:
                true_rank = lookup_dict[m_id]['true_rank']
                boxplot_data.append({
                    "Scheme": scheme_name,
                    "Run": run,
                    "Algorithm": "MOEA (NSGA-II)",
                    "Global Actual Rank": true_rank
                })
                
            # Print update
            rs_best_rank = min([lookup_dict[i]['true_rank'] for i in rs_top10_ids])
            moea_best_rank = min([lookup_dict[i]['true_rank'] for i in moea_top10_ids])
            print(f"   [Run {run+1}/{NUM_RUNS}] Best Rank Found -> RS: {rs_best_rank} | MOEA: {moea_best_rank}")

    # --- Save Data for Boxplots ---
    df_out = pd.DataFrame(boxplot_data)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_filename = os.path.join(args.output_dir, f"rs_vs_moea_ranks_budget100_{NUM_RUNS}runs.csv")
    df_out.to_csv(csv_filename, index=False)
    
    print("\n" + "="*70)
    print(f"⏱️ Total Execution Time: {(time.time() - start_time):.2f} seconds")
    print(f"[+] Data for Boxplots successfully saved to:\n    -> {csv_filename}")
    print("="*70)