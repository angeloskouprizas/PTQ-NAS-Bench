import sys
import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Encoded Dataset for Pareto Rank Predictor")
    parser.add_argument('--naszilla_dir', type=str, required=True, help="Path to the naszilla repository")
    parser.add_argument('--project_root', type=str, required=True, help="Path to the HW-PR-NAS project root")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV (e.g., LHS dataset)")
    parser.add_argument('--cache_file', type=str, required=True, help="Path to the FE cache .pt file")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the trained autoencoder weights")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the encoded dataset CSV")
    return parser.parse_args()

def main():
    args = parse_args()

    # Dynamically append paths
    sys.path.append(args.naszilla_dir)
    sys.path.append(args.project_root)

    from naszilla.nas_benchmarks import Nasbench201
    from PTQ_Conaq.models.Encoder.nas_dataset import ArchExtractor, collate_fn
    from PTQ_Conaq.models.Encoder.gcn_encoding import GCNEncoder
    from PTQ_Conaq.models.Encoder.lstm_encoding import LSTMEncoder

    # ============================================================================
    # FE Manager
    # ============================================================================
    class FE_Manager_Cached:
        def __init__(self, cache_path, nasbench_api, device='cpu'):
            self.cache = torch.load(cache_path).to(device)
            self.api = nasbench_api.nasbench
            self.device = device

        def get_fe_embedding(self, arch_strings):
            indices = [self.api.query_index_by_arch(s) for s in arch_strings]
            return self.cache[indices]

    # ============================================================================
    # MAIN GENERATOR
    # ============================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[🚀] Starting Encoded Dataset Generation on {device} (NO Pooling)...")

    # 1. Load the input CSV file
    if not os.path.exists(args.input_csv):
        print(f"[-] Input file not found at path: {args.input_csv}")
        return

    df_data = pd.read_csv(args.input_csv)
    print(f"[*] Found {len(df_data)} architectures for Encoding.")

    # 2. Load API and Extractor
    nasbench = Nasbench201(dataset='cifar10')
    api = nasbench.nasbench
    extractor = ArchExtractor(dataset='cifar10', nas_api=nasbench)
    VOCAB_SIZE = len(extractor.tokenizer.token2idx)

    # 3. Initialize and Load Models
    fe_manager = FE_Manager_Cached(args.cache_file, nasbench, device=device)
    
    gcn_enc = GCNEncoder(num_node_features=7).to(device)
    lstm_enc = LSTMEncoder(vocab_size=VOCAB_SIZE).to(device)

    # Load Checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"[*] Loading trained weights from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        gcn_enc.load_state_dict(checkpoint['gcn'])
        lstm_enc.load_state_dict(checkpoint['lstm'])
    else:
        print(f"[-] Weights file {args.checkpoint_path} not found. Exiting.")
        return

    gcn_enc.eval()
    lstm_enc.eval()

    # 4. Create Results List
    results = []

    print("\n[*] Extracting Embeddings...")
    with torch.no_grad():
        for index, row in tqdm(df_data.iterrows(), total=len(df_data)):
            # Adapt to the column names of the new CSV
            arch_id = int(row['model_id'])
            acc = float(row['true_accuracy']) 
            raw_flops = float(row['raw_flops'])
            rank = int(row['global_ranking']) if not pd.isna(row['global_ranking']) else -1
            
            # Retrieve string
            arch_str = api.arch(arch_id)
            ops, adj, tokens = extractor.extract(arch_str)
            
            # Use collate_fn 
            batch = [(arch_str, ops, adj, torch.LongTensor(tokens), raw_flops)]
            arch_strings_b, ops_batch, adj_batch, targets_b, _ = collate_fn(batch, device=device)
            '''
            # --- ENCODING / NO-POOLING ---
            gcn_emb = gcn_enc(ops_batch, adj_batch)
            
            # THE FIX: Take ONLY the lstm_hidden_state that preserves the sequence of operations!
            lstm_hidden_state, enc_outputs = lstm_enc(targets_b) 
            
            fe_emb = fe_manager.get_fe_embedding(arch_strings_b)
            
            # Concatenate GCN, lstm_hidden_state, and FE
            latent_vector = torch.cat([gcn_emb, lstm_hidden_state, fe_emb], dim=1)
            
            '''
            # --- ENCODING / WITH POOLING (Alternative Approach) ---
            gcn_emb = gcn_enc(ops_batch, adj_batch)
            
            # Extract enc_outputs from LSTM (contains all tokens)
            _, enc_outputs = lstm_enc(targets_b) 
            
            # THE POOLING FIX: Apply Mean Pooling across all tokens in the sequence!
            # enc_outputs shape is [batch_size, sequence_length, hidden_dim]
            lstm_pooled = torch.mean(enc_outputs, dim=1) 
            
            fe_emb = fe_manager.get_fe_embedding(arch_strings_b)
            
            # Concatenate GCN (Topology), lstm_pooled (Operations), and FE (Parameters)
            latent_vector = torch.cat([gcn_emb, lstm_pooled, fe_emb], dim=1)
            
            
            # --- SAVE ---
            results.append({
                'arch_index': arch_id,
                'concatenated': str(latent_vector.squeeze(0).cpu().numpy().tolist()),
                'accuracy': acc,
                'flops_32': raw_flops,
                'rank': rank
            })

    # 5. Save to new CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_output = pd.DataFrame(results)
    df_output.to_csv(args.output_csv, index=False)

    print(f"\n[+] Success! The final encoded dataset was saved to: \n{args.output_csv}")

if __name__ == "__main__":
    main()