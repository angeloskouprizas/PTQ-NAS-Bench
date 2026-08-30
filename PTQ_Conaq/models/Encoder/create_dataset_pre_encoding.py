import sys
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# --- Paths ---
BASE_DIR = '/home/lefterismyl/Downloads/HW_aware_NAS/HW-PR-NAS'
sys.path.append(os.path.join(BASE_DIR, 'naszilla'))
sys.path.append(BASE_DIR)

from naszilla.nas_benchmarks import Nasbench201
from Pareto_Rank_Predictor.MLP_Architecture.Encoding.nas_dataset import ArchExtractor, collate_fn
from Pareto_Rank_Predictor.MLP_Architecture.Encoding.gcn_encoding import GCNEncoder
from Pareto_Rank_Predictor.MLP_Architecture.Encoding.lstm_encoding import LSTMEncoder

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
def generate_dataset_from_csv():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Ξεκινάει η Δημιουργία του Encoded Dataset στο {device} (ΧΩΡΙΣ Pooling)...")

    # 1. Φόρτωση του ΕΝΟΣ CSV αρχείου
    # ---------------------------------------------------------------------
    dataset_file = os.path.join(BASE_DIR, "/home/lefterismyl/Downloads/HW_aware_NAS/HW-PR-NAS/Pareto_Rank_Predictor/MLP_Architecture/Dataset/FP32/lhs_dataset_500_cifar10_edgegpu_flops_fp32.csv")
    
    if not os.path.exists(dataset_file):
        print(f"❌ Το αρχείο δεν βρέθηκε στο path: {dataset_file}")
        return

    df_data = pd.read_csv(dataset_file)
    print(f"📊 Βρέθηκαν {len(df_data)} αρχιτεκτονικές για Encoding.")

    # 2. Φόρτωση API και Extractor
    nasbench = Nasbench201(dataset='cifar10')
    api = nasbench.nasbench
    extractor = ArchExtractor(dataset='cifar10', nas_api=nasbench)
    VOCAB_SIZE = len(extractor.tokenizer.token2idx)

    # 3. Αρχικοποίηση και Φόρτωση Models
    cache_file = os.path.join(BASE_DIR, "Pareto_Rank_Predictor/MLP_Architecture/Dataset/nas201_fe_cache.pt")
    fe_manager = FE_Manager_Cached(cache_file, nasbench, device=device)
    
    gcn_enc = GCNEncoder(num_node_features=7).to(device)
    lstm_enc = LSTMEncoder(vocab_size=VOCAB_SIZE).to(device)

    # Φόρτωση Checkpoint
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "Pareto_Rank_Predictor/MLP_Architecture/Encoding/Weights/autoencoder_epoch_dropout_latent_90.pth") 
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📥 Φόρτωση εκπαιδευμένων βαρών από {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        gcn_enc.load_state_dict(checkpoint['gcn'])
        lstm_enc.load_state_dict(checkpoint['lstm'])
    else:
        print(f"❌ Το αρχείο βαρών {CHECKPOINT_PATH} δεν βρέθηκε. Σταματάω.")
        return

    gcn_enc.eval()
    lstm_enc.eval()

    # 4. Δημιουργία Λίστας Αποτελεσμάτων
    results = []

    print("\n✍️ Εξαγωγή Embeddings...")
    with torch.no_grad():
        for index, row in tqdm(df_data.iterrows(), total=len(df_data)):
            # Προσαρμογή στα ονόματα των στηλών του νέου CSV
            arch_id = int(row['model_id'])
            acc = float(row['true_accuracy']) 
            raw_flops = float(row['raw_flops'])
            rank = int(row['global_ranking']) if not pd.isna(row['global_ranking']) else -1
            
            # Ανάκτηση string
            arch_str = api.arch(arch_id)
            ops, adj, tokens = extractor.extract(arch_str)
            
            # Χρήση του collate_fn 
            batch = [(arch_str, ops, adj, torch.LongTensor(tokens), raw_flops)]
            arch_strings_b, ops_batch, adj_batch, targets_b, _ = collate_fn(batch, device=device)
            
                        # --- ENCODING/NO-POOLING ---
            gcn_emb = gcn_enc(ops_batch, adj_batch)
            
            # 🔥 THE FIX: Παίρνουμε ΜΟΝΟ το lstm_hidden_state που διατηρεί τη ΣΕΙΡΑ των πράξεων!
            lstm_hidden_state, enc_outputs = lstm_enc(targets_b) 
            
            fe_emb = fe_manager.get_fe_embedding(arch_strings_b)
            
            # Ενώνουμε το GCN, το lstm_hidden_state, και το FE
            latent_vector = torch.cat([gcn_emb, lstm_hidden_state, fe_emb], dim=1)
            '''
            # --- ENCODING/POOLING ---
            gcn_emb = gcn_enc(ops_batch, adj_batch)
            
            # Τραβάμε τα enc_outputs από το LSTM (που περιέχουν όλα τα tokens)
            _, enc_outputs = lstm_enc(targets_b) 
            
            # 🔥 THE POOLING FIX: Κάνουμε Mean Pooling σε όλα τα tokens του sequence!
            # Το enc_outputs έχει σχήμα [batch_size, sequence_length, hidden_dim]
            lstm_pooled = torch.mean(enc_outputs, dim=1) 
            
            fe_emb = fe_manager.get_fe_embedding(arch_strings_b)
            
            # Ενώνουμε το GCN (Τοπολογία), το lstm_pooled (Πράξεις), και το FE (Παράμετροι)
            latent_vector = torch.cat([gcn_emb, lstm_pooled, fe_emb], dim=1)
            '''
            # --- ΑΠΟΘΗΚΕΥΣΗ ---
            results.append({
                'arch_index': arch_id,
                #'concatenated_gcn': str(gcn_emb.squeeze(0).cpu().numpy().tolist()),
                #'concatenated_lstm': str(lstm_pooled.squeeze(0).cpu().numpy().tolist()),
                #'fe_emb': str(fe_emb.squeeze(0).cpu().numpy().tolist()),
                'concatenated': str(latent_vector.squeeze(0).cpu().numpy().tolist()),
                'accuracy': acc,
                'flops_32': raw_flops,
                'rank': rank
            })
            

    # 5. Αποθήκευση σε νέο CSV
    # Αλλαγή ονόματος για να ξεχωρίζει από το dataset με το pooling
    OUTPUT_CSV = os.path.join(BASE_DIR, "Pareto_Rank_Predictor/MLP_Architecture/Dataset/FP32/encoded_500_dataset_cifar10_no_pooling_edgegpu_flops_fp32.csv")
    df_output = pd.DataFrame(results)
    df_output.to_csv(OUTPUT_CSV, index=False)

    print(f"\n🎉 Τέλεια! Το τελικό dataset (ΧΩΡΙΣ Pooling) αποθηκεύτηκε στο: \n{OUTPUT_CSV}")

if __name__ == "__main__":
    generate_dataset_from_csv()