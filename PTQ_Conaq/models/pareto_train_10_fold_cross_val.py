import sys
import os
import time
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import kendalltau
from sklearn.model_selection import KFold
import argparse

warnings.filterwarnings("ignore")

# ============================================================================
# 1. ListMLE-Loss-Function
# ============================================================================
def listMLE_loss(out_a, sorted_indices):
    """takes indices"""
    out_sorted = out_a[sorted_indices]
    B = out_sorted.size(0)
    loss = 0.0
    for i in range(B):
        term1 = -out_sorted[i]
        term2 = torch.logsumexp(out_sorted[i:], dim=0) 
        loss += (term1 + term2)
    return loss / B

# ============================================================================
# 2. Data Loading & Normalization (Adapted for the new LHS dataset)
# ============================================================================
def load_lhs_encoded_data(csv_file, device):
    concatenated_list, rank_list = [], [] 
    print(f"\n⏳ Loading architectures & LHS Ranks from: {csv_file}...")
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Read 'encoded_arch' which contains the concatenated embedding
            emb_list = json.loads(row['encoded_arch'])
            emb_tensor = torch.tensor(emb_list, dtype=torch.float32)
            concatenated_list.append(emb_tensor)
            
            # Read the rank
            rank_list.append(float(row['rank']))

    concatenated_emb = torch.stack(concatenated_list).to(device)
    ranks = torch.tensor(rank_list, dtype=torch.float32)
    
    print(f"✅ Successfully loaded {len(ranks)} samples with Embedding Dimension: {concatenated_emb.shape[1]}")
    return concatenated_emb, ranks

def compute_stats_from_tensors(embeddings):
    """Compute mean and std only for the training set"""
    X_mean = embeddings.mean(dim=0, keepdim=True)
    X_std = embeddings.std(dim=0, keepdim=True)
    X_std = torch.where(X_std < 1e-6, torch.tensor(1.0, device=embeddings.device), X_std)
    return {'X_mean': X_mean, 'X_std': X_std}

# ============================================================================
# 3. End-to-End Training Loop
# ============================================================================
def train_end2end_model(model, train_loader, val_loader, stats, epochs=100, lr=1e-3, device='cpu', fold_idx=0):
    model.to(device)
    X_mean, X_std = stats['X_mean'].to(device), stats['X_std'].to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_tau = -float('inf')
    best_state = None
    
    print(f"\n--- FOLD {fold_idx} ---")
    print(f"{'Epoch':^5} | {'Train Loss':^12} | {'Val Tau (Global)':^15}")
    print("-" * 40)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Now the loader returns exactly 2 tensors (Embedding, Rank)
        for concatenated_emb, global_ranks in train_loader:
            concatenated_emb = concatenated_emb.to(device)
            global_ranks = global_ranks.to(device)
            
            # Sort based on Pareto Ranks for the ListMLE loss
            sorted_indices = torch.argsort(global_ranks)

            norm_emb = (concatenated_emb - X_mean) / X_std
            
            optimizer.zero_grad()
            scores = model(norm_emb).view(-1)
            
            loss = listMLE_loss(scores, sorted_indices)
            l2_logits_penalty = 1e-3 * torch.mean(scores ** 2)
            total_loss = loss + l2_logits_penalty
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

        # --- VALIDATION LOOP ---
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for concatenated_emb, global_true_ranks in val_loader:
                concatenated_emb = concatenated_emb.to(device)
                norm_emb = (concatenated_emb - X_mean) / X_std
                scores = model(norm_emb)
                
                # The predictor predicts a higher score for the best (Rank 1), so we negate for comparison
                all_preds.extend((-scores.view(-1)).cpu().numpy())
                all_targets.extend(global_true_ranks.cpu().numpy())

        avg_v_tau, _ = kendalltau(all_targets, all_preds)
        if np.isnan(avg_v_tau): avg_v_tau = 0.0 

        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{epoch+1:^5} | {epoch_loss/len(train_loader):^12.4f} | {avg_v_tau:^15.4f}")
        
        if avg_v_tau > best_val_tau:
            best_val_tau = avg_v_tau
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    print(f"✅ Fold {fold_idx} Complete. Best Val Tau: {best_val_tau:.4f}")
    
    return model, best_val_tau, best_state

# ============================================================================
# MAIN: 10-Fold Cross Validation on LHS Dataset
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train End-to-End Model")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory of the project (e.g., /home/.../HW-PR-NAS)")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory containing the encoded LHS CSV files")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the best models")
    args = parser.parse_args()

    # --- Adjust Paths ---
    sys.path.append(os.path.join(args.base_dir, 'naszilla'))
    sys.path.append(args.base_dir)

    # Locate the MLP of pareto and parse it here.
    from PTQ_Conaq.models.Pareto_MLP.Pareto_mlp import ParetoRankPredictor

    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Settings and Paths
    TARGET_DATASET = 'cifar10' 
    QUANT_SCHEME = 'Q4'  # e.g. Q1, Q2, Q3, Q4 - So you know what you are training!
    QUANT_BITS = "2W2A"  # e.g. 4 bits quantization
    
    # The file created in the previous script
    csv_file = os.path.join(args.dataset_dir, f"encoded_lhs_{QUANT_SCHEME}_{QUANT_BITS}_FP32_with_LogQNP.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"❌ File not found: {csv_file}")
        
    # Load Data
    concatenated_emb, true_ranks = load_lhs_encoded_data(csv_file, device)
    full_ds = TensorDataset(concatenated_emb, true_ranks)
    
    # Training Settings (Hyperparameters)
    batch_size = 18 
    epochs = 120 
    k_folds = 10 
    
    print(f"\n🚀 Starting {k_folds}-Fold Cross Validation for Quantization Scheme: [{QUANT_SCHEME}]...")
    
    # Initialize K-Fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    best_overall_tau = -float('inf')
    best_overall_model_state = None
    best_overall_stats = None

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_ds)):
        
        # 1. Create Subsets for Train and Validation
        train_sub = Subset(full_ds, train_ids)
        val_sub = Subset(full_ds, val_ids)
        
        # 2. Compute Stats ONLY from Train Set of this fold to avoid Data Leakage
        train_emb_fold = concatenated_emb[train_ids]
        stats = compute_stats_from_tensors(train_emb_fold)
        
        # 3. DataLoaders
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        
        # 4. Initialize BRAND NEW model for each fold
        pareto_model = ParetoRankPredictor(
            input_dim=concatenated_emb.shape[1], 
            hidden_dim=64,
            dropout=0.1
        ).to(device)
        
        # 5. Train the Fold
        trained_model, best_fold_tau, fold_state = train_end2end_model(
            model=pareto_model,
            train_loader=train_loader,
            val_loader=val_loader,
            stats=stats,
            epochs=epochs, 
            lr=5e-4,
            device=device,
            fold_idx=fold + 1
        )
        
        fold_results.append(best_fold_tau)
        
        # Keep the absolutely best model from all folds
        if best_fold_tau > best_overall_tau:
            best_overall_tau = best_fold_tau
            best_overall_model_state = fold_state.copy()
            best_overall_stats = stats.copy()

    # ==========================================
    # Results & Saving
    # ==========================================
    print("\n" + "="*50)
    print(f"🏆 10-FOLD CROSS VALIDATION RESULTS ({QUANT_SCHEME})")
    print("="*50)
    for i, res in enumerate(fold_results):
        print(f"Fold {i+1}: Kendall Tau = {res:.4f}")
    
    avg_tau = np.mean(fold_results)
    std_tau = np.std(fold_results)
    print("-" * 50)
    print(f"🎯 AVERAGE KENDALL TAU: {avg_tau:.4f} ± {std_tau:.4f}")
    print("="*50)

    # Save weights of the best Fold
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Added {QUANT_SCHEME} and the word 'lhs' to the name to stand out!
    save_path = os.path.join(args.save_dir, f'end2end_model_{len(full_ds)}_lhs_{TARGET_DATASET}_{epochs}ep_best_{QUANT_SCHEME}_{QUANT_BITS}.pth')
    stats_path = os.path.join(args.save_dir, f'end2end_stats_{len(full_ds)}_lhs_{TARGET_DATASET}_{epochs}ep_best_{QUANT_SCHEME}_{QUANT_BITS}.pth')
    
    print(f"\n💾 Saving BEST fold model (KT={best_overall_tau:.4f}) to:\n   -> {save_path}")
    
    pareto_model.load_state_dict(best_overall_model_state)
    torch.save(pareto_model.state_dict(), save_path)
    torch.save(best_overall_stats, stats_path)
    
    print(f"\n🚀 Process completed successfully in {(time.time() - start)/60:.2f} minutes!")