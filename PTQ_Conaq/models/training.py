import sys
import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder for Pareto Rank Predictor")
    parser.add_argument('--naszilla_dir', type=str, required=True, help="Path to the naszilla repository")
    parser.add_argument('--project_root', type=str, required=True, help="Path to the HW-PR-NAS project root")
    parser.add_argument('--cache_file', type=str, required=True, help="Path to the FE cache .pt file")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dynamically append paths
    sys.path.append(args.naszilla_dir)
    sys.path.append(args.project_root)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Imports must happen after sys.path is updated
    # Imports must happen after sys.path is updated
    from PTQ_Conaq.models.Encoder.nas_dataset import NasBench201Dataset, collate_fn
    from naszilla.nas_benchmarks import Nasbench201
    from PTQ_Conaq.models.Encoder.gcn_encoding import GCNEncoder
    from PTQ_Conaq.models.Encoder.lstm_encoding import LSTMEncoder
    from PTQ_Conaq.models.Encoder.decoder import AttentionDecoder

    class FE_Manager_Cached:
        def __init__(self, cache_path, nasbench_api, device='cpu'):
            print(f"[*] Loading FE cache from {cache_path}...")
            self.cache = torch.load(cache_path).to(device)
            self.api = nasbench_api.nasbench
            self.device = device

        def get_fe_embedding(self, arch_strings):
            indices = [self.api.query_index_by_arch(s) for s in arch_strings]
            return self.cache[indices]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # Load API and Dataset
    print("[*] Initializing NAS-Bench-201 API and Dataset...")
    nasbench = Nasbench201(dataset='cifar10')
    full_dataset = NasBench201Dataset(dataset_name='cifar10', num_samples=None, shuffle=True)
    
    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=lambda b: collate_fn(b, device=device))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=lambda b: collate_fn(b, device=device))

    VOCAB_SIZE = len(full_dataset.extractor.tokenizer.token2idx)
    
    fe_manager = FE_Manager_Cached(args.cache_file, nasbench, device=device)
    
    gcn_enc = GCNEncoder(num_node_features=7).to(device)
    lstm_enc = LSTMEncoder(vocab_size=VOCAB_SIZE).to(device)
    decoder = AttentionDecoder(vocab_size=VOCAB_SIZE).to(device)

    optimizer = optim.Adam(
        list(gcn_enc.parameters()) + list(lstm_enc.parameters()) + list(decoder.parameters()), 
        lr=args.lr, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n[🚀] Starting Fast Training...")
    
    for epoch in range(args.epochs):
        # 1. TRAINING LOOP
        gcn_enc.train(); lstm_enc.train(); decoder.train()
        train_loss_total = 0
        start_t = time.time()
        
        for batch_idx, (arch_strings, ops_batch, adj_batch, targets, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Encoding
            gcn_emb = gcn_enc(ops_batch, adj_batch)
            lstm_emb, enc_outputs = lstm_enc(targets)
            fe_emb = fe_manager.get_fe_embedding(arch_strings) 
            
            latent_vector = torch.cat([gcn_emb, lstm_emb, fe_emb], dim=1)
            
            # Decoding with Teacher Forcing
            teacher_forcing_ratio = 0.5 
            hidden, cell = decoder.init_hidden_from_concat(latent_vector)
            decoder_input = targets[:, 0].unsqueeze(1) 

            batch_loss = 0
            for t in range(1, targets.size(1)):
                pred, hidden, cell = decoder(decoder_input, hidden, cell, enc_outputs, latent_vector)
                batch_loss += criterion(pred, targets[:, t])
                
                if random.random() < teacher_forcing_ratio:
                    decoder_input = targets[:, t].unsqueeze(1) 
                else:
                    top1 = pred.argmax(1) 
                    decoder_input = top1.unsqueeze(1)
                    
            batch_loss /= (targets.size(1) - 1)
            batch_loss.backward()
            optimizer.step()
            train_loss_total += batch_loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch:02d} | Batch {batch_idx:03d}/{len(train_loader)} | Train Loss: {batch_loss.item():.4f}", flush=True)

        avg_train_loss = train_loss_total / len(train_loader)

        # 2. VALIDATION LOOP
        gcn_enc.eval(); lstm_enc.eval(); decoder.eval()
        val_loss_total = 0
        
        with torch.no_grad():
            for batch_idx, (arch_strings, ops_batch, adj_batch, targets, _) in enumerate(val_loader):
                gcn_emb = gcn_enc(ops_batch, adj_batch)
                lstm_emb, enc_outputs = lstm_enc(targets)
                fe_emb = fe_manager.get_fe_embedding(arch_strings)
                
                latent_vector = torch.cat([gcn_emb, lstm_emb, fe_emb], dim=1)
                
                hidden, cell = decoder.init_hidden_from_concat(latent_vector)
                decoder_input = targets[:, 0].unsqueeze(1)
                
                batch_val_loss = 0
                for t in range(1, targets.size(1)):
                    pred, hidden, cell = decoder(decoder_input, hidden, cell, enc_outputs, latent_vector)
                    batch_val_loss += criterion(pred, targets[:, t])
                    decoder_input = targets[:, t].unsqueeze(1) 
                
                batch_val_loss /= (targets.size(1) - 1)
                val_loss_total += batch_val_loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        
        print(f"✅ Epoch {epoch:02d} Done | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_t:.2f}s")

        # Save Checkpoint
        if epoch % 10 == 0 or epoch == (args.epochs - 1):
            ckpt_path = os.path.join(args.checkpoint_dir, f"autoencoder_epoch_{epoch}.pth")
            torch.save({
                'gcn': gcn_enc.state_dict(),
                'lstm': lstm_enc.state_dict(),
                'decoder': decoder.state_dict()
            }, ckpt_path)

if __name__ == "__main__":
    main()