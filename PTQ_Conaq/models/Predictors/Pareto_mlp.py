import torch
import torch.nn as nn

class ParetoRankPredictor(nn.Module):
    """3-Layer FCNN for Pareto Rank Prediction"""
    def __init__(self, input_dim, hidden_dim=64, dropout=0.02):
        super().__init__()
        
        # A simple and small network.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, embedding):
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        # We return the raw scores (logits) for ListMLE
        f_a = self.net(embedding) 
        return f_a
