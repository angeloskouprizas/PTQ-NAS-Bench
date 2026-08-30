import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    LSTM Encoding Scheme (According to Table 2 of the paper)
    - 2 Layers
    - Hidden units: [32, 64]
    - FC dimension: 32
    """
    def __init__(self, vocab_size, embed_dim=32, fc_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Layer 1: 32 hidden units
        self.lstm1 = nn.LSTM(embed_dim, 32, batch_first=True)
        # Layer 2: 64 hidden units
        self.lstm2 = nn.LSTM(32, 64, batch_first=True)
        
        self.fc = nn.Linear(64, fc_dim)

    def forward(self, x):
        # x shape: (Batch, Sequence_Length)
        embedded = self.embedding(x)
        
        out1, _ = self.lstm1(embedded)
        encoder_outputs, (h_n, c_n) = self.lstm2(out1)
        
        # Get the last hidden state of the last layer
        last_hidden = h_n[-1]
        
        # Final Dense Layer to reduce dimension to 32
        lstm_emb = self.fc(last_hidden)
        
        # Return the embedding (for concatenation) AND 
        # the encoder_outputs (for the Decoder's Attention mechanism)
        return lstm_emb, encoder_outputs
