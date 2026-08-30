import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + 64, hidden_dim) # 64 is the output size of LSTM encoder
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        # Repeat the hidden state for each time step of the encoder
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate Energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        
        return F.softmax(attention_scores, dim=1)

class AttentionDecoder(nn.Module):
    """
    Decoder (According to Section 4.1 and Table 2)
    - Receives the Concatenated Encoding (GCN 32 + LSTM 32 + FE 8 = 72)
    - 3 LSTM layers (based on the table, with 0.2 dropout)
    - Attention mechanism
    """
    def __init__(self, vocab_size, concat_dim=72, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = Attention(hidden_dim)
        
        # Projector: Converts (Batch, 72) to the initial Hidden State of the Decoder (Batch, 64)
        self.init_h = nn.Linear(concat_dim, hidden_dim)
        self.init_c = nn.Linear(concat_dim, hidden_dim)
        
        # LSTM input comprises: Word Embedding (hidden_dim) + Context Vector (64) + Latent Vector (concat_dim)
        self.lstm = nn.LSTM(hidden_dim + 64 + concat_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim + 64, vocab_size)

    def init_hidden_from_concat(self, concatenated_vector):
        """The 72-dim vector acts as a seed for the Decoder."""
        h = self.init_h(concatenated_vector).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(concatenated_vector).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, c
    
    def forward(self, input_token, hidden, cell, encoder_outputs, latent_vector):
        embedded = self.embedding(input_token) 
        
        # Drop a percentage of encoder outputs during training to force reliance on the hidden state
        if self.training:
            encoder_outputs = F.dropout(encoder_outputs, p=0.5)
        
        # Attention using the top hidden layer
        top_hidden = hidden[-1]
        a = self.attention(top_hidden, encoder_outputs).unsqueeze(1)
        
        # Context vector (weighted sum of encoder outputs)
        context = torch.bmm(a, encoder_outputs) 
        
        # Prepare latent_vector for LSTM input (Batch, 1, 72)
        latent_expanded = latent_vector.unsqueeze(1)
        
        # Concatenate: input token + context vector + latent vector
        rnn_input = torch.cat((embedded, context, latent_expanded), dim=2)
        
        # One LSTM step
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        # Combine LSTM output (64) with Attention context (64) = 128
        combined = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc_out(combined)

        return prediction, hidden, cell