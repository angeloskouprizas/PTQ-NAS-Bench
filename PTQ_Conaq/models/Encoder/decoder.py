

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
        # Επανάληψη του hidden state για κάθε χρονικό βήμα του encoder
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Υπολογισμός Energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        
        return F.softmax(attention_scores, dim=1)

class AttentionDecoder(nn.Module):
    """
    Decoder (Σύμφωνα με το Section 4.1 και Table 2)
    - Παίρνει το Concatenated Encoding (GCN 32 + LSTM 32 + FE 8 = 72)
    - 3 ή 4 LSTM layers (χρησιμοποιούμε 3 βάσει του πίνακα, με 0.2 dropout)
    - Μηχανισμός Attention
    """
    def __init__(self, vocab_size, concat_dim=72, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = Attention(hidden_dim)
        
        # Projector: Μετατρέπει το (Batch, 72) στο αρχικό Hidden State του Decoder (Batch, 64)
        self.init_h = nn.Linear(concat_dim, hidden_dim)
        self.init_c = nn.Linear(concat_dim, hidden_dim)
        
        # 🔥 ΝΕΟ: Το LSTM πλέον δέχεται 3 πράγματα:
        # 1. Το Embedding της λέξης (hidden_dim)
        # 2. Το Context Vector από το Attention (64)
        # 3. Το ΟΛΟΚΛΗΡΟ latent_vector (concat_dim, δηλαδή 72)
        # Σύνολο: 64 + 64 + 72 = 200 dimensions στην είσοδο του LSTM!
        self.lstm = nn.LSTM(hidden_dim + 64 + concat_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim + 64, vocab_size)

    def init_hidden_from_concat(self, concatenated_vector):
        """Εδώ το τεράστιο vector των 72-dim μπαίνει ως σπόρος (seed) στον Decoder!"""
        h = self.init_h(concatenated_vector).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(concatenated_vector).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, c
    
    '''
    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: (Batch, 1)
        embedded = self.embedding(input_token) # (Batch, 1, hidden_dim)
        
        # Attention χρησιμοποιώντας το πάνω-πάνω hidden layer
        top_hidden = hidden[-1]
        a = self.attention(top_hidden, encoder_outputs) # (Batch, Seq_Len)
        a = a.unsqueeze(1) # (Batch, 1, Seq_Len)
        
        # Context vector (σταθμισμένο άθροισμα των encoder outputs)
        context = torch.bmm(a, encoder_outputs) # (Batch, 1, 64)
        
        # Συνένωση του input token και του context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # Ένα βήμα (1 step) στον Decoder
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # Τελική Πρόβλεψη του επόμενου token (operation)
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        
        return prediction, hidden, cell
    '''
    # 🔥 ΝΕΟ: Η συνάρτηση forward πλέον δέχεται ως όρισμα ΚΑΙ το latent_vector!
    def forward(self, input_token, hidden, cell, encoder_outputs, latent_vector):
        # input_token: (Batch, 1)
        embedded = self.embedding(input_token) # (Batch, 1, hidden_dim)
        
        # -------------------------------------------------------------
        # ΤΟ ΜΥΣΤΙΚΟ TRICK ΓΙΑ ΝΑ ΜΑΘΕΙ ΤΟ 72-DIM VECTOR:
        # Ρίχνουμε "μαύρο" σε ένα ποσοστό (π.χ. 50%) των encoder outputs
        # ΜΟΝΟ κατά την εκπαίδευση. Έτσι το Attention δεν μπορεί να
        # "κλέψει" όλη την πληροφορία και το δίκτυο στηρίζεται στο hidden state!
        # -------------------------------------------------------------
        if self.training:
            encoder_outputs = F.dropout(encoder_outputs, p=0.5)
        
        # Attention χρησιμοποιώντας το πάνω-πάνω hidden layer
        top_hidden = hidden[-1]
        a = self.attention(top_hidden, encoder_outputs) # (Batch, Seq_Len)
        a = a.unsqueeze(1) # (Batch, 1, Seq_Len)
        
        # Context vector (σταθμισμένο άθροισμα των encoder outputs)
        context = torch.bmm(a, encoder_outputs) # (Batch, 1, 64)
        
        # 🔥 ΝΕΟ: Ετοιμάζουμε το latent_vector (από 2D σε 3D) για να μπει στο LSTM
        # Από (Batch, 72) γίνεται (Batch, 1, 72)
        latent_expanded = latent_vector.unsqueeze(1)
        
        # 🔥 ΝΕΟ: Συνένωση 3 στοιχείων: input token + context vector + latent vector!
        rnn_input = torch.cat((embedded, context, latent_expanded), dim=2)
        
        # Ένα βήμα του LSTM
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        # Ενώνουμε το output του LSTM (64) με το context του Attention (64) = 128
        combined = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc_out(combined)

        return prediction, hidden, cell