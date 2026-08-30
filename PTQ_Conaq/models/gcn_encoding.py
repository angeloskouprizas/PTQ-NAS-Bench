import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """Simple Graph Convolution layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: (Batch, Nodes, Features)
        # adj: (Batch, Nodes, Nodes) normalized adjacency matrix
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class GCNEncoder(nn.Module):
    """
    GCN Encoding Scheme (According to Table 2 of the paper)
    - 2 Layers
    - 128 Hidden depth
    - 32 FC dimension
    """
    def __init__(self, num_node_features, hidden_depth=128, fc_dim=32):
        super().__init__()
        self.gc1 = GraphConvolution(num_node_features, hidden_depth)
        self.gc2 = GraphConvolution(hidden_depth, hidden_depth)
        self.fc = nn.Linear(hidden_depth, fc_dim)

    def forward(self, x, adj):
        # Pass through 2 GCN layers with ReLU
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        
        # Global Mean Pooling (compressing the entire graph into one vector)
        x_pooled = torch.mean(x, dim=1)
        
        # Final Dense Layer to reduce dimension to 32
        gcn_emb = self.fc(x_pooled)
        return gcn_emb