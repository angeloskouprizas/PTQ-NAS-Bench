"""
NAS-Bench-201 dataset extractor for LSTM/GCN training.
"""

import sys
import torch
from torch.utils.data import Dataset
import numpy as np

from naszilla.nas_benchmarks import Nasbench201
from naszilla.nas_bench_201.cell_201 import Cell201

# --- ΕΝΗΜΕΡΩΜΕΝΑ IMPORTS ---
from PTQ_Conaq.models.Encoder.Get_item import ArchTokenizer
from PTQ_Conaq.utils.pred_utils_fp32 import get_metrics


class ArchExtractor:
    """Extracts ops, adjacency, and tokens from NAS-Bench-201 architectures."""
    
    def __init__(self, dataset='cifar10', nas_api=None):
        # Reuse provided API; only create if not provided
        if nas_api is not None:
            print("ArchExtractor: Using provided NAS-Bench-201 API")
            self.nb201 = nas_api
        else:
            print("ArchExtractor: Creating new NAS-Bench-201 API")
            self.nb201 = Nasbench201(dataset=dataset)
        
        # Expose the underlying API for compatibility
        self.api = self.nb201.nasbench
        
        self.tokenizer = ArchTokenizer()
        self.pad_idx = self.tokenizer.token2idx['<PAD>']
    
    def extract(self, arch_strings):
        """Extract (ops, adj, tokens) from single string or list of strings."""
        is_single = isinstance(arch_strings, str)
        if is_single:
            arch_strings = [arch_strings]
        
        ops_list, adj_list, tokens_list = [], [], []
        
        for arch_str in arch_strings:
            cell = Cell201(arch_str)
            gcn_encoding = cell.gcn_encoding(self.nb201.nasbench, deterministic=True)
            
            ops_list.append(gcn_encoding['operations'])
            adj_list.append(gcn_encoding['adjacency'])
            tokens_list.append(self.tokenizer.encode(arch_str))
        
        if is_single:
            return ops_list[0], adj_list[0], tokens_list[0]
        return ops_list, adj_list, tokens_list
    
    def get_arch_by_index(self, index):
        """Get architecture string by index (0-15624)."""
        return self.nb201.nasbench.query_by_index(index).arch_str


class NasBench201Dataset(Dataset):
    """
    PyTorch Dataset for NAS-Bench-201 architectures.
    Returns architecture string, ops encoding, adjacency matrix, token sequence, AND LATENCY.
    """
    
    def __init__(self, dataset_name='cifar10', num_samples=None, shuffle=False, api=None, device_name='edgegpu'):
        """
        Args:
            dataset_name: 'cifar10', 'cifar100', or 'ImageNet16-120'
            num_samples: Number of architectures to sample (None = all 15625)
            shuffle: Whether to shuffle indices
            api: Optional pre-loaded NAS-Bench-201 API instance (Nasbench201 wrapper)
            device_name: 'edgegpu', 'raspi4', 'edgetpu', 'pixel3', 'eyeriss', 'fpga'
        """
        # Use provided API or create new one via ArchExtractor
        if api is not None:
            print("NasBench201Dataset: Using provided NAS-Bench-201 API")
            self.extractor = ArchExtractor(dataset_name, nas_api=api)
        else:
            print("NasBench201Dataset: Creating new NAS-Bench-201 API")
            self.extractor = ArchExtractor(dataset_name)
        
        self.dataset_name = dataset_name
        self.device_name = device_name
        self.api = self.extractor.api
        
        # Sample indices
        total_archs = len(self.api)
        if num_samples is None or num_samples > total_archs:
            num_samples = total_archs
        
        self.indices = list(range(total_archs))
        if shuffle:
            import random
            random.shuffle(self.indices)
        
        self.indices = self.indices[:num_samples]
        
        # Load HW metrics into memory once for speed
        print(f"NasBench201Dataset: Loading HW metrics for device '{self.device_name}'...")
        self.csv_indices, _, self.csv_latencies, _ = get_metrics(self.device_name)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        arch_str = self.extractor.get_arch_by_index(idx)
        ops, adj, tokens = self.extractor.extract(arch_str)
        
        # Find Latency for the current architecture
        arch_idx = self.api.query_index_by_arch(arch_str)
        idx_pos = np.where(self.csv_indices == arch_idx)[0]
        
        if len(idx_pos) > 0:
            latency = float(self.csv_latencies[idx_pos[0]])
        else:
            latency = 0.0 # Fallback (if not found)
            
        return arch_str, ops, adj, torch.LongTensor(tokens), latency
    

def collate_fn(batch, device='cpu'):
    """Collate batch into tensors with padding."""
    # Zip lat_list along with other elements
    arch_strings, ops_list, adj_list, tokens_list, lat_list = zip(*batch)
    
    ops_batch = torch.stack([torch.from_numpy(ops).float() for ops in ops_list]).to(device)
    adj_batch = torch.stack([torch.from_numpy(adj).float() for adj in adj_list]).to(device)
    
    tokens_batch = torch.nn.utils.rnn.pad_sequence(
        list(tokens_list), batch_first=True, padding_value=0
    ).to(device)
    
    # Stack latencies into a Tensor
    latencies_batch = torch.tensor(lat_list, dtype=torch.float32).to(device)
    
    # Returning 5 elements
    return list(arch_strings), ops_batch, adj_batch, tokens_batch, latencies_batch
