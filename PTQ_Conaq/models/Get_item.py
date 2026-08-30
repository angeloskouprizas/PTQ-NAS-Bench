import sys
from naszilla.nas_bench_201.cell_201 import Cell201

class ArchTokenizer:
    """
    Minimal tokenizer for NAS-Bench-201 architectures.
    Only encodes operations to token indices for decoder training.
    """
    
    def __init__(self):
        """Initialize tokenizer with vocabulary for NAS-Bench-201 operations."""
        # Define all possible operations in NAS-Bench-201
        self.operations = [
            'avg_pool_3x3',
            'nor_conv_1x1', 
            'nor_conv_3x3',
            'none',
            'skip_connect'
        ]
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2
        }
        
        # Build vocabulary: special tokens + operations
        self.token2idx = self.special_tokens.copy()
        current_idx = len(self.special_tokens)
        for op in self.operations:
            self.token2idx[op] = current_idx
            current_idx += 1
        
        # Vocabulary size
        self.vocab_size = len(self.token2idx)
    
    def encode(self, arch_string, add_special_tokens=True):
        """
        Convert architecture string to sequence of token indices.
        
        Args:
            arch_string: NAS-Bench-201 architecture string
            add_special_tokens: whether to add <START> and <END> tokens
        
        Returns:
            list of token indices
        """
        # Get operation list using Cell201
        cell = Cell201(arch_string)
        ops = cell.get_op_list()
        
        # Convert operations to indices
        indices = []
        
        if add_special_tokens:
            indices.append(self.token2idx['<START>'])
        
        for op in ops:
            if op in self.token2idx:
                indices.append(self.token2idx[op])
        
        if add_special_tokens:
            indices.append(self.token2idx['<END>'])
        
        return indices
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        return self.vocab_size