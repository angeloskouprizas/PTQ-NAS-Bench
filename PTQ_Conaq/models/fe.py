import torch
from torchinfo import summary


class FE:
    def __init__(self, arch=None) -> None:
        self.arch = arch
        self.features_names = ["depth", "nb_conv", "FLOPs", "nb_param", "input_size", "1_channel", "last_channel", "nb_down_sample"]
        self.features = {}

    def extract_features(self, input_size=(1, 3, 32, 32)):
        """
        Extract architectural features as a numerical vector.
        
        Args:
            input_size: Input tensor size (batch, channels, height, width)
        
        Returns:
            torch.Tensor: Feature vector of shape (8,)
        """
        stats = summary(self.arch, input_size=input_size, verbose=0)
        
        # Extract features in one pass
        depth = max(layer.depth for layer in stats.summary_list)
        nb_conv = sum(1 for layer in stats.summary_list if 'Conv2d' in str(layer.class_name))
        nb_down_sample = sum(1 for layer in stats.summary_list if 'ResNetBasicblock' in str(layer.class_name))
        
        # Get first channel (output of first Conv2d)
        first_channel = next((layer.output_size[1] for layer in stats.summary_list 
                              if 'Conv2d' in str(layer.class_name) and hasattr(layer, 'output_size') and layer.output_size), 16)
        
        # Get last channel (input to Linear layer)
        last_channel = next((layer.input_size[-1] if isinstance(layer.input_size, (list, tuple)) else layer.input_size
                             for layer in reversed(stats.summary_list) 
                             if 'Linear' in str(layer.class_name) and hasattr(layer, 'input_size') and layer.input_size), 64)
        
        # Store in dictionary
        self.features = {
            'depth': depth,
            'nb_conv': nb_conv,
            'FLOPs': stats.total_mult_adds,
            'nb_param': stats.total_params,
            'input_size': input_size[2],
            '1_channel': first_channel,
            'last_channel': last_channel,
            'nb_down_sample': nb_down_sample
        }
        
        flops_log = torch.log10(torch.tensor(stats.total_mult_adds, dtype=torch.float32) + 1)
        params_log = torch.log10(torch.tensor(stats.total_params, dtype=torch.float32) + 1)
        
        # Return as tensor
        feature_vector = torch.tensor([
            float(depth), 
            float(nb_conv), 
            float(flops_log),      # Instead of stats.total_mult_adds
            float(params_log),     # Instead of stats.total_params
            float(input_size[2]), 
            float(first_channel), 
            float(last_channel), 
            float(nb_down_sample)
        ], dtype=torch.float32)
        
        return feature_vector