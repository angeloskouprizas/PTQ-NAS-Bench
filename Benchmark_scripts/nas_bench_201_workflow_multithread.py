import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms  #! ADDED: For proper normalization
from PIL import Image              #! ADDED: For conversion to a format compatible with transforms
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, parent_dir)

def load_cifar_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

def load_cifar10(data_dir="cifar-10-batches-py"):
    xs = []
    ys = []

    for i in range(1, 6):
        batch = load_cifar_batch(os.path.join(data_dir, f"data_batch_{i}"))
        xs.append(batch[b'data'])
        ys.extend(batch[b'labels'])

    X_train = np.concatenate(xs)
    y_train = np.array(ys)

    test_batch = load_cifar_batch(os.path.join(data_dir, "test_batch"))
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])

    #! CHANGED: Removed / 255.0 division here, it is now handled automatically by transforms.ToTensor()
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    return X_train, y_train, X_test, y_test

class CIFAR10FromFiles(Dataset):
    def __init__(self, data_dir="cifar-10-batches-py", train=True):
        if train:
            self.X, self.y, _, _ = load_cifar10(data_dir)
        else:
            _, _, self.X, self.y = load_cifar10(data_dir)

        #! CHANGED: Transpose from (C, H, W) to (H, W, C) so PIL/Transforms can read it
        self.X = self.X.transpose((0, 2, 3, 1))

        #! ADDED: The required Normalization (Mean/Std) for NATS-Bench/NAS-Bench-201
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts to [0, 1] and (C, H, W)
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        #! CHANGED: X remains numpy for __getitem__, y becomes a tensor
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        #! CHANGED: Apply transforms to each sample (Image -> Tensor -> Normalize)
        img = self.X[idx]
        img = Image.fromarray(img.astype('uint8')) # Convert to PIL Image
        img = self.transform(img) # ToTensor + Normalize
        
        return img, self.y[idx]
    

class PTQConfig:
    """Configuration class for PTQ parameters that can be easily modified"""
    def __init__(self):
        self.trainset = None
        self.full_trainset = None
        self.testset = None

        # Data loading
        self.workers = 8
        self.batch_size_calibration = 512 # 128
        self.batch_size_validation = 512 # 128
        self.calibration_samples = 10000 # 1024

        # Model and device
        self.gpu = 0
        self.dtype = 'float'
        
        # Export settings
        self.export_dir = '.'
        self.export_onnx_qcdq = False
        self.export_torch_qcdq = False
        self.onnx_opset_version = None
        
        # Quantization backend
        self.target_backend = 'fx'  # choices: 'fx', 'layerwise', 'flexml'
        
        # Bit widths
        self.act_bit_width = 8
        self.weight_bit_width = 8
        self.layerwise_first_last_bit_width = 8
        self.bias_bit_width = 32  # Can be 32, 16, or None
        
        # Float quantization bit widths
        self.weight_mantissa_bit_width = 4
        self.weight_exponent_bit_width = 3
        self.act_mantissa_bit_width = 4
        self.act_exponent_bit_width = 3
        self.layerwise_first_last_mantissa_bit_width = 4
        self.layerwise_first_last_exponent_bit_width = 3
        
        # Quantization types and granularity
        self.scale_factor_type = 'po2_scale'  # choices: 'float_scale', 'po2_scale'
        self.act_quant_type = 'sym'  # choices: 'sym', 'asym'
        self.weight_quant_type = 'sym'  # choices: 'sym', 'asym'
        self.weight_quant_granularity = 'per_tensor'  # choices: 'per_tensor', 'per_channel', 'per_group'
        self.act_quant_granularity = 'per_tensor'  # choices: 'per_tensor', 'per_group'
        self.quant_format = 'int'  # choices: 'int', 'float', 'float_ocp'
        
        # Calibration settings
        self.weight_quant_calibration_type = 'stats'  # choices: 'stats', 'mse', 'hqo'
        self.act_quant_calibration_type = 'stats'  # choices: 'stats', 'mse'
        self.act_quant_percentile = 99.999
        self.act_scale_computation_type = 'static'  # choices: 'static', 'dynamic'
        
        # Equalization and correction
        self.act_equalization = None  # choices: 'fx', 'layerwise', None
        self.graph_eq_iterations = 20
        self.graph_eq_merge_bias = True
        self.bias_corr = None
        self.weight_narrow_range = False
        
        # Advanced quantization methods
        self.gptq = False
        self.gpfq = False
        self.gpfq_p = 1.0
        self.qronos = False
        self.gpfa2q = False
        self.gpxq_act_order = False
        self.accumulator_bit_width = None
        
        # Learned round
        self.learned_round = None  # choices: None, 'linear_round', 'hard_sigmoid_round', 'sigmoid_round'
        self.learned_round_block_name = "layer\\d+"
        self.learned_round_loss = 'regularised_mse'  # choices: 'regularised_mse', 'mse'
        self.learned_round_mode = 'layerwise'  # choices: 'layerwise', 'blockwise'
        self.learned_round_iters = 1000
        self.learned_round_lr_scheduler = None  # choices: None, 'linear'
        self.learned_round_lr = 1e-3
        self.learned_round_batch_size = 1
        
        # Other settings
        self.calibrate_bn = False
        self.merge_bn = True
        self.uint_sym_act_for_unsigned_values = True
        self.compile = False
        self.channel_splitting_ratio = 0.0
        self.channel_splitting_split_input = False
        self.compression_rate = 0.0

    def update(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self
    
    def copy(self):
        """Create a copy of the configuration"""
        import copy
        return copy.deepcopy(self)
