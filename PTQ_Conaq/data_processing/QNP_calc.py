import os
import sys
import re
import bz2
import csv
import time
import glob
import torch
import pickle
import tarfile
import argparse
import subprocess
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
import numpy as np
import gc

# ============================================================================
# 1. Configuration & QNP Calculation Functions
# ============================================================================

def get_scheme_bits(scheme: str):
    """
    Returns the weight and activation bitwidths for a given PTQ scheme.
    Bitwidths are aligned with the PTQ-NAS-Bench paper specifications.
    """
    scheme_bits = {
        "Q1": (8, 8),  # INT8w8a
        "Q2": (4, 4),  # INT4w4a
        "Q3": (2, 8),  # INT2w8a
        "Q4": (2, 2),  # INT2w2a
    }
    return scheme_bits[scheme]

class ActivationHook:
    def __init__(self):
        self.activations = {}
        self.gradients   = {}
        self._handles    = []

    def register(self, model: nn.Module, target_types=(nn.Conv2d, nn.Linear)):
        for name, module in model.named_modules():
            if isinstance(module, target_types):
                self._handles.append(
                    module.register_forward_hook(
                        lambda m, inp, out, n=name: self._fwd(n, out)
                    )
                )
                self._handles.append(
                    module.register_backward_hook(
                        lambda m, gin, gout, n=name: self._bwd(n, gout[0])
                    )
                )

    def _fwd(self, name, output):
        self.activations[name] = output

    def _bwd(self, name, grad):
        self.gradients[name] = grad

    def remove(self):
        for h in self._handles:
            h.remove()
        # FIX: Fully clear the hook dictionaries to break references
        self._handles.clear()
        self.activations.clear()
        self.gradients.clear()

def compute_qnp(model: nn.Module, data_loader, scheme: str, device: str = "cpu", n_batches: int = 1) -> float:
    model.eval()
    model.to(device)

    bw, ba = get_scheme_bits(scheme)
    denom_w = (2 ** (bw - 1)) - 1
    denom_a = (2 ** (ba - 1)) - 1

    hook = ActivationHook()
    hook.register(model)

    criterion = nn.CrossEntropyLoss()
    total_qnp = 0.0
    batches_processed = 0

    for i, (images, labels) in enumerate(data_loader):
        if i >= n_batches: break
        
        # FIX: CLONE(). Fully isolates it from the initial fixed_batch
        images = images.clone().to(device)
        labels = labels.clone().to(device)

        model.zero_grad()
        logits = model(images)
        if isinstance(logits, tuple): logits = logits[1] 

        loss = criterion(logits, labels)
        loss.backward()

        batch_qnp = 0.0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # --- Activations ---
                if name in hook.activations and name in hook.gradients:
                    act = hook.activations[name].detach()
                    grad_a = hook.gradients[name].detach()
                    
                    max_a = act.abs().max().item()
                    F_a = (grad_a ** 2).mean().item()
                    delta_a = max_a / max(denom_a, 1e-8)
                    noise_a = (delta_a ** 2) / 12.0
                else:
                    F_a, noise_a = 0.0, 0.0

                # --- Weights ---
                if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                    weight = module.weight.detach()
                    grad_w = module.weight.grad.detach()
                    
                    max_w = weight.abs().max().item()
                    F_w = (grad_w ** 2).mean().item()
                    delta_w = max_w / max(denom_w, 1e-8)
                    noise_w = (delta_w ** 2) / 12.0
                else:
                    F_w, noise_w = 0.0, 0.0

                layer_qnp = (F_w * noise_w) + (F_a * noise_a)
                batch_qnp += layer_qnp

        total_qnp += batch_qnp
        batches_processed += 1
        
        # FIX: Clean up PyTorch Graph residuals
        model.zero_grad() 
        del images, labels, logits, loss

    hook.remove()
    if batches_processed == 0: return 0.0
    return total_qnp / batches_processed


# ============================================================================
# 2. CIFAR-10 Loader
# ============================================================================

def get_cifar10_loader(batch_size: int = 64, data_dir: str = './data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


# ============================================================================
# 3. Main Extraction Loop with In-Memory Streaming
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Extract QNP values from NATS-Bench streaming weights.")
    parser.add_argument('--naszilla_dir', type=str, required=True, help="Path to the naszilla directory.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory to store/load CIFAR-10 data.")
    parser.add_argument('--weights_pattern', type=str, required=True, help="File pattern for NATS-Bench weights tar file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the resulting CSV.")
    parser.add_argument('--scheme', type=str, choices=['Q1', 'Q2', 'Q3', 'Q4'], required=True, help="Quantization scheme.")
    parser.add_argument('--target_epoch', type=str, default='200', help="Target epoch to evaluate.")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Append naszilla to path dynamically
    sys.path.append(args.naszilla_dir)
    sys.path.append(os.path.dirname(args.naszilla_dir))
    
    # Import NAS-Bench API locally after appending to path
    from naszilla.nas_benchmarks import Nasbench201
    from nas_benchmark.HW_NAS_Bench.hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net

    device = 'cpu' 
    dataset_name = 'cifar10'
    TARGET_SEED = 777
    
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("[*] Loading NAS-Bench-201 API...")
    nasbench = Nasbench201(dataset=dataset_name)
    api = nasbench.nasbench

    os.makedirs(args.data_dir, exist_ok=True)
    real_loader = get_cifar10_loader(batch_size=64, data_dir=args.data_dir)
    
    print("[*] Extracting the fixed Data Batch...")
    data_iterator = iter(real_loader)
    fixed_images, fixed_labels = next(data_iterator)
    fixed_batch = [(fixed_images, fixed_labels)] 

    os.makedirs(args.output_dir, exist_ok=True) 
    csv_path = os.path.join(args.output_dir, f"nas201_qnp_{args.scheme}_REAL_WEIGHTS.csv")
    
    cmd = f"cat {args.weights_pattern}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rx = re.compile(r"(\d{6})\.pickle\.pbz2$")
    
    print(f"[*] Opening Tar Stream: {cmd}")
    
    processed_count = 0
    failed_count = 0
    seen_archs = set()

    try:
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["model_id", "QNP"])
            
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    '''
                    if processed_count >= 20:
                        print("\n[!] 20-sample test completed! Stopping read.")
                        break
                    '''
                    
                    m = rx.search(os.path.basename(member.name))
                    if not m:
                        continue

                    arch_id = int(m.group(1))
                    if arch_id in seen_archs:
                        continue
                    
                    # FIX: Initialize variables for safe deletion in finally block
                    f = content = full_weights = epoch_data = results = state_dict = model = None
                    
                    try:
                        f = tar.extractfile(member)
                        if not f: continue
                        
                        content = bz2.decompress(f.read())
                        f.close() # FIX: Close the stream
                        
                        full_weights = pickle.loads(content)
                        
                        if args.target_epoch not in full_weights: continue
                        epoch_data = full_weights[args.target_epoch]
                        
                        results = epoch_data.get("all_results", {})
                        target_key = (dataset_name, TARGET_SEED)
                        
                        if target_key not in results: continue
                        state_dict = results[target_key]["net_state_dict"]

                        if any(k.startswith("module.") for k in state_dict.keys()):
                            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

                        config = api.get_net_config(arch_id, dataset_name)
                        model = get_cell_based_tiny_net(config)
                        
                        try:
                            model.load_state_dict(state_dict, strict=True)
                        except RuntimeError:
                            model.load_state_dict(state_dict, strict=False)

                        qnp_value = compute_qnp(
                            model=model, 
                            data_loader=fixed_batch, 
                            scheme=args.scheme, 
                            device=device, 
                            n_batches=1
                        )
                        
                        writer.writerow([arch_id, qnp_value])
                        csv_file.flush()
                        
                        seen_archs.add(arch_id)
                        processed_count += 1
                        
                        if processed_count % 50 == 0:
                            print(f"[+] Successfully processed {processed_count} models...")

                    except Exception as e:
                        print(f"[-] Error at ID {arch_id}: {str(e)}")
                        failed_count += 1
                        writer.writerow([arch_id, f"ERROR: {str(e)}"])
                        csv_file.flush()
                        
                    finally:
                        # FIX: Absolute Cleanup. ALWAYS runs even on continue or exception
                        if f is not None and not f.closed: 
                            f.close()
                        
                        # Kill large objects
                        del f, content, full_weights, epoch_data, results, state_dict, model
                        
                        # MAGIC BULLET: Clear tarfile's hidden history!
                        tar.members = [] 
                        
                        gc.collect()

    finally:
        try:
            proc.kill()
        except Exception:
            pass

    print("=" * 60)
    print(f"Process Completed! Success: {processed_count} | Failed: {failed_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()