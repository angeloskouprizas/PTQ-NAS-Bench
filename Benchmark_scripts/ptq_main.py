# import pdb
# import argparse
from functools import partial
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torch.utils.data import DataLoader

# from brevitas.export import export_onnx_qcdq
# from brevitas.export import export_torch_qcdq
from brevitas.export import export_qonnx
from brevitas.export.inference import quant_inference_mode
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_act_equalization
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gpfq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gptq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_qronos
# from brevitas_examples.imagenet_classification.ptq.learned_round_utils import apply_learned_round
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate_bn
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
# from brevitas_examples.common.parse_utils import add_bool_arg
# from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
# from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
# from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

# Ignore warnings about __torch_function__
warnings.filterwarnings("ignore")

def generate_ref_input(center_crop_shape, device, dtype):
    img_shape = center_crop_shape
    return torch.ones(1, 3, img_shape, img_shape, device=device, dtype=dtype)


def ptq_main(arch_id, args):
    dtype = getattr(torch, args.dtype)
    size = center_crop_shape = 32

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.act_quant_calibration_type == 'stats':
        act_quant_calib_config = str(args.act_quant_percentile) + 'stats'
    else:
        act_quant_calib_config = args.act_quant_calibration_type

    if args.act_bit_width == 0:
        args.act_bit_width = None

    config = (
        # f"{args.model_name}_"
        f"{args.target_backend}_"
        f"{args.quant_format}_"
        f"{str(args.weight_mantissa_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{str(args.weight_exponent_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{str(args.act_mantissa_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{str(args.act_exponent_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{args.scale_factor_type}_"
        f"a{args.act_bit_width}"
        f"w{args.weight_bit_width}_"
        f"{'gptq_' if args.gptq else ''}"
        f"{'gpfq_' if args.gpfq else ''}"
        f"{'gpfa2q_' if args.gpfa2q else ''}"
        f"{'gpxq_act_order_' if args.gpxq_act_order else ''}"
        f"{'learned_round_' if args.learned_round else ''}"
        f"{'weight_narrow_range_' if args.weight_narrow_range else ''}"
        f"{args.bias_bit_width}bias_"
        f"{args.weight_quant_granularity}_"
        f"{args.act_quant_type}_"
        f"{'bc_' if args.bias_corr else ''}"
        f"{args.graph_eq_iterations}geiters_"
        f"{'mb_' if args.graph_eq_merge_bias else ''}"
        f"{act_quant_calib_config}_"
        f"{args.weight_quant_calibration_type}_"
        f"{'bnc_' if args.calibrate_bn else ''}"
        f"{'channel_splitting' if args.channel_splitting_ratio else ''}")

    print(
        # f"Model: {args.model_name} - "
        f"Target backend: {args.target_backend} - "
        f"Quantization type: {args.scale_factor_type} - "
        f"Activation bit width: {args.act_bit_width} - "
        f"Weight bit width: {args.weight_bit_width} - "
        f"GPTQ: {args.gptq} - "
        f"GPFQ: {args.gpfq} - "
        f"GPFA2Q: {args.gpfa2q} - "
        f"GPFQ P: {args.gpfq_p} - "
        f"GPxQ Act Order: {args.gpxq_act_order} - "
        f"GPFA2Q Accumulator Bit Width: {args.accumulator_bit_width} - "
        f"Learned Round: {args.learned_round} - "
        f"Weight narrow range: {args.weight_narrow_range} - "
        f"Bias bit width: {args.bias_bit_width} - "
        f"Weight scale factors type: {args.weight_quant_granularity} - "
        f"Activation scale factors type: {args.act_quant_granularity} - "
        f"Activation quant type: {args.act_quant_type} - "
        f"Bias Correction Enabled: {args.bias_corr} - "
        f"Iterations for graph equalization: {args.graph_eq_iterations} - "
        f"Merge bias in graph equalization: {args.graph_eq_merge_bias} - "
        f"Activation quant calibration type: {act_quant_calib_config} - "
        f"Weight quant calibration type: {args.weight_quant_calibration_type} - "
        f"Calibrate BN: {args.calibrate_bn} - "
        f"Channel Splitting Ratio: {args.channel_splitting_ratio} - "
        f"Split Input: {args.channel_splitting_split_input} - "
        f"Merge BN: {args.merge_bn}")

    calib_loader = DataLoader(args.trainset, args.batch_size_calibration, shuffle=True)
    val_loader = DataLoader(args.testset, args.batch_size_validation, shuffle=False)
    
    # The first parameter of ptq_main is the arch_id
    arch_id_int = int(arch_id) 
    
    # Check if a dynamic staging_dir was passed via args from the caller script,
    # otherwise fallback to the hardcoded local relative path.
    if hasattr(args, 'staging_dir') and args.staging_dir:
        checkpoint_path = os.path.join(args.staging_dir, f'arch_{arch_id_int:06d}_complete.pth')
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Dynamic path fallback to three_way_out/bucket_0/staging
        checkpoint_path = os.path.join(script_dir, '..', 'PTQ_Benchmark', 'three_way_out', 'bucket_0','staging', f'arch_{arch_id_int:06d}_complete.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[-] Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = checkpoint['model'] 
    model.eval()

    # Preprocess the model for quantization
    if args.target_backend == 'flexml':
        # flexml requires static shapes, pass a representative input in
        img_shape = center_crop_shape
        model = preprocess_for_flexml_quantize(
            model,
            torch.ones(1, 3, img_shape, img_shape, dtype=dtype),
            equalize_iters=args.graph_eq_iterations,
            equalize_merge_bias=args.graph_eq_merge_bias,
            merge_bn=not args.calibrate_bn)
    elif args.target_backend == 'fx' or args.target_backend == 'layerwise':
        model = preprocess_for_quantize(
            model,
            equalize_iters=args.graph_eq_iterations,
            equalize_merge_bias=args.graph_eq_merge_bias,
            merge_bn=args.merge_bn,
            channel_splitting_ratio=args.channel_splitting_ratio,
            channel_splitting_split_input=args.channel_splitting_split_input)
    else:
        raise RuntimeError(f"{args.target_backend} backend not supported.")

    # If available, use the selected GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = False

    if args.act_equalization is not None:
        print("Applying activation equalization:")
        apply_act_equalization(model, calib_loader, layerwise=args.act_equalization == 'layerwise')
    device = next(iter(model.parameters())).device

    # Define the quantized model
    quant_model = quantize_model(
        model,
        dtype=dtype,
        device=device,
        backend=args.target_backend,
        scale_factor_type=args.scale_factor_type,
        bias_bit_width=args.bias_bit_width,
        weight_bit_width=args.weight_bit_width,
        weight_narrow_range=args.weight_narrow_range,
        weight_param_method=args.weight_quant_calibration_type,
        weight_quant_granularity=args.weight_quant_granularity,
        act_quant_granularity=args.act_quant_granularity,
        weight_quant_type=args.weight_quant_type,
        layerwise_first_last_bit_width=args.layerwise_first_last_bit_width,
        act_bit_width=args.act_bit_width,
        act_param_method=args.act_quant_calibration_type,
        act_quant_percentile=args.act_quant_percentile,
        act_quant_type=args.act_quant_type,
        quant_format=args.quant_format,
        layerwise_first_last_mantissa_bit_width=args.layerwise_first_last_mantissa_bit_width,
        layerwise_first_last_exponent_bit_width=args.layerwise_first_last_exponent_bit_width,
        weight_mantissa_bit_width=args.weight_mantissa_bit_width,
        weight_exponent_bit_width=args.weight_exponent_bit_width,
        act_mantissa_bit_width=args.act_mantissa_bit_width,
        act_exponent_bit_width=args.act_exponent_bit_width,
        act_scale_computation_type=args.act_scale_computation_type,
        uint_sym_act_for_unsigned_values=args.uint_sym_act_for_unsigned_values)

    # Some quantizer configurations require a forward pass to initialize scale factors.
    # This forward pass ensures that subsequent algorithms work as intended
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    images, _ = next(iter(calib_loader))
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        model(images)

    if args.act_scale_computation_type == 'static':
        # Calibrate the quant_model on the calibration dataloader
        print("Starting activation calibration:")
        calibrate(calib_loader, quant_model)

    if args.gpfq:
        print("Performing GPFQ:")
        apply_gpfq(
            calib_loader,
            quant_model,
            p=args.gpfq_p,
            act_order=args.gpxq_act_order,
            compression_rate=args.compression_rate)

    if args.gpfa2q:
        print("Performing GPFA2Q:")
        apply_gpfq(
            calib_loader,
            quant_model,
            p=args.gpfq_p,
            act_order=args.gpxq_act_order,
            use_gpfa2q=args.gpfa2q,
            accumulator_bit_width=args.accumulator_bit_width,
            compression_rate=args.compression_rate)

    if args.gptq:
        print("Performing GPTQ:")
        apply_gptq(calib_loader, quant_model, act_order=args.gpxq_act_order)

    if args.qronos:
        print("Performing Qronos:")
        apply_qronos(quant_model, calib_loader, act_order=args.gpxq_act_order)

    # if args.learned_round:
    #     print("Applying Learned Round:")
    #     apply_learned_round(
    #         model=quant_model,
    #         calibration_loader=calib_loader,
    #         iters=args.learned_round_iters,
    #         learned_round=args.learned_round,
    #         learned_round_loss=args.learned_round_loss,
    #         block_name_attribute=args.learned_round_block_name,
    #         optimizer=args.optimizer,
    #         lr_scheduler=args.learned_round_lr_scheduler,
    #         optimizer_kwargs={'lr': args.learned_round_lr},
    #         batch_size=args.learned_round_batch_size,
    #         learned_round_mode=args.learned_round_mode,
    #     )

    if args.calibrate_bn:
        print("Calibrate BN:")
        calibrate_bn(calib_loader, quant_model)

    if args.bias_corr:
        print("Applying bias correction:")
        apply_bias_correction(calib_loader, quant_model)

    # qonnx_filename = f"model_{arch_id}.onnx"
    # qonnx_path = os.path.join("qonnx_models", qonnx_filename)

    quant_model.eval()
    print("Starting validation (pre-finetuning):")
    with torch.no_grad(), quant_inference_mode(quant_model):
        param = next(iter(quant_model.parameters()))
        device, dtype = param.device, param.dtype
        ref_input = generate_ref_input(center_crop_shape, device, dtype)
        quant_model(ref_input)
        compiled_model = torch.compile(quant_model, fullgraph=True, disable=not args.compile)
        top1_accuracy_pre = validate(val_loader, compiled_model, stable=dtype != torch.bfloat16)

    # =====================================================================
    # FINE-TUNING (QUANTIZATION-AWARE TRAINING) - 2 EPOCHS
    # =====================================================================
    print("\n[🚀] Starting Fine-Tuning (QAT) for 2 epochs...")
    
    # Set model to training mode
    quant_model.train()
    
    # Define Loss Function and Optimizer (small Learning Rate for fine-tuning)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(quant_model.parameters(), lr=1e-4)
    
    # Create DataLoader for training. 
    # Using the full trainset (not just the calibration subset if specified)
    train_loader = DataLoader(
        args.full_trainset, 
        batch_size=args.batch_size_calibration, 
        shuffle=True, 
        num_workers=getattr(args, 'workers', 8)
    )
    
    epochs = 2
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device=device, dtype=dtype)
            labels = labels.to(device=device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = quant_model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and weight update
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"   ➤ Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
        
    print("[+] Fine-tuning completed.\n")
    
    # Restore model to evaluation mode (eval) for inference
    quant_model.eval()
    # =====================================================================

    # export_qonnx(quant_model, torch.randn(1, 3, size, size), qonnx_path)

    # Validate the quant_model on the validation dataloader
    print("Starting validation (post-finetuning):")
    with torch.no_grad(), quant_inference_mode(quant_model):
        param = next(iter(quant_model.parameters()))
        device, dtype = param.device, param.dtype
        ref_input = generate_ref_input(center_crop_shape, device, dtype)
        quant_model(ref_input)
        compiled_model = torch.compile(quant_model, fullgraph=True, disable=not args.compile)
        top1_accuracy = validate(val_loader, compiled_model, stable=dtype != torch.bfloat16)

    # if args.export_onnx_qcdq or args.export_torch_qcdq:
    #     # Generate reference input tensor to drive the export process
    #     param = next(iter(quant_model.parameters()))
    #     device, dtype = param.device, param.dtype
    #     ref_input = generate_ref_input(center_crop_shape, device, dtype)

    #     export_name = os.path.join(args.export_dir, config)
    #     if args.export_onnx_qcdq:
    #         export_name = export_name + '.onnx'
    #         export_onnx_qcdq(
    #             quant_model, ref_input, export_name, opset_version=args.onnx_opset_version)
    #     if args.export_torch_qcdq:
    #         export_name = export_name + '.pt'
    #         export_torch_qcdq(quant_model, ref_input, export_name)

    return top1_accuracy_pre, top1_accuracy, act_quant_calib_config