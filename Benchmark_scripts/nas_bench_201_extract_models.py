#!/usr/bin/env python3
import os
import re
import sys
import bz2
import json
import time
import glob
import torch
import pickle
import tarfile
import shutil
import argparse
import subprocess
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def available_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def save_model_only_package(model, arch_info, output_path, device):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        ensure_dir(out_dir)

    model_cpu = model.cpu()

    package = {
        "model": model_cpu,
        "arch_id": arch_info["arch_id"],
        "arch_str": arch_info["arch_str"],
        "channel": arch_info["channel"],
        "num_cells": arch_info["num_cells"],
        "class_num": arch_info["class_num"],
        "arch_config": arch_info["arch_config"],
        "dataset": arch_info["dataset"],
        "seed": arch_info["seed"],
        "epoch": arch_info["epoch"],
        "metadata": arch_info["metadata"],
        "pytorch_version": torch.__version__,
        "device": str(device),
    }
    torch.save(package, output_path)


def build_and_load_model(arch_str, C, N, num_classes, state_dict, device, CellStructure, get_cell_based_tiny_net):
    genotype = CellStructure.str2structure(arch_str)
    model = get_cell_based_tiny_net(
        {"name": "infer.tiny", "C": C, "N": N, "genotype": genotype, "num_classes": num_classes}
    ).to(torch.device(device))

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def flush_bucket_to_chunk(bucket_stage_dir, bucket_chunks_dir, bucket_id, chunk_index, manifest_path):
    files = sorted(glob.glob(os.path.join(bucket_stage_dir, "arch_*_complete.pth")))
    if not files:
        return chunk_index, 0

    first_name = os.path.basename(files[0])
    last_name = os.path.basename(files[-1])

    chunk_name = f"bucket_{bucket_id}_chunk_{chunk_index:05d}_{first_name}__{last_name}.tar.gz"
    chunk_path = os.path.join(bucket_chunks_dir, chunk_name)

    with tarfile.open(chunk_path, mode="w:gz") as tar:
        for fp in files:
            tar.add(fp, arcname=os.path.basename(fp))

    for fp in files:
        os.remove(fp)

    with open(manifest_path, "a", encoding="utf-8") as mf:
        mf.write(f"{datetime.now().isoformat()} | bucket={bucket_id} | chunk={chunk_name} | files={len(files)}\n")

    return chunk_index + 1, len(files)


def main():
    parser = argparse.ArgumentParser(description="One-pass NATS extractor with 3-way chunking")
    parser.add_argument("--project-root", type=str, default=".", help="Root directory of the project for module imports")
    parser.add_argument("--weights-folder", required=True, help="Folder containing NATS-tss-v1_0-3ffb9-full.tara*")
    parser.add_argument("--output-root", required=True, help="Root output folder for 3 buckets/chunks")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "ImageNet16-120"])
    parser.add_argument("--seed", type=int, default=777, choices=[777, 888, 999])
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=15624)
    parser.add_argument("--chunk-size", type=int, default=200, help="Flush bucket to tar.gz every N files")
    parser.add_argument("--min-free-gb", type=float, default=20.0, help="Stop safely if free space drops below this")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--progress-file", default="progress_done_ids.txt")
    parser.add_argument("--fail-file", default="progress_failed_ids.txt")
    args = parser.parse_args()

    # Dynamic imports
    sys.path.append(args.project_root)
    sys.path.append(os.path.join(args.project_root, "nas_benchmark/HW_NAS_Bench/hw_nas_bench_api"))

    from nas_benchmark.HW_NAS_Bench.hw_nas_bench_api.nas_201_models import CellStructure
    from nas_benchmark.HW_NAS_Bench.hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net

    ensure_dir(args.output_root)

    # Bucket layout
    bucket_stage = {}
    bucket_chunks = {}
    bucket_manifest = {}
    bucket_chunk_idx = {0: 0, 1: 0, 2: 0}

    for b in [0, 1, 2]:
        b_root = os.path.join(args.output_root, f"bucket_{b}")
        stg = os.path.join(b_root, "staging")
        chk = os.path.join(b_root, "chunks")
        man = os.path.join(b_root, "chunk_manifest.txt")
        ensure_dir(stg)
        ensure_dir(chk)
        bucket_stage[b] = stg
        bucket_chunks[b] = chk
        bucket_manifest[b] = man

    progress_path = os.path.join(args.output_root, args.progress_file)
    fail_path = os.path.join(args.output_root, args.fail_file)

    done_ids = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    done_ids.add(int(line))

    target_ids = set(range(args.start_id, args.end_id + 1))
    remaining_ids = target_ids - done_ids

    print(f"Total target IDs: {len(target_ids)}")
    print(f"Already done: {len(done_ids & target_ids)}")
    print(f"Remaining: {len(remaining_ids)}")

    if not remaining_ids:
        print("Nothing to do.")
        return

    # Build cat command for split tar shards
    file_pattern = os.path.join(args.weights_folder, "NATS-tss-v1_0-3ffb9-full.tara*")
    cmd = f"cat {file_pattern}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Match filenames like 000123.pickle.pbz2 (possibly with directory prefixes)
    rx = re.compile(r"(\d{6})\.pickle\.pbz2$")

    processed = 0
    failed = 0
    scanned_members = 0
    t0 = time.time()

    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                scanned_members += 1
                m = rx.search(os.path.basename(member.name))
                if not m:
                    continue

                arch_id = int(m.group(1))
                if arch_id not in remaining_ids:
                    continue

                if available_gb(args.output_root) < args.min_free_gb:
                    print(f"STOP: free space below threshold ({args.min_free_gb} GB).")
                    break

                try:
                    f = tar.extractfile(member)
                    if not f:
                        raise RuntimeError("tar.extractfile returned None")

                    content = bz2.decompress(f.read())
                    full_weights = pickle.loads(content)

                    epoch_key = str(args.epoch)
                    if epoch_key not in full_weights:
                        raise KeyError(f"epoch {args.epoch} missing")

                    epoch_data = full_weights[epoch_key]
                    results = epoch_data["all_results"]
                    target_key = (args.dataset, args.seed)
                    if target_key not in results:
                        raise KeyError(f"dataset/seed missing: {target_key}")

                    data = results[target_key]
                    arch_config = data["arch_config"]
                    arch_str = arch_config["arch_str"]
                    C = arch_config["channel"]
                    N = arch_config["num_cells"]
                    num_classes = arch_config["class_num"]
                    state_dict = data["net_state_dict"]

                    model = build_and_load_model(
                        arch_str=arch_str,
                        C=C,
                        N=N,
                        num_classes=num_classes,
                        state_dict=state_dict,
                        device=args.device,
                        CellStructure=CellStructure,
                        get_cell_based_tiny_net=get_cell_based_tiny_net,
                    )

                    arch_info = {
                        "arch_id": arch_id,
                        "arch_str": arch_str,
                        "channel": C,
                        "num_cells": N,
                        "class_num": num_classes,
                        "arch_config": arch_config,
                        "dataset": args.dataset,
                        "seed": args.seed,
                        "epoch": args.epoch,
                        "metadata": {
                            "name": data.get("name", "unknown"),
                            "params": data.get("params", "unknown"),
                            "flop": data.get("flop", "unknown"),
                            "latency": data.get("latency", "unknown"),
                        },
                    }

                    bucket = arch_id % 3
                    out_name = f"arch_{arch_id:06d}_complete.pth"
                    out_path = os.path.join(bucket_stage[bucket], out_name)

                    save_model_only_package(model, arch_info, out_path, args.device)

                    with open(progress_path, "a", encoding="utf-8") as pf:
                        pf.write(f"{arch_id}\n")

                    processed += 1
                    if processed % 50 == 0:
                        dt = (time.time() - t0) / 60.0
                        free = available_gb(args.output_root)
                        print(f"Processed={processed}, Failed={failed}, Minutes={dt:.1f}, FreeGB={free:.1f}")

                    # Flush this bucket if chunk size reached
                    staged = glob.glob(os.path.join(bucket_stage[bucket], "arch_*_complete.pth"))
                    if len(staged) >= args.chunk_size:
                        bucket_chunk_idx[bucket], n = flush_bucket_to_chunk(
                            bucket_stage_dir=bucket_stage[bucket],
                            bucket_chunks_dir=bucket_chunks[bucket],
                            bucket_id=bucket,
                            chunk_index=bucket_chunk_idx[bucket],
                            manifest_path=bucket_manifest[bucket],
                        )
                        print(f"Flushed bucket {bucket}: {n} files to chunk index {bucket_chunk_idx[bucket]-1}")

                except Exception as e:
                    failed += 1
                    with open(fail_path, "a", encoding="utf-8") as ff:
                        ff.write(f"{arch_id}\t{type(e).__name__}\t{str(e)}\n")

                remaining_ids.discard(arch_id)
                if not remaining_ids:
                    print("All requested IDs are done.")
                    break

    finally:
        try:
            proc.kill()
        except Exception:
            pass

    # Final flush of all buckets
    for b in [0, 1, 2]:
        bucket_chunk_idx[b], n = flush_bucket_to_chunk(
            bucket_stage_dir=bucket_stage[b],
            bucket_chunks_dir=bucket_chunks[b],
            bucket_id=b,
            chunk_index=bucket_chunk_idx[b],
            manifest_path=bucket_manifest[b],
        )
        if n > 0:
            print(f"Final flush bucket {b}: {n} files")

    dt = (time.time() - t0) / 60.0
    print("=" * 70)
    print(f"Done. Processed={processed}, Failed={failed}, ScannedMembers={scanned_members}, Minutes={dt:.2f}")
    print(f"Progress file: {progress_path}")
    print(f"Failure file:  {fail_path}")
    print("Output per machine in:")
    print(f"  {os.path.join(args.output_root, 'bucket_0', 'chunks')}")
    print(f"  {os.path.join(args.output_root, 'bucket_1', 'chunks')}")
    print(f"  {os.path.join(args.output_root, 'bucket_2', 'chunks')}")
    print("=" * 70)


if __name__ == "__main__":
    main()