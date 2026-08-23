#!/usr/bin/env python3
import csv
import os
import re
import glob
import shutil
import tarfile
import argparse
import sys
import torch

ARCH_RE = re.compile(r"arch_(\d+)_complete\.pth$")
CHUNK_RE = re.compile(
    r"bucket_(\d+)_chunk_(\d+)_arch_(\d+)_complete\.pth__arch_(\d+)_complete\.pth\.tar\.gz$"
)

def clear_dir(path):
    if not os.path.exists(path):
        return
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and not os.path.islink(full_path):
            shutil.rmtree(full_path)
        else:
            try:
                os.remove(full_path)
            except FileNotFoundError:
                pass

def load_local_arch_ids(snapshots_dir):
    arch_ids = []
    for file_path in glob.glob(os.path.join(snapshots_dir, "arch_*_complete.pth")):
        filename = os.path.basename(file_path)
        m = ARCH_RE.match(filename)
        if m:
            arch_ids.append(int(m.group(1)))
    return sorted(set(arch_ids))

def safe_extract_tar(tar_path, extract_dir):
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extract_dir, member.name))
            if not member_path.startswith(os.path.abspath(extract_dir) + os.sep):
                raise RuntimeError(f"Blocked unsafe path in tar: {member.name}")
        tar.extractall(extract_dir)

def load_done_ids(progress_file):
    done_ids = set()
    done_in_order = []

    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.isdigit():
                    val = int(s)
                    done_in_order.append(val)
                    done_ids.add(val)

    last_done = done_in_order[-1] if done_in_order else None
    return done_ids, last_done

def append_done_id(progress_file, arch_id):
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"{arch_id}\n")

def append_fail(fail_file, arch_id, exc):
    with open(fail_file, "a", encoding="utf-8") as f:
        f.write(f"{arch_id}\t{type(exc).__name__}\t{exc}\n")

def ensure_csv_header(csv_filename, csv_headers):
    if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()

def parse_chunk_info(chunk_path):
    name = os.path.basename(chunk_path)
    m = CHUNK_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected chunk filename format: {name}")
    bucket_id = int(m.group(1))
    chunk_idx = int(m.group(2))
    first_arch = int(m.group(3))
    last_arch = int(m.group(4))
    return {
        "path": chunk_path,
        "name": name,
        "bucket_id": bucket_id,
        "chunk_idx": chunk_idx,
        "first_arch": first_arch,
        "last_arch": last_arch,
    }

def discover_chunks(chunks_dir):
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "*.tar.gz")))
    return [parse_chunk_info(p) for p in chunk_files]

def find_chunk_index_for_arch(chunks, arch_id):
    for i, c in enumerate(chunks):
        if c["first_arch"] <= arch_id <= c["last_arch"]:
            return i
    return None

def infer_resume_point(chunks, staging_arch_ids, last_done):
    if staging_arch_ids:
        st_min = min(staging_arch_ids)
        st_max = max(staging_arch_ids)
        candidate = None

        for i, c in enumerate(chunks):
            if c["first_arch"] <= st_min and st_max <= c["last_arch"]:
                candidate = i
                break

        if candidate is None:
            raise RuntimeError(
                f"Staging contains arch IDs [{st_min}, {st_max}] but no matching chunk range found."
            )

        return candidate, True, (
            f"Resuming from existing staging data (IDs {st_min}..{st_max}) "
            f"in chunk {chunks[candidate]['name']}"
        )

    if last_done is None:
        return 0, False, "No previous progress found. Starting from first chunk."

    next_arch = last_done + 1
    idx = find_chunk_index_for_arch(chunks, next_arch)

    if idx is not None:
        return idx, False, (
            f"Last done arch is {last_done}; next arch is {next_arch}; "
            f"resuming from chunk {chunks[idx]['name']}"
        )

    for i, c in enumerate(chunks):
        if c["last_arch"] > last_done:
            return i, False, (
                f"Last done arch is {last_done}; mapped by range fallback to chunk {c['name']}"
            )

    return len(chunks), False, "All chunk ranges are below/equal last done arch. Nothing to do."

def main():
    parser = argparse.ArgumentParser(description="Process NATS chunks and run PTQ.")
    parser.add_argument("--project_root", required=True, help="Root path of the PTQ-NAS-Bench repository")
    parser.add_argument("--bucket_dir", required=True, help="Directory of the target bucket")
    parser.add_argument("--csv_file", required=True, help="Path to save the output server results CSV")
    parser.add_argument("--done_file", required=True, help="Path to the progress done IDs text file")
    parser.add_argument("--fail_file", required=True, help="Path to the progress failed IDs text file")
    args = parser.parse_args()

    sys.path.append(args.project_root)
    try:
        from PTQ_NAS_BENCH_GitHub.Benchmark_scripts.nas_bench_201_workflow_multithread import PTQConfig, CIFAR10FromFiles
        from PTQ_NAS_BENCH_GitHub.Benchmark_scripts.ptq_main import ptq_main
    except ImportError as e:
        print(f"[-] Error loading custom modules: {e}. Please ensure --project_root points to the correct location.")
        return

    bucket_dir = args.bucket_dir
    csv_file = args.csv_file
    done_file = args.done_file
    fail_file = args.fail_file

    chunks_dir = os.path.join(bucket_dir, "chunks")
    staging_dir = os.path.join(bucket_dir, "staging")
    os.makedirs(staging_dir, exist_ok=True)

    csv_headers = [
        "model_id", "act_bits", "weight_bits", "bias_bits", "backend", "equalization",
        "act_quant_type", "weight_quant_type", "scale_factor_type", "weight_quant_granularity",
        "act_quant_granularity", "bias_corr", "graph_eq_iterations", "graph_eq_merge_bias",
        "act_quant_calibration_type", "act_scale_computation_type", "weight_quant_calibration_type",
        "calibrate_bn", "merge_bn", "top1_accuracy_pre","top1_accuracy"
    ]
    ensure_csv_header(csv_file, csv_headers)

    chunks = discover_chunks(chunks_dir)
    if not chunks:
        print(f"[-] No chunk files found in {chunks_dir}")
        return

    done_ids, last_done = load_done_ids(done_file)
    staging_arch_ids = load_local_arch_ids(staging_dir)

    resume_chunk_idx, use_existing_staging, reason = infer_resume_point(
        chunks=chunks,
        staging_arch_ids=staging_arch_ids,
        last_done=last_done,
    )

    print(f"[*] Bucket: {bucket_dir}")
    print(f"[*] Total chunks found: {len(chunks)}")
    print(f"[*] Done IDs loaded: {len(done_ids)}")
    print(f"[*] Last done arch: {last_done}")
    print(f"[*] Staging currently has {len(staging_arch_ids)} arch file(s)")
    print(f"[*] Resume decision: {reason}")

    if resume_chunk_idx >= len(chunks):
        print("[+] Nothing to process. Exiting.")
        return

    trainset = CIFAR10FromFiles(train=True)
    testset = CIFAR10FromFiles(train=False)
    base_args = PTQConfig()

    act_bits = 8
    weight_bits = 2
    bias_bits = None
    backend = "fx"
    equalization = "layerwise"
    act_quant = "sym"
    weight_quant = "sym"
    weight_quant_gran = "per_tensor"
    weight_quant_cal_type = "mse"
    act_quant_gran = "per_tensor"
    act_quant_cal_type = "stats"
    act_scale_comp_type = "static"
    bias_corr = None

    for i in range(resume_chunk_idx, len(chunks)):
        c = chunks[i]
        chunk_path = c["path"]
        chunk_name = c["name"]

        print("=" * 70)
        print(f"Chunk [{i+1}/{len(chunks)}]: {chunk_name}")
        print("=" * 70, flush=True)

        if i == resume_chunk_idx and use_existing_staging:
            print("[*] Using existing staging content (no extraction).", flush=True)
        else:
            clear_dir(staging_dir)
            safe_extract_tar(chunk_path, staging_dir)
            print("[*] Chunk extracted into staging.", flush=True)

        arch_ids = load_local_arch_ids(staging_dir)
        if not arch_ids:
            print("[-] No arch_*.pth files found in this chunk, skipping.", flush=True)
            clear_dir(staging_dir)
            continue

        pending_arch_ids = [aid for aid in arch_ids if aid not in done_ids]
        if not pending_arch_ids:
            print("[*] All architectures in this chunk are already processed. Skipping.", flush=True)
            clear_dir(staging_dir)
            continue

        print(
            f"[*] Chunk architectures: {len(arch_ids)} | Pending after resume filter: {len(pending_arch_ids)}",
            flush=True
        )

        for arch_id in pending_arch_ids:
            print(f"{'=' * 50}\nProcessing Arch ID: {arch_id}\n{'=' * 50}", flush=True)

            trainset_final = trainset
            if base_args.calibration_samples is not None:
                trainset_final = torch.utils.data.Subset(
                    trainset, list(range(base_args.calibration_samples))
                )

            args_ptq = PTQConfig().update(
                trainset=trainset_final,
                full_trainset=trainset,
                testset=testset,
                act_bit_width=act_bits,
                weight_bit_width=weight_bits,
                bias_bit_width=bias_bits,
                target_backend=backend,
                act_equalization=equalization,
                act_quant_type=act_quant,
                weight_quant_type=weight_quant,
                weight_quant_granularity=weight_quant_gran,
                weight_quant_calibration_type=weight_quant_cal_type,
                act_quant_granularity=act_quant_gran,
                act_quant_calibration_type=act_quant_cal_type,
                act_scale_computation_type=act_scale_comp_type,
                bias_corr=bias_corr
            )

            try:
                top1_accuracy_pre,top1_accuracy, _ = ptq_main(arch_id, args_ptq)
            except Exception as exc:
                print(f"[-] Error during PTQ for Arch {arch_id}: {exc}", flush=True)
                append_fail(fail_file, arch_id, exc)
                continue

            csv_row = {
                "model_id": arch_id,
                "act_bits": act_bits,
                "weight_bits": weight_bits,
                "bias_bits": bias_bits if bias_bits is not None else "None",
                "backend": backend,
                "equalization": equalization,
                "act_quant_type": act_quant,
                "weight_quant_type": weight_quant,
                "scale_factor_type": act_scale_comp_type,
                "weight_quant_granularity": weight_quant_gran,
                "act_quant_granularity": act_quant_gran,
                "bias_corr": args_ptq.bias_corr,
                "graph_eq_iterations": args_ptq.graph_eq_iterations,
                "graph_eq_merge_bias": args_ptq.graph_eq_merge_bias,
                "act_quant_calibration_type": act_quant_cal_type,
                "act_scale_computation_type": act_scale_comp_type,
                "weight_quant_calibration_type": weight_quant_cal_type,
                "calibrate_bn": args_ptq.calibrate_bn,
                "merge_bn": args_ptq.merge_bn,
                "top1_accuracy_pre": float(top1_accuracy_pre),
                "top1_accuracy": float(top1_accuracy),
            }

            with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                writer.writerow(csv_row)

            append_done_id(done_file, arch_id)
            done_ids.add(arch_id)

            print(f"[+] Completed Arch {arch_id} | Accuracy = {top1_accuracy:.3f}%", flush=True)

        clear_dir(staging_dir)
        print(f"[*] Finished chunk {chunk_name}. Staging cleared.\n", flush=True)

    print("[+] All eligible chunks for this bucket were processed.", flush=True)

if __name__ == "__main__":
    main()