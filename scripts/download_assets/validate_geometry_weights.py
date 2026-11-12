#!/usr/bin/env python3
"""CPU-only strict pretrained loading, without forward, backward or experiments."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_geometry_encoders import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, choices=("utonia-tiny", "vggt-1b", "dinov3-hplus", "dinov3-l"))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    directory = ROOT / "encoder" / args.name
    report = {"name": args.name, "directory": str(directory), "device": "cpu",
              "forward_executed": False, "backward_executed": False,
              "gpu_memory": None, "status": "loading", "torch": torch.__version__}
    if args.name in ("utonia-tiny", "vggt-1b"):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "downloaded_sha256_verified":
            raise ValueError("Download and hash verification must complete before loading")
        code = ROOT / manifest["code_directory"]
        actual_commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
        if actual_commit != manifest["code_commit"]:
            raise ValueError("Source commit differs from prepared manifest")
        sys.path.insert(0, str(code))
        filename = "utonia_tiny.pth" if args.name == "utonia-tiny" else "model.safetensors"
        if digest(directory / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError("Checkpoint hash changed since preparation")
        report.update(checkpoint_sha256=manifest["files"][filename]["sha256"], code_commit=actual_commit)
        if args.name == "utonia-tiny":
            import utonia
            # Only the attention implementation is selected for CPU loading;
            # official checkpoint geometry and all tensor shapes are retained.
            model = utonia.load(str(directory / filename), custom_config={"enable_flash": False})
            report["load_method"] = "official utonia.load(local_path); load_state_dict strict=True by default"
            report["attention_override"] = {"enable_flash": False}
        else:
            from safetensors.torch import load_file
            from vggt.models.vggt import VGGT
            config = json.loads((directory / "config.json").read_text())
            # Native normalization has non-persistent buffers. Construct on CPU
            # so those buffers are real tensors even though they are not saved.
            model = VGGT(**config)
            result = model.load_state_dict(load_file(str(directory / filename), device="cpu"), strict=True, assign=True)
            report.update(load_method="VGGT(**official_config) + safetensors + strict=True, assign=True",
                          missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys)
    else:
        asset_name = "dinov3_hplus" if args.name == "dinov3-hplus" else "dinov3_l"
        spec = json.loads((ROOT / "assets/registry.json").read_text())["pretrained"][asset_name]
        checkpoint = directory / "model.safetensors"
        actual_hash = digest(checkpoint)
        if actual_hash != spec["reference_sha256"]["model.safetensors"]:
            raise ValueError("Reused DINOv3 hash differs from the original infra registry")
        from transformers import AutoModel
        # Match the infra loader: Transformers applies its official checkpoint
        # key migration (legacy `layer.*` -> current `model.layer.*`). Verify
        # that this leaves zero missing/unexpected/mismatched tensors.
        model, loading = AutoModel.from_pretrained(
            str(directory), local_files_only=True, output_loading_info=True,
            ignore_mismatched_sizes=False,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Incomplete pretrained DINOv3 load: {loading}")
        report.update(checkpoint_sha256=actual_hash, source_revision=spec["source_revision"],
                      load_method="infra-compatible AutoModel.from_pretrained; official key migration; zero missing/unexpected/mismatched keys",
                      loading_info={key: sorted(value) if isinstance(value, set) else value
                                    for key, value in loading.items()})
    model.eval().requires_grad_(False)
    parameters = list(model.parameters())
    if any(p.is_meta or p.device.type != "cpu" or p.requires_grad for p in parameters):
        raise RuntimeError("Unloaded, non-CPU, or unfrozen parameters remain")
    if any(b.is_meta or b.device.type != "cpu" for b in model.buffers()):
        raise RuntimeError("Unloaded or non-CPU model buffers remain")
    report.update(status="strict_cpu_load_passed", model_class=type(model).__name__,
                  parameter_count=sum(p.numel() for p in parameters),
                  trainable_parameter_count=sum(p.numel() for p in parameters if p.requires_grad),
                  frozen=True, eval_mode=not model.training,
                  elapsed_seconds=time.monotonic() - started,
                  note="Weight preparation validation only. No feature extraction, gradient-flow, integration, or training claim.")
    report["dependencies"] = {}
    for package in ("torch", "torch-scatter", "spconv-cu126", "cumm-cu126", "transformers", "huggingface-hub", "safetensors", "timm", "addict"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(ROOT / "preparation" / f"{args.name}_load_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
