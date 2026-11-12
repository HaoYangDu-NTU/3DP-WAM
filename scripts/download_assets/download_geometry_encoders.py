#!/usr/bin/env python3
"""Prepare pinned official geometry assets. Never run a model forward or training.

Uses OpenWAM's existing resumable direct downloader and encoder/ +
assets/pretrained/ layout. A manifest is complete only after official SHA256
checks; model construction/strict loading is a separate preparation check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_visual_encoder import _download_direct

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def fetch(url, path, expected=None):
    expected = expected or {}
    if not path.exists():
        # _download_direct keeps .part and resumes with HTTP Range. A failure
        # leaves the part file for this invocation or the next one to resume.
        for attempt in range(4):
            try:
                if url.startswith("https://creativecommons.org/"):
                    # The license site rejects urllib's user-agent on this host.
                    part = path.with_name(path.name + ".part")
                    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                    "--connect-timeout", "20", "--max-time", "180",
                                    "--continue-at", "-", "--output", str(part), url], check=True)
                    os.replace(part, path)
                else:
                    _download_direct(url, path)
                break
            except Exception as exc:
                print(f"Download attempt {attempt + 1}/4: {type(exc).__name__}: {exc}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(2)
    actual = {"bytes": path.stat().st_size, "sha256": digest(path), "url": url}
    for key in ("bytes", "sha256"):
        if key in expected and actual[key] != expected[key]:
            raise ValueError(f"{path}: {key} mismatch, expected {expected[key]}, got {actual[key]}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=("all", "utonia-tiny", "vggt-1b"), default="all")
    parser.add_argument("--reuse-from", type=Path, help="Reuse byte-identical official files from a local directory (requires --name)")
    args = parser.parse_args()
    if args.reuse_from and args.name == "all":
        parser.error("--reuse-from requires a single --name")
    registry = json.loads((ROOT / "assets/geometry_registry.json").read_text())
    for name, spec in registry["models"].items():
        if args.name not in ("all", name):
            continue
        target = ROOT / "encoder" / spec["subdir"]
        target.mkdir(parents=True, exist_ok=True)
        code = ROOT / spec["code_directory"]
        commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
        if commit != spec["code_revision"]:
            raise ValueError(f"{name}: checkout {spec['code_revision']} before preparing weights; got {commit}")
        manifest = dict(spec, code_commit=commit, local_directory=str(target),
                        status="preparing", model_load_status="not_checked", files={})
        write_json(target / "manifest.json", manifest)
        for filename, expected in spec["files"].items():
            url = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{filename}"
            print(f"Preparing {name}/{filename}", flush=True)
            local = args.reuse_from / filename if args.reuse_from else None
            if local and local.is_file() and "sha256" in expected and not (target / filename).exists():
                if local.stat().st_size != expected["bytes"] or digest(local) != expected["sha256"]:
                    raise ValueError(f"Local reuse candidate differs from official checkpoint: {local}")
                # Keep shared assets read-only by convention; no changes to their contents.
                (target / filename).symlink_to(local.resolve())
                manifest.setdefault("reused_from", {})[filename] = str(local.resolve())
            manifest["files"][filename] = fetch(url, target / filename, expected)
            if (target / filename).is_symlink():
                manifest.setdefault("reused_from", {})[filename] = str((target / filename).resolve())
            write_json(target / "manifest.json", manifest)
            print(f"Verified {filename}: {manifest['files'][filename]['sha256']}", flush=True)
        manifest["files"]["WEIGHTS_LICENSE.txt"] = fetch(spec["weight_license_url"], target / "WEIGHTS_LICENSE.txt")
        shutil.copyfile(code / spec["code_license_file"], target / "CODE_LICENSE.txt")
        manifest["files"]["CODE_LICENSE.txt"] = {"sha256": digest(target / "CODE_LICENSE.txt")}
        manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "downloaded_sha256_verified"
        write_json(target / "manifest.json", manifest)
        link = ROOT / "assets/pretrained" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError(f"Unexpected resource link: {link}")
        print(f"READY {name}: {target}; strict model loading still requires a separate check", flush=True)


if __name__ == "__main__":
    main()
