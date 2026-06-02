from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .common import project_root, run_command, write_json

REPOS = {
    "aesthetic-predictor-v2-5": {
        "url": "https://github.com/discus0434/aesthetic-predictor-v2-5.git",
        "commit": "c0e15567fa61252cbff43d70c49a8bd27202ca9a",
    },
    "T2AV-Compass": {
        "url": "https://github.com/NJU-LINK/T2AV-Compass.git",
        "commit": "b726966a183c52de766627fe9a505bedae201d23",
    },
    "visualquality-r1": {
        "url": "https://github.com/tianhewu/visualquality-r1.git",
        "commit": "a16dcdb64eea03a27afdb411d43cc1c9d723a1b2",
    },
}

DIRECT_FILES = [
    {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "local": "weights/sam2/sam2_hiera_large.pt",
    },
]

HF_FILES = [
    {
        "repo": "facebook/cotracker3",
        "file": "scaled_offline.pth",
        "local": "weights/cotracker/scaled_offline.pth",
    },
]

HF_SNAPSHOT_MODELS = [
    {
        "repo": "TianheWu/VisualQuality-R1-7B",
        "local": "weights/hf/VisualQuality-R1-7B",
    },
    {
        "repo": "facebook/dinov2-small",
        "local": "weights/hf/dinov2-small",
    },
    {
        "repo": "IDEA-Research/grounding-dino-base",
        "local": "weights/hf/grounding-dino-base",
    },
]

DEFAULT_SUBJECTIVE_MODEL_REPO = "Qwen/Qwen3.5-122B-A10B"



def run_git(args: list[str], cwd: Path | None = None) -> None:
    run_command(["git", *args], cwd=cwd)


def clone_repo(name: str, url: str, commit: str, dest: Path, proxy_prefix: str) -> None:
    clone_url = proxy_prefix + url if proxy_prefix and url.startswith("https://github.com/") else url
    if not dest.exists():
        run_git(["clone", "--filter=blob:none", clone_url, str(dest)])
    run_git(["fetch", "--depth", "1", "origin", commit], cwd=dest)
    run_git(["checkout", commit], cwd=dest)
    if shutil.which("git-lfs"):
        subprocess.run(["git", "lfs", "pull"], cwd=dest, check=False)


def hf_download(repo: str, filename: str | None, local_dir: Path, endpoint: str) -> None:
    env = os.environ.copy()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf_bin is None:
        raise RuntimeError("Neither `hf` nor `huggingface-cli` is available. Install huggingface-hub[cli].")
    cmd = [hf_bin, "download", repo]
    if filename:
        cmd.extend([filename, "--local-dir", str(local_dir.parent)])
    else:
        cmd.extend(["--local-dir", str(local_dir)])
    subprocess.run(cmd, check=True, env=env)


def download_file(url: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 1024 * 1024:
        return
    tmp = local.with_suffix(local.suffix + ".tmp")
    print(f"Downloading {url} -> {local}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(local)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CoVEBench metric repos and weights.")
    parser.add_argument("--root", default=str(project_root()), help="covebench_eval project root.")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", ""), help="Optional HF mirror, e.g. https://hf-mirror.com.")
    parser.add_argument("--github-proxy", default=os.environ.get("GITHUB_PROXY", ""), help="Optional GitHub proxy prefix.")
    parser.add_argument("--skip-hf", action="store_true", help="Only clone metric code repositories.")
    parser.add_argument("--include-subjective", action="store_true", help="Also download the MLLM judge model for subjective metrics.")
    parser.add_argument(
        "--subjective-model-repo",
        default=os.environ.get("SUBJECTIVE_MODEL_REPO", DEFAULT_SUBJECTIVE_MODEL_REPO),
        help=f"HF repo id for the subjective MLLM judge. Default: {DEFAULT_SUBJECTIVE_MODEL_REPO}.",
    )
    parser.add_argument(
        "--subjective-model-dir",
        default="weights/hf/subjective_judge",
        help="Local directory for the subjective MLLM judge model.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    external = root / "external"
    weights = root / "weights"
    external.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"repos": {}, "hf": []}
    for name, meta in REPOS.items():
        dest = external / name
        clone_repo(name, meta["url"], meta["commit"], dest, args.github_proxy)
        manifest["repos"][name] = {
            "url": meta["url"],
            "commit": meta["commit"],
            "path": str(dest.relative_to(root)),
        }

    if not args.skip_hf:
        for item in DIRECT_FILES:
            local = root / item["local"]
            download_file(item["url"], local)
            manifest["hf"].append({**item, "path": str(local.relative_to(root))})
        for item in HF_FILES:
            local = root / item["local"]
            if not local.exists():
                hf_download(item["repo"], item["file"], local, args.hf_endpoint)
            manifest["hf"].append({**item, "path": str(local.relative_to(root))})
        for item in HF_SNAPSHOT_MODELS:
            local = root / item["local"]
            if not local.exists() or not any(local.iterdir()):
                hf_download(item["repo"], None, local, args.hf_endpoint)
            manifest["hf"].append({**item, "path": str(local.relative_to(root))})

        if args.include_subjective:
            if not args.subjective_model_repo:
                raise SystemExit(
                    "--include-subjective requires --subjective-model-repo or SUBJECTIVE_MODEL_REPO. "
                    f"The released CoVEBench judge model is {DEFAULT_SUBJECTIVE_MODEL_REPO}."
                )
            local = root / args.subjective_model_dir
            if not local.exists() or not any(local.iterdir()):
                hf_download(args.subjective_model_repo, None, local, args.hf_endpoint)
            manifest["hf"].append(
                {
                    "repo": args.subjective_model_repo,
                    "local": args.subjective_model_dir,
                    "path": str(local.relative_to(root)),
                    "usage": "subjective_mllm_judge",
                }
            )

    # Convenience copies/symlinks for repos that store weights internally.
    aes_weight = external / "aesthetic-predictor-v2-5" / "models" / "aesthetic_predictor_v2_5.pth"
    if aes_weight.exists():
        if aes_weight.stat().st_size < 1024 * 1024:
            print(f"WARNING: {aes_weight} looks too small. Git LFS may not have pulled the real weight.")
        target = weights / "aes" / "aesthetic_predictor_v2_5.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(aes_weight, target)

    dover_weight = (
        external
        / "T2AV-Compass"
        / "t2av-compass"
        / "Objective"
        / "Video"
        / "DOVER"
        / "pretrained_weights"
        / "DOVER_plus_plus.pth"
    )
    if dover_weight.exists():
        if dover_weight.stat().st_size < 1024 * 1024:
            print(f"WARNING: {dover_weight} looks too small. Git LFS may not have pulled the real weight.")
        target = weights / "dover" / "DOVER_plus_plus.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(dover_weight, target)

    write_json(root / "weights" / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
