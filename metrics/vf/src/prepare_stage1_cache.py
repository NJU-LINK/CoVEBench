#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path


root = Path(__file__).resolve().parent
cache_dir = root / "stage1_cache"
outputs_dir = root / "outputs"
repo_root = root.parents[2]
checklist_path = Path(
    os.environ.get("CHECKLIST_PATH", repo_root / "data" / "checklist.json")
).expanduser().resolve()

data_dir = Path(os.environ.get("CUT_DATA_DIR", "data")).expanduser().resolve()

if not cache_dir.exists():
    raise SystemExit(f"missing cache dir: {cache_dir}")
if not checklist_path.exists():
    raise SystemExit(f"missing checklist: {checklist_path}")

items = {
    int(item["id"]): item
    for item in json.loads(checklist_path.read_text(encoding="utf-8"))
}

if outputs_dir.exists():
    shutil.rmtree(outputs_dir)
shutil.copytree(cache_dir, outputs_dir)

for meta_path in outputs_dir.glob("*/metadata.json"):
    item_id = int(meta_path.parent.name)
    item = items[item_id]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta["videoA_path"] = item["videoA_path"]
    meta["local_path"] = str(data_dir / Path(item["videoA_path"]).name)

    if meta.get("status") == "completed":
        meta["status"] = "ready_for_gpu"

    for key in (
        "num_frames",
        "fps",
        "gdino_records",
        "matched_entities",
        "error",
    ):
        meta.pop(key, None)

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

print(f"prepared cache in {outputs_dir}")
print(f"CUT_DATA_DIR={data_dir}")
