"""Pin Table-5 data/model inputs and materialize shared offline splits."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Image, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
REPOS = {
    "cifar100": "uoft-cs/cifar100", "cifar10": "uoft-cs/cifar10",
    "oxfordpets": "timm/oxford-iiit-pet", "stanfordcars": "tanganke/stanford_cars",
    "dtd": "tanganke/dtd", "eurosat": "blanchon/EuroSAT_RGB",
    "resisc45": "timm/resisc45",
}
ORDER = ["cifar100", "dtd", "cifar10", "oxfordpets", "stanfordcars", "eurosat", "resisc45", "fgvc"]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def prepare_models():
    for size in ("base", "large"):
        dest = ROOT / "models" / f"{size}.json"
        if dest.exists():
            continue
        repo = f"google/vit-{size}-patch16-224-in21k"
        info = HfApi().model_info(repo)
        names = {x.rfilename for x in info.siblings}
        weights = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
        path = snapshot_download(repo, revision=info.sha,
                                 allow_patterns=["config.json", "preprocessor_config.json", weights],
                                 max_workers=2)
        atomic_json(dest, dict(repo=repo, revision=info.sha, path=path, weights=weights))
        print("MODEL READY", size, info.sha, flush=True)


def load_source(name):
    if name == "fgvc":
        import torchvision
        from torchvision.datasets import FGVCAircraft
        root = ROOT / "downloads" / "fgvc"
        result = {}
        for split, torchsplit in (("train", "train"), ("validation", "val"), ("test", "test")):
            data = FGVCAircraft(str(root), split=torchsplit, annotation_level="variant", download=True)
            result[split] = Dataset.from_dict({"image": data._image_files, "label": data._labels})
            result[split] = result[split].cast_column("image", Image()).cast_column("label", ClassLabel(names=data.classes))
        return DatasetDict(result), dict(source="torchvision.datasets.FGVCAircraft", version=torchvision.__version__, annotation="variant", splits="official train/val/test")
    repo = REPOS[name]
    info = HfApi().dataset_info(repo)
    files = {}
    for split in ("train", "validation", "test"):
        matches = sorted(x.rfilename for x in info.siblings if x.rfilename.endswith(".parquet") and Path(x.rfilename).name.startswith(split + "-"))
        if matches:
            with ThreadPoolExecutor(max_workers=3) as pool:
                files[split] = list(pool.map(lambda f: hf_hub_download(repo, f, repo_type="dataset", revision=info.sha), matches))
    if not files:
        raise RuntimeError(f"No data files found for {repo}")
    return load_dataset("parquet", data_files=files), dict(repo=repo, revision=info.sha, files=files)


def prepare_dataset(name):
    dest = ROOT / "data" / name
    if (dest / "manifest.json").exists():
        print("DATA EXISTS", name, flush=True)
        return
    raw, source = load_source(name)
    parts = {}
    for split, ds in raw.items():
        image_key = "image" if "image" in ds.column_names else "img"
        label_key = "fine_label" if "fine_label" in ds.column_names else "label"
        ds = ds.select_columns([image_key, label_key])
        if image_key != "image": ds = ds.rename_column(image_key, "image")
        if label_key != "label": ds = ds.rename_column(label_key, "label")
        ds = ds.add_column("example_id", [f"{split}:{i}" for i in range(len(ds))])
        parts[split] = ds
    recipe = "official train/validation/test"
    if name in ("cifar10", "cifar100", "oxfordpets", "stanfordcars"):
        split = parts["train"].train_test_split(test_size=0.1, seed=42)
        parts["train"], parts["validation"] = split["train"], split["test"]
        recipe = "original train split held out 10% with seed 42; official test"
    elif name == "dtd":
        full = concatenate_datasets([parts[k] for k in ("train", "validation", "test") if k in parts])
        # Stable imagefolder-like ordering, then the legacy script's fixed splits.
        undecoded = full.cast_column("image", Image(decode=False))
        labels = list(full["label"])
        filenames = [x.get("path") or "" for x in undecoded["image"]]
        if any(not x for x in filenames):
            raise RuntimeError("DTD source image names missing; cannot reconstruct stable imagefolder order")
        order = sorted(range(len(full)), key=lambda i: (labels[i], Path(filenames[i]).name))
        full = full.select(order).shuffle(seed=42)
        first = full.train_test_split(test_size=0.28, seed=42)
        second = first["test"].train_test_split(test_size=0.715, seed=42)
        parts = dict(train=first["train"], validation=second["train"], test=second["test"])
        recipe = "DTD all images sorted by label/filename; shuffle 42; split .28/seed42 then .715/seed42 (legacy recipe; manifests fixed)"
    assert set(parts) == {"train", "validation", "test"}, (name, parts.keys())
    ids = {k: list(v["example_id"]) for k, v in parts.items()}
    sets = [set(v) for v in ids.values()]
    assert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i+1:])
    assert all(len(sets[i]) == len(list(parts.values())[i]) for i in range(3))
    names = parts["train"].features["label"].names
    expected = dict(cifar100=100, cifar10=10, dtd=47, oxfordpets=37, stanfordcars=196, eurosat=10, resisc45=45, fgvc=100)[name]
    assert len(names) == expected
    assert len(set(parts["train"]["label"])) == expected
    for split, ds in parts.items():
        assert ds.features["label"].names == names
        for i in list(range(min(8, len(ds)))) + [len(ds)-1]:
            ds[i]["image"].convert("RGB").load()
    dest.mkdir(parents=True, exist_ok=True)
    for split, ds in parts.items():
        ds.save_to_disk(str(dest / split), max_shard_size="500MB")
    atomic_json(dest / "indices.json", ids)
    metadata = dict(dataset=name, source=source, split_recipe=recipe, split_seed=42,
                    sizes={k: len(v) for k,v in parts.items()}, classes=names,
                    indices_sha256=hashlib.sha256((dest / "indices.json").read_bytes()).hexdigest(),
                    datasets_version=datasets.__version__)
    metadata["manifest_sha256"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    atomic_json(dest / "manifest.json", metadata)
    print("DATA READY", name, metadata["sizes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=ORDER)
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    if not args.skip_models:
        prepare_models()
    for dataset in args.datasets:
        prepare_dataset(dataset)
