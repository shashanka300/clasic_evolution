"""Dataset loaders, shared cache, timers, and frozen JSONL result writer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.request

import h5py
import numpy as np
import torch
from tqdm import tqdm


REQUIRED_RESULT_KEYS = (
    "track",
    "method",
    "dataset",
    "n",
    "d",
    "k_or_units",
    "epochs",
    "batch",
    "alpha0",
    "seed",
    "train_loss",
    "test_loss",
    "wall_s",
    "gpu_s",
    "extra",
)


@dataclass
class DatasetSplit:
    """Standard train/val/test split returned by benchmark loaders."""

    name: str
    train: np.ndarray
    val: np.ndarray | None
    test: np.ndarray
    queries: np.ndarray | None = None
    groundtruth: np.ndarray | None = None


def data_root() -> Path:
    """Return the shared ``data/`` directory (respecting ``CLASSIC_ALGOS_DATA_DIR``)."""
    env = os.environ.get("CLASSIC_ALGOS_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        p = (Path.cwd() / "data").resolve()
    for sub in ("raw", "processed", "cache"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def download_file(url: str, dest: Path, chunk_bytes: int = 1 << 20) -> Path:
    """Stream-download ``url`` to ``dest`` atomically with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "classic-algos-bench/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(part, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            while True:
                buf = resp.read(chunk_bytes)
                if not buf:
                    break
                f.write(buf)
                bar.update(len(buf))
    part.replace(dest)
    return dest


def _load_ann_hdf5(
    name: str,
    url: str,
    val_size: int = 10_000,
    normalize: bool = False,
    max_train: int | None = None,
) -> DatasetSplit:
    root = data_root()
    raw_path = root / "raw" / Path(url).name
    proc_dir = root / "processed" / f"{name}{'_norm' if normalize else ''}"
    train_npy = proc_dir / "train.npy"
    val_npy = proc_dir / "val.npy"
    test_npy = proc_dir / "test.npy"
    gt_npy = proc_dir / "neighbors.npy"

    if not (train_npy.exists() and test_npy.exists()):
        download_file(url, raw_path)
        proc_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(raw_path, "r") as f:
            train_all = np.asarray(f["train"][:max_train], dtype=np.float32)
            test_arr = np.asarray(f["test"][:], dtype=np.float32)
            gt_arr = np.asarray(f["neighbors"][:], dtype=np.int32) if "neighbors" in f else None
        if normalize:
            train_all /= np.maximum(np.linalg.norm(train_all, axis=1, keepdims=True), 1e-12)
            test_arr /= np.maximum(np.linalg.norm(test_arr, axis=1, keepdims=True), 1e-12)
        if val_size > 0 and train_all.shape[0] > val_size:
            train_arr = train_all[:-val_size]
            val_arr = train_all[-val_size:]
            np.save(val_npy, val_arr)
        else:
            train_arr = train_all
        np.save(train_npy, train_arr)
        np.save(test_npy, test_arr)
        if gt_arr is not None:
            np.save(gt_npy, gt_arr)

    train = np.load(train_npy, mmap_mode="r")
    val = np.load(val_npy, mmap_mode="r") if val_npy.exists() else None
    test = np.load(test_npy, mmap_mode="r")
    gt = np.load(gt_npy, mmap_mode="r") if gt_npy.exists() else None
    return DatasetSplit(name=name, train=train, val=val, test=test, queries=test, groundtruth=gt)


def load_sift1m(val_size: int = 10_000, normalize: bool = False) -> DatasetSplit:
    """Load SIFT1M (ann-benchmarks ``sift-128-euclidean.hdf5``)."""
    return _load_ann_hdf5(
        name="sift1m",
        url="https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        val_size=val_size,
        normalize=normalize,
    )


def load_gist1m(val_size: int = 1_000, normalize: bool = False) -> DatasetSplit:
    """Load GIST1M (ann-benchmarks ``gist-960-euclidean.hdf5``)."""
    return _load_ann_hdf5(
        name="gist1m",
        url="https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        val_size=val_size,
        normalize=normalize,
    )


def load_deep(max_train: int = 1_000_000, val_size: int = 10_000, normalize: bool = True) -> DatasetSplit:
    """Load Deep10M (or a ``max_train`` slice such as 1M for fast sweeps)."""
    tag = f"deep_{max_train // 1_000_000}m" if max_train >= 1_000_000 else f"deep_{max_train}"
    return _load_ann_hdf5(
        name=tag,
        url="https://ann-benchmarks.com/deep-image-96-angular.hdf5",
        val_size=val_size,
        normalize=normalize,
        max_train=max_train,
    )


def load_dinov2_cifar100(
    model_name: str = "facebook/dinov2-small",
    batch_size: int = 256,
    val_size: int = 5_000,
    normalize: bool = True,
) -> DatasetSplit:
    """Extract and cache DINOv2 image embeddings on CIFAR-100 (50k train + 10k test)."""
    root = data_root()
    slug = model_name.split("/")[-1].replace("-", "_")
    cache_dir = root / "cache" / f"{slug}_cifar100{'_norm' if normalize else ''}"
    train_npy = cache_dir / "train.npy"
    val_npy = cache_dir / "val.npy"
    test_npy = cache_dir / "test.npy"

    if not (train_npy.exists() and test_npy.exists()):
        from datasets import load_dataset
        from transformers import AutoImageProcessor, AutoModel

        from core.device import accel
        device = accel()
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        def embed_images(imgs: list, split_name: str) -> np.ndarray:
            outs: list[np.ndarray] = []
            with torch.inference_mode():
                for start in tqdm(range(0, len(imgs), batch_size), desc=f"{slug}:{split_name}"):
                    batch_imgs = imgs[start : start + batch_size]
                    inputs = processor(images=batch_imgs, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                    feats = model(pixel_values=pixel_values).last_hidden_state[:, 0]
                    outs.append(feats.cpu().numpy().astype(np.float32))
            arr = np.concatenate(outs, axis=0)
            if normalize:
                arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
            return arr

        hf_cache = root / "raw" / "hf_cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        ds_train = load_dataset("uoft-cs/cifar100", split="train", cache_dir=str(hf_cache))
        ds_test = load_dataset("uoft-cs/cifar100", split="test", cache_dir=str(hf_cache))
        train_all = embed_images(list(ds_train["img"]), "train")
        test_arr = embed_images(list(ds_test["img"]), "test")

        cache_dir.mkdir(parents=True, exist_ok=True)
        if val_size > 0 and train_all.shape[0] > val_size:
            np.save(train_npy, train_all[:-val_size])
            np.save(val_npy, train_all[-val_size:])
        else:
            np.save(train_npy, train_all)
        np.save(test_npy, test_arr)

    return DatasetSplit(
        name=f"{slug}_cifar100",
        train=np.load(train_npy, mmap_mode="r"),
        val=np.load(val_npy, mmap_mode="r") if val_npy.exists() else None,
        test=np.load(test_npy, mmap_mode="r"),
    )


def load_text_embeddings(
    dataset_name: str = "ag_news",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_train: int = 100_000,
    max_test: int = 7_600,
    val_size: int = 5_000,
    batch_size: int = 256,
    normalize: bool = True,
) -> DatasetSplit:
    """Embed and cache a text dataset (e.g. ``ag_news`` or ``ms_marco``) via SentenceTransformer."""
    root = data_root()
    mslug = model_name.split("/")[-1].replace("-", "_")
    dslug = dataset_name.replace("/", "_")
    cache_dir = root / "cache" / f"{dslug}_{mslug}_{max_train}{'_norm' if normalize else ''}"
    train_npy = cache_dir / "train.npy"
    val_npy = cache_dir / "val.npy"
    test_npy = cache_dir / "test.npy"

    if not (train_npy.exists() and test_npy.exists()):
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        from core.device import accel
        device = accel()
        encoder = SentenceTransformer(model_name, device=device)

        if dataset_name == "ag_news":
            ds = load_dataset("fancyzhx/ag_news")
            train_texts = list(ds["train"]["text"][:max_train])
            test_texts = list(ds["test"]["text"][:max_test])
        else:
            ds_tr = load_dataset(dataset_name, split=f"train[:{max_train}]")
            col = "text" if "text" in ds_tr.column_names else ds_tr.column_names[0]
            train_texts = list(ds_tr[col])
            test_texts = train_texts[-max_test:]
            train_texts = train_texts[:-max_test]

        train_all = encoder.encode(
            train_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        ).astype(np.float32)
        test_arr = encoder.encode(
            test_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        ).astype(np.float32)

        cache_dir.mkdir(parents=True, exist_ok=True)
        if val_size > 0 and train_all.shape[0] > val_size:
            np.save(train_npy, train_all[:-val_size])
            np.save(val_npy, train_all[-val_size:])
        else:
            np.save(train_npy, train_all)
        np.save(test_npy, test_arr)

    return DatasetSplit(
        name=f"{dslug}_{mslug}",
        train=np.load(train_npy, mmap_mode="r"),
        val=np.load(val_npy, mmap_mode="r") if val_npy.exists() else None,
        test=np.load(test_npy, mmap_mode="r"),
    )


class Timer:
    """Wall-clock timer that synchronizes Apple Silicon MPS when active."""

    def __init__(self, use_mps: bool = False) -> None:
        from core.device import accel
        self.use_mps = bool(use_mps and accel() != "cpu")  # name kept for compatibility: any accelerator
        self.wall_s: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> Timer:
        if self.use_mps:
            from core.device import sync
            sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self.use_mps:
            from core.device import sync
            sync()
        self.wall_s = time.perf_counter() - self._t0


def write_result(path: str | Path, record: dict[str, Any]) -> None:
    """Validate and append one result record to a JSONL file."""
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in record]
    if missing:
        raise ValueError(f"Result record missing required keys: {missing}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=False) + "\n")


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """Read all JSONL result records from ``path``."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

