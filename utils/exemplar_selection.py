import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
import data_utils
from utils_my import load_biomedclip, _is_biomedclip_name


def safe_clip_name(clip_name: str) -> str:
    return clip_name.replace("/", "").replace(":", "")


def embedding_cache_path(
    dataset_name: str,
    split: str,
    clip_name: str,
    cache_dir: str,
) -> Path:
    return Path(cache_dir) / f"{dataset_name}_{split}_{safe_clip_name(clip_name)}.pt"


def load_or_compute_clip_embeddings(
    dataset_name: str,
    split: str = "train",
    clip_name: str = "biomedclip",
    cache_dir: str = "data/embedding_cache",
    activation_dir: str = "",
    batch_size: int = 512,
    device: str = "cuda",
) -> Dict:
    cache_path = embedding_cache_path(dataset_name, split, clip_name, cache_dir)
    if cache_path.exists():
        print(f"[exemplar] embedding cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_for_paths = data_utils.get_data(dataset_name, split, transform=None)

    if activation_dir:
        activation_path = Path(activation_dir) / f"{dataset_name}_{split}_clip_{safe_clip_name(clip_name)}.pt"
        if activation_path.exists():
            features = torch.load(activation_path, map_location="cpu").float()
            features = F.normalize(features, dim=1)
            payload = {
                "features": features,
                "paths": [path for path, _ in dataset_for_paths.samples],
                "targets": torch.tensor(dataset_for_paths.targets, dtype=torch.long),
                "classes": list(dataset_for_paths.classes),
                "dataset": dataset_name,
                "split": split,
                "clip_name": clip_name,
                "source_activation_path": str(activation_path),
            }
            torch.save(payload, cache_path)
            print(f"[exemplar] reused activation cache: {activation_path}")
            print(f"[exemplar] embedding cache: {cache_path}")
            return payload

    if _is_biomedclip_name(clip_name):
        model, preprocess, _ = load_biomedclip(device=device)
    else:
        model, preprocess = clip.load(clip_name, device=device)
    dataset = data_utils.get_data(dataset_name, split, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Embedding {dataset_name}_{split}"):
            image_features = model.encode_image(images.to(device)).float()
            image_features = F.normalize(image_features, dim=1)
            features.append(image_features.cpu())

    payload = {
        "features": torch.cat(features, dim=0),
        "paths": [path for path, _ in dataset.samples],
        "targets": torch.tensor(dataset.targets, dtype=torch.long),
        "classes": list(dataset.classes),
        "dataset": dataset_name,
        "split": split,
        "clip_name": clip_name,
    }
    torch.save(payload, cache_path)
    print(f"[exemplar] embedding cache: {cache_path}")
    return payload


def _select_random(indices: List[int], num_exemplars: int, rng: random.Random) -> List[int]:
    if len(indices) <= num_exemplars:
        return list(indices)
    return rng.sample(indices, num_exemplars)


def _nearest_to_kmeans_centers(
    features: torch.Tensor,
    indices: List[int],
    num_centers: int,
    seed: int,
) -> List[int]:
    if len(indices) <= num_centers:
        return list(indices)

    class_features = features[indices].numpy()
    n_clusters = min(num_centers, len(indices))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    cluster_ids = kmeans.fit_predict(class_features)

    selected = []
    for cluster_id in range(n_clusters):
        members = [i for i, cid in enumerate(cluster_ids) if cid == cluster_id]
        if not members:
            continue
        member_features = torch.tensor(class_features[members])
        center = torch.tensor(kmeans.cluster_centers_[cluster_id], dtype=member_features.dtype)
        distances = torch.linalg.norm(member_features - center, dim=1)
        selected.append(indices[members[int(torch.argmin(distances).item())]])
    return selected


def _class_centers(features: torch.Tensor, targets: torch.Tensor, class_to_indices: Dict[int, List[int]]) -> Dict[int, torch.Tensor]:
    centers = {}
    for class_idx, indices in class_to_indices.items():
        center = features[indices].mean(dim=0)
        centers[class_idx] = F.normalize(center, dim=0)
    return centers


def _select_boundary_samples(
    features: torch.Tensor,
    indices: List[int],
    class_idx: int,
    centers: Dict[int, torch.Tensor],
    count: int,
    exclude: set,
) -> List[int]:
    if count <= 0:
        return []

    other_centers = [center for other_idx, center in centers.items() if other_idx != class_idx]
    if not other_centers:
        return []
    other_centers = torch.stack(other_centers, dim=0)

    scored = []
    for idx in indices:
        if idx in exclude:
            continue
        distances = torch.linalg.norm(other_centers - features[idx], dim=1)
        scored.append((float(distances.min().item()), idx))

    scored.sort(key=lambda item: item[0])
    return [idx for _, idx in scored[:count]]


def select_exemplars(
    dataset_name: str,
    split: str = "train",
    clip_name: str = "biomedclip",
    sample_strategy: str = "random",
    num_exemplars: int = 8,
    output_root: str = "data/selected_image",
    cache_dir: str = "data/embedding_cache",
    activation_dir: str = "",
    batch_size: int = 512,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    strategy_aliases = {
        "representative": "diverse",
        "representative_only": "diverse",
        "boundary_only": "boundary",
        "representative_boundary": "hybrid",
        "representative+boundary": "hybrid",
    }
    original_sample_strategy = sample_strategy
    sample_strategy = strategy_aliases.get(sample_strategy, sample_strategy)
    if sample_strategy not in {"random", "diverse", "boundary", "hybrid"}:
        raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

    payload = load_or_compute_clip_embeddings(
        dataset_name=dataset_name,
        split=split,
        clip_name=clip_name,
        cache_dir=cache_dir,
        activation_dir=activation_dir,
        batch_size=batch_size,
        device=device,
    )
    features = payload["features"]
    targets = payload["targets"]
    paths = payload["paths"]
    classes = payload["classes"]

    class_to_indices = {
        class_idx: (targets == class_idx).nonzero(as_tuple=True)[0].tolist()
        for class_idx in range(len(classes))
    }
    rng = random.Random(seed)
    centers = _class_centers(features, targets, class_to_indices)

    output_dir = Path(output_root) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": dataset_name,
        "split": split,
        "clip_name": clip_name,
        "sample_strategy": original_sample_strategy,
        "resolved_sample_strategy": sample_strategy,
        "num_exemplars": num_exemplars,
        "embedding_cache": str(embedding_cache_path(dataset_name, split, clip_name, cache_dir)),
        "classes": {},
    }

    print(f"[exemplar] strategy: {sample_strategy}")
    for class_idx, indices in class_to_indices.items():
        class_name = classes[class_idx]
        selected_types = {}

        if sample_strategy == "random":
            selected = _select_random(indices, num_exemplars, rng)
            selected_types = {idx: "random" for idx in selected}
        elif sample_strategy == "diverse":
            selected = _nearest_to_kmeans_centers(features, indices, num_exemplars, seed)
            selected_types = {idx: "representative" for idx in selected}
        elif sample_strategy == "boundary":
            selected = _select_boundary_samples(
                features,
                indices,
                class_idx,
                centers,
                num_exemplars,
                exclude=set(),
            )
            if len(selected) < min(num_exemplars, len(indices)):
                fill = [idx for idx in indices if idx not in set(selected)]
                selected.extend(fill[: num_exemplars - len(selected)])
            selected = selected[:num_exemplars]
            selected_types = {idx: "boundary" for idx in selected}
        else:
            representative_count = (num_exemplars + 1) // 2
            boundary_count = num_exemplars - representative_count
            reps = _nearest_to_kmeans_centers(features, indices, representative_count, seed)
            boundary = _select_boundary_samples(
                features,
                indices,
                class_idx,
                centers,
                boundary_count,
                exclude=set(reps),
            )
            selected = reps + boundary
            if len(selected) < min(num_exemplars, len(indices)):
                fill = [idx for idx in indices if idx not in set(selected)]
                selected.extend(fill[: num_exemplars - len(selected)])
            selected = selected[:num_exemplars]
            selected_types = {idx: "representative" for idx in reps}
            selected_types.update({idx: "boundary" for idx in boundary})
            for idx in selected:
                selected_types.setdefault(idx, "fill")

        class_dir = output_dir / class_name
        if class_dir.exists():
            for old_file in class_dir.iterdir():
                if old_file.is_file():
                    old_file.unlink()
        class_dir.mkdir(parents=True, exist_ok=True)

        selected_files = []
        for idx in selected:
            src = Path(paths[idx])
            dst = class_dir / src.name
            shutil.copy(src, dst)
            selected_files.append(
                {
                    "path": str(src),
                    "filename": src.name,
                    "selection_type": selected_types.get(idx, sample_strategy),
                }
            )

        metadata["classes"][class_name] = selected_files
        print(f"[exemplar] {class_name}: {[item['filename'] for item in selected_files]}")

    metadata_path = output_dir / f"exemplar_selection_{original_sample_strategy}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[exemplar] selection log: {metadata_path}")
    return metadata
