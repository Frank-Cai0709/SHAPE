import os
import math
import torch
import clip
import json
import data_utils
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

PM_SUFFIX = {"max": "_max", "avg": ""}


BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _is_biomedclip_name(clip_name):
    return clip_name.lower() in {"biomedclip", "biomedclip_vit", BIOMEDCLIP_MODEL_NAME.lower()}


def _safe_encoder_name(clip_name):
    return clip_name.replace("/", "").replace(":", "")


def get_activation(outputs, mode):
    if mode == 'avg':
        def hook(model, input, output):
            if output.dim() == 4:
                outputs.append(output.mean(dim=[2, 3]).detach().cpu())
            else:
                outputs.append(output.detach().cpu())
    elif mode == 'max':
        def hook(model, input, output):
            if output.dim() == 4:
                outputs.append(output.amax(dim=[2, 3]).detach().cpu())
            else:
                outputs.append(output.detach().cpu())
    return hook


def _make_save_dir(save_name):
    save_dir = os.path.dirname(save_name)
    os.makedirs(save_dir, exist_ok=True)


def _all_saved(save_names):
    return all(os.path.exists(path) for path in save_names.values())


def save_clip_text_features(model, text, save_path, batch_size=1000):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text) / batch_size))):
            features.append(model.encode_text(text[i * batch_size:(i + 1) * batch_size]))
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()


def save_clip_image_features(model, dataset, save_path, batch_size=1000, device="cuda"):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features.append(model.encode_image(images.to(device)).cpu())
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()


def load_biomedclip(device="cuda"):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "BiomedCLIP is loaded through open_clip. Install it with: pip install open_clip_torch"
        ) from exc

    model_name = os.environ.get("BIOMEDCLIP_MODEL_NAME", BIOMEDCLIP_MODEL_NAME)
    cache_dir = os.environ.get("BIOMEDCLIP_CACHE_DIR") or os.environ.get("HF_HOME")
    try:
        if os.path.isdir(model_name):
            config_path = os.path.join(model_name, "open_clip_config.json")
            weights_path = os.path.join(model_name, "open_clip_pytorch_model.bin")
            with open(config_path, "r", encoding="utf-8") as f:
                biomedclip_config = json.load(f)
            local_config_path = os.path.join(model_name, "biomedclip_local.json")
            with open(local_config_path, "w", encoding="utf-8") as f:
                json.dump(biomedclip_config["model_cfg"], f)
            open_clip.add_model_config(local_config_path)
            model = open_clip.create_model(
                "biomedclip_local",
                pretrained=weights_path,
                force_preprocess_cfg=biomedclip_config.get("preprocess_cfg"),
                device=device,
            )
            from open_clip.transform import PreprocessCfg, image_transform_v2
            preprocess_cfg = biomedclip_config.get("preprocess_cfg", {})
            image_size = biomedclip_config["model_cfg"]["vision_cfg"].get("image_size", 224)
            preprocess = image_transform_v2(
                PreprocessCfg(
                    size=image_size,
                    mean=tuple(preprocess_cfg.get("mean", (0.48145466, 0.4578275, 0.40821073))),
                    std=tuple(preprocess_cfg.get("std", (0.26862954, 0.26130258, 0.27577711))),
                ),
                is_train=False,
            )
            tokenizer = open_clip.get_tokenizer(
                f"local-dir:{model_name}",
                context_length=biomedclip_config["model_cfg"]["text_cfg"].get("context_length", 256),
            )
        else:
            model, preprocess = open_clip.create_model_from_pretrained(model_name, cache_dir=cache_dir)
            tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load BiomedCLIP. This usually means the Hugging Face model "
            "has not been downloaded or the current network cannot reach Hugging Face. "
            "Try setting HF_ENDPOINT=https://hf-mirror.com, or pre-download "
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 into the local "
            "HF cache. You can override the model id with BIOMEDCLIP_MODEL_NAME and "
            "the cache directory with BIOMEDCLIP_CACHE_DIR."
        ) from exc
    model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def save_biomedclip_text_features(model, tokenizer, concepts, save_path, batch_size=1000, device="cuda"):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(concepts) / batch_size))):
            batch = concepts[i * batch_size:(i + 1) * batch_size]
            text_tokens = tokenizer(batch, context_length=256).to(device)
            features.append(model.encode_text(text_tokens).cpu())
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()


def save_biomedclip_image_features(model, dataset, save_path, batch_size=1000, device="cuda"):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features.append(model.encode_image(images.to(device)).cpu())
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()


def save_clip_rn_penultimate_features(model, dataset, save_path, batch_size=1000, device="cuda"):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features.append(model(images.to(device)).cpu())
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()
    

def save_target_activations(model, dataset, save_template, target_layers, batch_size=1000, device="cuda", pool_mode="avg"):
    
    save_paths = {layer: save_template.format(layer) for layer in target_layers}
    if _all_saved(save_paths): 
        return

    outputs = {layer: [] for layer in target_layers}
    hooks = {}

    for layer in target_layers:
        try:
            layer_obj = _get_layer_by_name(model, layer)

            hooks[layer] = layer_obj.register_forward_hook(
                get_activation(outputs[layer], pool_mode)
            )
        except AttributeError as e:
            raise ValueError(f"No such '{layer}': {str(e)}")

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            model(images.to(device))

    for layer in target_layers:
        torch.save(torch.cat(outputs[layer]), save_paths[layer])
        hooks[layer].remove() 
    
    torch.cuda.empty_cache()

def _get_layer_by_name(model, layer_name):

    parts = layer_name.split('.')
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current

def get_feature_paths(
    clip_name,
    target_name,
    d_probe,
    concept_set_path,
    target_layers=None,
    pool_mode=None,
    save_dir=".",
    use_penultimate=False,
):
   
    safe_clip_name = _safe_encoder_name(clip_name)
    clip_feat_path = f"{save_dir}/{d_probe}_clip_{safe_clip_name}.pt"
    
    concept_name = os.path.splitext(os.path.basename(concept_set_path))[0]
    text_feat_path = f"{save_dir}/{concept_name}_{safe_clip_name}.pt"
    
    if _is_biomedclip_name(target_name):
        target_paths = clip_feat_path
    elif target_name.startswith("clip_"):
        penultimate_suffix = "_penultimate" if use_penultimate else ""
        target_paths = f"{save_dir}/{d_probe}_{target_name[5:].replace('/', '')}{penultimate_suffix}.pt"
    else:
        if target_layers is None or pool_mode is None:
            raise ValueError("Error!")
        target_paths = {
            layer: f"{save_dir}/{d_probe}_{target_name}_{layer}{PM_SUFFIX[pool_mode]}.pt"
            for layer in target_layers
        }
    
    return {
        "clip": clip_feat_path,
        "text": text_feat_path,
        "target": target_paths
    }

def save_all_features(clip_name, target_name, target_layers, d_probe, concept_set_path,
                    batch_size, device, pool_mode, save_dir, use_penultimate=False):

    paths = get_feature_paths(
        clip_name=clip_name,
        target_name=target_name,
        d_probe=d_probe,
        concept_set_path=concept_set_path,
        target_layers=target_layers,
        pool_mode=pool_mode,
        save_dir=save_dir,
        use_penultimate=use_penultimate
    )
    

    check_paths = {"clip": paths["clip"], "text": paths["text"]}
    if isinstance(paths["target"], dict):
        check_paths.update(paths["target"])
    else:
        check_paths["target"] = paths["target"]
    
    if _all_saved(check_paths): 
        return

    if _is_biomedclip_name(clip_name):
        clip_model, clip_preprocess, clip_tokenizer = load_biomedclip(device=device)
    else:
        clip_model, clip_preprocess = clip.load(clip_name, device=device)
        clip_tokenizer = None

    if _is_biomedclip_name(target_name):
        target_model, target_preprocess = None, clip_preprocess
    elif target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)

    dataset_name, split = d_probe.split("_")
    data_c = data_utils.get_data(dataset_name, split, clip_preprocess)
    data_t = data_utils.get_data(dataset_name, split, target_preprocess)

    with open(concept_set_path, 'r', encoding='utf-8') as f:
        concepts = json.load(f)["concepts"]

    if _is_biomedclip_name(clip_name):
        save_biomedclip_text_features(clip_model, clip_tokenizer, concepts, paths["text"], batch_size, device)
        save_biomedclip_image_features(clip_model, data_c, paths["clip"], batch_size, device)
    else:
        text_tokens = clip.tokenize(concepts).to(device)
        save_clip_text_features(clip_model, text_tokens, paths["text"], batch_size)
        save_clip_image_features(clip_model, data_c, paths["clip"], batch_size, device)

    if _is_biomedclip_name(target_name):
        return

    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
        visual = target_model.visual
        
        if use_penultimate:
            if hasattr(visual, "attnpool") and hasattr(visual.attnpool, "c_proj"):
                N = visual.attnpool.c_proj.in_features
                identity = torch.nn.Linear(N, N, dtype=torch.float16, device=device)
                torch.nn.init.zeros_(identity.bias)
                identity.weight.data.copy_(torch.eye(N))
                visual.attnpool.c_proj = identity
            elif hasattr(visual, "proj"):
                if isinstance(visual.proj, torch.nn.Parameter):
                    N = visual.proj.shape[0]
                    visual.proj = torch.nn.Parameter(torch.eye(N, dtype=torch.float16, device=device))
                else:
                    N = visual.proj.in_features
                    identity = torch.nn.Linear(N, N, dtype=torch.float16, device=device)
                    torch.nn.init.zeros_(identity.bias)
                    identity.weight.data.copy_(torch.eye(N))
                    visual.proj = identity
            else:
                raise ValueError(f"Unknown CLIP visual model: {target_name}")
            save_clip_rn_penultimate_features(visual.float(), data_t, paths["target"], batch_size, device)
        else: 
            save_clip_image_features(target_model, data_t, paths["target"], batch_size, device)
    
    else:
        save_target_activations(
            target_model, 
            data_t, 
            f"{save_dir}/{d_probe}_{target_name}_" + "{}.pt",
            target_layers, 
            batch_size, 
            device, 
            pool_mode
        )



def filter_and_merge_concepts(
    clip_features, text_features, concepts, concepts_to_class,
    images_to_class_train, Tconf=0.5, Tmerge=0.95, K=4, strategy='max',
    K_indep=5
):
    """
    Step 1: Filter weak concepts by Tconf
    Step 2: Merge similar concepts by Tmerge using greedy coverage
    Step 3: Prune redundant independent concepts per class by top-K mean similarity

    strategy: 'max' or 'median'
        - 'max': select concept with largest coverage set
        - 'median': select concept whose coverage size is median among uncovered
    """
    n_images, n_concepts = clip_features.shape

    # Step 1: Filter weak concepts
    valid_idx = []
    for i, cls in enumerate(concepts_to_class):
        img_idxs = [j for j, c in enumerate(images_to_class_train) if c == cls]
        if len(img_idxs) == 0:
            continue
        
        sims = clip_features[img_idxs, i]
        topk = sims.topk(min(K, sims.shape[0])).values
        mean_topk = topk.mean().item()
        if mean_topk >= Tconf:
            valid_idx.append(i)

    print(f"Initial concepts before filtering: {len(concepts)}")
    # Filter data
    concepts = [concepts[i] for i in valid_idx]
    concepts_to_class = [concepts_to_class[i] for i in valid_idx]
    text_features = text_features[valid_idx]
    clip_features = clip_features[:, valid_idx]
    
    print(f"Remaining concepts after filtering: {len(valid_idx)}")

    # Step 2: Merge similar concepts
    concept_vecs = clip_features.T
    concept_vecs = concept_vecs / concept_vecs.norm(dim=1, keepdim=True)
    sim_matrix = concept_vecs @ concept_vecs.T

    covered_sets = [
        set((sim_matrix[i] >= Tmerge).nonzero(as_tuple=True)[0].tolist())
        for i in range(len(concepts))
    ]
    uncovered = set(range(len(concepts)))
    representative_of = {}
    representatives = []

    while uncovered:
        cover_sizes = [(i, len(covered_sets[i] & uncovered)) for i in uncovered]

        if strategy == 'max':
            best = max(cover_sizes, key=lambda x: x[1])[0]
        elif strategy == 'median':
            sorted_sizes = sorted(size for _, size in cover_sizes)
            median_size = sorted_sizes[len(sorted_sizes) // 2]
            median_candidates = [i for i, size in cover_sizes if size == median_size]
            best = min(median_candidates)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        covers = covered_sets[best] & uncovered
        for c in covers:
            representative_of[c] = best
        representatives.append(best)
        uncovered -= covers

    # Remap indices
    representatives = sorted(set(representatives))
    old_to_new = {old: new_idx for new_idx, old in enumerate(representatives)}
    concept_redirect_map = {
        old_idx: old_to_new[representative_of[old_idx]]
        for old_idx in range(len(concepts))
    }

    # Build structures
    merged_concepts = [concepts[i] for i in representatives]
    merged_text_features = text_features[representatives]

    concept_to_classes = defaultdict(set)
    for old_idx, cls in enumerate(concepts_to_class):
        new_idx = concept_redirect_map[old_idx]
        concept_to_classes[new_idx].add(cls)

    print(f"Remaining concepts after merging: {len(merged_concepts)}")

    # Step 3: For each class, keep top-K independent concepts
    class_to_concepts = defaultdict(list)
    for concept_idx, class_set in concept_to_classes.items():
        for cls in class_set:
            class_to_concepts[cls].append(concept_idx)

    final_representatives = set()

    for cls, concept_idxs in class_to_concepts.items():
        shared = [i for i in concept_idxs if len(concept_to_classes[i]) > 1]
        unique = [i for i in concept_idxs if len(concept_to_classes[i]) == 1]

        img_idxs = [j for j, c in enumerate(images_to_class_train) if c == cls]
        if not img_idxs:
            continue
        image_features = clip_features[img_idxs][:, representatives]  # [N, M]

        concept_scores = []
        for i in unique:
            sim_scores = image_features[:, i]
            mean_score = sim_scores.mean().item()
            concept_scores.append((i, mean_score))

        concept_scores.sort(key=lambda x: -x[1])
        topk_unique = [i for i, _ in concept_scores[:K_indep]]

        final_representatives.update(shared)
        final_representatives.update(topk_unique)

    # Final mapping
    final_representatives = sorted(final_representatives)
    old_to_new_final = {old: i for i, old in enumerate(final_representatives)}

    new_concepts = [merged_concepts[i] for i in final_representatives]
    new_text_features = merged_text_features[final_representatives]

    concept_to_classes_final = {
        old_to_new_final[i]: concept_to_classes[i]
        for i in final_representatives
    }

    concept_redirect_map_final = {}
    for old_idx in range(len(concepts_to_class)):
        rep_idx = representative_of[old_idx]
        if rep_idx in representatives and old_to_new.get(rep_idx) in old_to_new_final:
            concept_redirect_map_final[old_idx] = old_to_new_final[old_to_new[rep_idx]]

    print(f"Remaining concepts after pruning: {len(new_concepts)}")
    print(new_concepts)

    return (
        new_concepts,
        new_text_features,
        concept_to_classes_final,
        concept_redirect_map_final,
    )


def build_class_aware_concept_mask(images_to_class, concept_to_classes, n_concepts=None, device=None):
    """新增函数: build a [n_images, n_concepts] mask for class-valid concepts."""
    if n_concepts is None:
        n_concepts = len(concept_to_classes)
    targets = torch.as_tensor(images_to_class, dtype=torch.long, device=device)
    max_target = int(targets.max().item()) if targets.numel() > 0 else -1
    max_class = max(
        [max(classes) for classes in concept_to_classes.values() if len(classes) > 0],
        default=max_target,
    )
    n_classes = max(max_target, max_class) + 1
    class_concept = torch.zeros((n_concepts, n_classes), dtype=torch.bool, device=device)
    for concept_idx in range(n_concepts):
        valid_classes = concept_to_classes.get(concept_idx, set())
        if len(valid_classes) == 0:
            continue
        class_ids = torch.as_tensor(list(valid_classes), dtype=torch.long, device=device)
        class_concept[concept_idx, class_ids] = True
    return class_concept[:, targets].T


def compute_adaptive_concept_thresholds(
    sim_matrix,
    images_to_class,
    concept_to_classes,
    lambda_std=0.5,
    fallback_threshold=0.20,
    eps=1e-6,
):
    """新增函数: calibrate concept-specific thresholds from class-valid samples."""
    device = sim_matrix.device
    dtype = sim_matrix.dtype
    n_concepts = sim_matrix.shape[1]
    valid_mask = build_class_aware_concept_mask(
        images_to_class, concept_to_classes, n_concepts=n_concepts, device=device
    )
    mu = torch.full((n_concepts,), float(fallback_threshold), dtype=dtype, device=device)
    sigma = torch.zeros((n_concepts,), dtype=dtype, device=device)
    counts = valid_mask.sum(dim=0).to(torch.long)

    for concept_idx in range(n_concepts):
        concept_valid = valid_mask[:, concept_idx]
        if concept_valid.any():
            values = sim_matrix[concept_valid, concept_idx]
            concept_mu = torch.nan_to_num(values.mean(), nan=float(fallback_threshold))
            concept_sigma = torch.nan_to_num(values.std(unbiased=False), nan=0.0).clamp_min(0.0)
            mu[concept_idx] = concept_mu
            sigma[concept_idx] = concept_sigma

    sigma = sigma.clamp_min(eps)
    tau = torch.nan_to_num(mu + lambda_std * sigma, nan=float(fallback_threshold))
    return {
        "mu": mu,
        "sigma": sigma,
        "tau": tau,
        "counts": counts,
        "lambda_std": float(lambda_std),
        "fallback_threshold": float(fallback_threshold),
    }


def generate_concept_labels(
    image_features,
    text_features,
    images_to_class,
    concept_to_classes,
    Tconf=0.9,
    use_soft_concept_labels=False,
    adaptive_threshold=False,
    lambda_std=0.5,
    temperature=0.1,
    threshold_stats=None,
    return_info=False,
):
    """
    Generate class-aware concept labels.

    新增行为:
    - adaptive_threshold=True: use per-concept tau_j = mu_j + lambda_std * sigma_j.
    - use_soft_concept_labels=True: use sigmoid((sim_ij - tau_j) / temperature).
    - class-invalid concepts are always forced to 0.
    """
    sim_matrix = image_features @ text_features.T  # [n_images, n_concepts]
    valid_mask = build_class_aware_concept_mask(
        images_to_class,
        concept_to_classes,
        n_concepts=text_features.size(0),
        device=sim_matrix.device,
    )

    if adaptive_threshold:
        if threshold_stats is None:
            threshold_stats = compute_adaptive_concept_thresholds(
                sim_matrix,
                images_to_class,
                concept_to_classes,
                lambda_std=lambda_std,
                fallback_threshold=Tconf,
            )
        tau = threshold_stats["tau"].to(device=sim_matrix.device, dtype=sim_matrix.dtype)
    else:
        tau = torch.full(
            (text_features.size(0),),
            float(Tconf),
            dtype=sim_matrix.dtype,
            device=sim_matrix.device,
        )
        if threshold_stats is None:
            threshold_stats = {
                "mu": torch.full_like(tau, float("nan")),
                "sigma": torch.zeros_like(tau),
                "tau": tau,
                "counts": valid_mask.sum(dim=0).to(torch.long),
                "lambda_std": float(lambda_std),
                "fallback_threshold": float(Tconf),
            }

    if use_soft_concept_labels:
        temp = max(float(temperature), 1e-6)
        concept_labels = torch.sigmoid((sim_matrix - tau.view(1, -1)) / temp).clamp(0.0, 1.0)
    else:
        concept_labels = (sim_matrix >= tau.view(1, -1)).to(sim_matrix.dtype)

    concept_labels = concept_labels * valid_mask.to(sim_matrix.dtype)

    if return_info:
        return concept_labels, {
            "threshold_stats": threshold_stats,
            "sim_matrix": sim_matrix,
            "valid_mask": valid_mask,
            "binary_labels": (concept_labels >= 0.5).to(sim_matrix.dtype),
        }
    return concept_labels
