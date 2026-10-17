# Shared utilities used by AKI, MSC, JGP, and evaluation.
import os
import json
import re
import sys
import traceback
from datetime import datetime
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import clip

try:
    import networkx as nx
except ImportError:
    nx = None

from config import *
import utils.iterative_label_propagation as ilp_module
from utils.iterative_label_propagation import iterative_label_propagation

def expand_dataset_arg(datasets_arg):
    """Expand dataset CLI argument.

    Examples:
        Cars
        Aircraft/Cars/DTD
        ALL
        ALL_EXCEPT_CARS_IMAGENET
    """
    raw = str(datasets_arg or "").strip()
    if not raw:
        return list(SUPPORTED_DATASETS)
    if raw in DATASET_GROUPS:
        return list(DATASET_GROUPS[raw])

    datasets = [x.strip() for x in raw.split("/") if x.strip()]
    bad = [x for x in datasets if x not in SUPPORTED_DATASETS]
    if bad:
        raise ValueError(
            f"Unsupported dataset(s): {bad}. Supported: {SUPPORTED_DATASETS}. "
            f"Groups: {list(DATASET_GROUPS.keys())}"
        )
    return datasets

def _first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else ""

def resolve_dataset_resource_paths(cli_args, dataset_name):
    """Resolve prompt_json and cupl_json for one dataset.

    Priority:
        1) dataset-specific auto path if it exists
        2) CLI-provided path as fallback
        3) default candidate path, even if missing, so the error is explicit
    """
    qwen_root = getattr(cli_args, "qwen_attr_root", "") or DEFAULT_QWEN_ATTR_ROOT
    cupl_root = getattr(cli_args, "cupl_root", "") or DEFAULT_CUPL_ROOT

    attribute_source = str(
        getattr(cli_args, "attribute_text_source", "cupl") or "cupl"
    ).lower()
    patch_sources = {
        part.strip().lower()
        for part in str(getattr(cli_args, "patch_text_source", "cupl") or "cupl")
        .replace("/", ",")
        .split(",")
        if part.strip()
    }
    qwen_requested = (
        attribute_source == "qwen"
        or "qwen" in patch_sources
        or "both" in patch_sources
        or "all" in patch_sources
        or str(getattr(cli_args, "aki_text_mode", "")).lower() == "qwen_aux_node"
    )
    if qwen_requested:
        auto_prompt = os.path.join(qwen_root, dataset_name, "merged_results.json")
        imagenet_ood = dataset_name in {"ImageNet-A", "ImageNet-V2", "ImageNet-R", "ImageNet-Sketch"}
        source_prompt = os.path.join(qwen_root, "ImageNet", "merged_results.json") if imagenet_ood else ""
        if os.path.exists(auto_prompt):
            prompt_json = auto_prompt
        elif source_prompt and os.path.exists(source_prompt):
            prompt_json = source_prompt
        elif getattr(cli_args, "prompt_json", ""):
            prompt_json = cli_args.prompt_json
        else:
            prompt_json = auto_prompt
    else:
        # The standard paper path is CuPL-only.  An empty value makes it
        # impossible for downstream code to accidentally open merged_results.
        prompt_json = ""

    candidate_names = DATASET_CUPL_CANDIDATES.get(dataset_name, [])
    candidate_paths = [os.path.join(cupl_root, name) for name in candidate_names]
    existing_cupl = _first_existing_path(candidate_paths)
    if existing_cupl and os.path.exists(existing_cupl):
        cupl_json = existing_cupl
    elif getattr(cli_args, "cupl_json", ""):
        cupl_json = cli_args.cupl_json
    else:
        cupl_json = existing_cupl

    print("=" * 80)
    print("Dataset-specific resource paths")
    print("=" * 80)
    print(f"Dataset     : {dataset_name}")
    print(f"Qwen JSON   : {prompt_json if prompt_json else '[unused: CuPL mode]'}")
    print(f"Qwen exist  : {bool(prompt_json) and os.path.exists(prompt_json)}")
    print(f"CuPL JSON   : {cupl_json}")
    print(f"CuPL exist  : {os.path.exists(cupl_json)}")
    print("=" * 80)

    return prompt_json, cupl_json

def _candidate_meta_dirs(cli_args):
    dirs = []

    if getattr(cli_args, "meta_root", ""):
        dirs.append(cli_args.meta_root)

    explicit_meta = getattr(cli_args, "meta_file", "")
    if explicit_meta:
        dirs.append(os.path.dirname(explicit_meta))

    moco_root = (
        getattr(cli_args, "moco_features_path", "")
        or getattr(cli_args, "dino_features_path", "")
        or ""
    )
    if moco_root:
        norm = os.path.normpath(moco_root)
        dirs.append(os.path.dirname(os.path.dirname(norm)))

    features_path = getattr(cli_args, "features_path", "")
    task = getattr(cli_args, "task", "fine_grained")
    if features_path:
        dirs.append(os.path.join(features_path, task))

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out

def resolve_dataset_meta_path(cli_args, dataset_name):
    """Resolve aligned meta path for one dataset.

    Priority:
        1) <meta_root>/<Dataset>_aligned_meta.pt if it exists
        2) directory of explicit --meta_file, but rewritten to current dataset
        3) <moco_root parent parent>/<Dataset>_aligned_meta.pt
        4) <features_path/task>/<Dataset>_aligned_meta.pt
        5) explicit --meta_file as final fallback
    """
    candidate_dirs = _candidate_meta_dirs(cli_args)
    candidate_paths = [os.path.join(d, f"{dataset_name}_aligned_meta.pt") for d in candidate_dirs]
    existing = _first_existing_path(candidate_paths)

    if existing and os.path.exists(existing):
        meta_path = existing
    elif getattr(cli_args, "meta_file", ""):
        meta_path = cli_args.meta_file
    else:
        meta_path = candidate_paths[0] if candidate_paths else ""

    print(f"Meta file   : {meta_path}")
    print(f"Meta exist  : {os.path.exists(meta_path)}")
    print("=" * 80)
    return meta_path

def _normalize_dataset_key_for_baseline(dataset_name):
    key = str(dataset_name).lower().replace("_", "").replace("-", "").replace(" ", "")
    return DATASET_BASELINE_ALIASES.get(key, dataset_name)

def get_external_baseline(dataset_name, shot):
    shot = int(shot)
    dataset_key = _normalize_dataset_key_for_baseline(dataset_name)
    return EXTERNAL_BASELINE.get(shot, {}).get(dataset_key, None)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_label(x):
    x = str(x).lower().strip()
    x = x.replace("_", " ")
    x = " ".join(x.split())
    return x

def rewrite_legacy_path(path, args=None):
    path = str(path)
    if not path:
        return path
    if os.path.exists(path):
        return path

    # Explicit project/data migrations must precede the generic /home aliases.
    # The current AutoDL image stores both projects directly under
    # /root/autodl-tmp and therefore drops the old intermediate MLLM folder.
    prefix_pairs = [
        (
            "/home/chy/chy/AMU-Tuning/AMU-Tuning/datasets/ImageNet2012/",
            "/root/autodl-tmp/datasets/ImageNet2012/",
        ),
        (
            "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/datasets/ImageNet2012/",
            "/root/autodl-tmp/datasets/ImageNet2012/",
        ),
        (
            "/home/chy/chy/AMU-Tuning/AMU-Tuning/",
            "/root/autodl-tmp/AMU-Tuning/AMU-Tuning/",
        ),
        (
            "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/",
            "/root/autodl-tmp/AMU-Tuning/AMU-Tuning/",
        ),
        (
            "/home/chy/chy/MLLM/ECALP-master/",
            "/root/autodl-tmp/ECALP-master/",
        ),
        (
            "/home/zh/chy/chy/MLLM/ECALP-master/",
            "/root/autodl-tmp/ECALP-master/",
        ),
        ("/home/chy/chy/", "/home/zh/chy/chy/"),
        ("/home/zh/chy/chy/", "/home/chy/chy/"),
    ]
    for old_prefix, new_prefix in prefix_pairs:
        if path.startswith(old_prefix):
            candidate = new_prefix + path[len(old_prefix):]
            if os.path.exists(candidate):
                return candidate

    return path

def same_resolved_path(a, b, args=None):
    ra = os.path.normpath(rewrite_legacy_path(str(a), args=args))
    rb = os.path.normpath(rewrite_legacy_path(str(b), args=args))
    return ra == rb

def get_name_variants(name):
    raw = str(name).strip()
    if not raw:
        return []

    synset_match = re.search(r"\bn\d{8}\b", raw)
    synset = synset_match.group(0) if synset_match else ""
    stripped = re.sub(r"^\s*\d+\s*[:\s]\s*", "", raw).strip()
    stripped = re.sub(r"^n\d{8}\s*[:\s,]*", "", stripped).strip()
    variants = [raw, stripped]
    if synset:
        variants.append(synset)
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    parts.extend([x.strip() for x in stripped.split(",") if x.strip()])
    variants.extend(parts)

    out = []
    seen = set()
    for v in variants:
        nv = normalize_label(v)
        if nv and nv not in seen:
            seen.add(nv)
            out.append(v)
    return out

def normalize_cupl_prompts(prompts):
    if isinstance(prompts, list):
        return [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    if isinstance(prompts, str) and prompts.strip():
        return [prompts.strip()]
    return []

def class_prompt_name(class_name):
    variants = get_name_variants(class_name)
    for v in variants:
        if not re.fullmatch(r"n\d{8}", str(v).strip()):
            cleaned = re.sub(r"^n\d{8}\s*[:\s,]*", "", str(v).strip()).strip()
            if cleaned:
                return cleaned
    return str(class_name).strip()

def build_global_prompt_class_map(class_names, source_global_class_ids=None):
    """Map ImageNet global class ids in prompt JSONs to the local OOD label space."""
    if source_global_class_ids is None:
        return {}

    global_ids = torch.as_tensor(source_global_class_ids).reshape(-1).tolist()
    if len(global_ids) != len(class_names):
        print(
            "[Warning] Ignore source_global_class_ids because its length "
            f"{len(global_ids)} != local classes {len(class_names)}"
        )
        return {}

    return {
        int(global_id): normalize_label(class_names[local_id])
        for local_id, global_id in enumerate(global_ids)
    }

def resolve_prompt_class_key(item, class_variant_to_key, global_prompt_class_map):
    """Resolve one prompt to a local class, preferring stable ImageNet ids."""
    for field in ("query_label_id", "label_id", "class_id"):
        value = item.get(field)
        if value is None:
            continue
        try:
            matched_key = global_prompt_class_map.get(int(value))
        except (TypeError, ValueError):
            matched_key = None
        if matched_key is not None:
            return matched_key, "global_id"

    label = normalize_label(item.get("query_label", ""))
    matched_key = class_variant_to_key.get(label)
    if matched_key is not None:
        return matched_key, "class_name"
    return None, "unmatched"

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def dataset_pt_name(dataset_name):
    """ECALP/DMN uses I.pt for ImageNet in some feature folders.

    Important: do not call this function recursively for non-ImageNet datasets.
    Non-ImageNet feature files are saved as <Dataset>.pt.
    """
    names = {
        "ImageNet": "I.pt",
        "ImageNet-A": "A.pt",
        "ImageNet-V2": "V.pt",
        "ImageNet-R": "R.pt",
        "ImageNet-Sketch": "K.pt",
    }
    return names.get(str(dataset_name), f"{dataset_name}.pt")

def dataset_pt_candidates(dataset_name):
    """Return legacy and explicit dataset cache names in priority order.

    ImageNet caches in this project are not named consistently: older ECALP
    exports use I.pt while the aligned CLIP/MoCo extractor writes ImageNet.pt.
    OOD caches have the same short-name/full-name variation.
    """
    candidates = [dataset_pt_name(dataset_name), f"{dataset_name}.pt"]
    out = []
    for name in candidates:
        if name not in out:
            out.append(name)
    return out

def resolve_dataset_pt_path(root, dataset_name, subdir="", description="feature"):
    search_root = os.path.join(root, subdir) if subdir else root
    candidates = [
        os.path.join(search_root, filename)
        for filename in dataset_pt_candidates(dataset_name)
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Resolved {description}: {path}")
            return path
    attempted = "\n  - ".join(candidates)
    raise FileNotFoundError(
        f"{description.capitalize()} file not found. Tried:\n  - {attempted}\n"
        "ImageNet may be stored as either I.pt or ImageNet.pt. "
        "Set the corresponding feature-root argument if both paths are absent."
    )

def l2n(x, dim=-1):
    return F.normalize(x.float(), dim=dim)

def collapse_encoder_output(out, name):
    if isinstance(out, dict):
        for key in ["image_features", "x_norm_clstoken", "cls_token", "features", "last_hidden_state"]:
            if key in out:
                out = out[key]
                break
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name} output is not a tensor")
    if out.dim() == 3:
        if out.shape[1] == 1:
            out = out[:, 0, :]
        else:
            out = out.mean(dim=1)
    if out.dim() == 4:
        out = out.mean(dim=(2, 3))
    if out.dim() != 2:
        raise RuntimeError(f"{name} output must collapse to [B, D], got {tuple(out.shape)}")
    return out

def load_image(path):
    return Image.open(path).convert("RGB")

def solve_clip_to_moco_bridge(clip_features, moco_features, ridge_lambda=1e-2):
    x = l2n(clip_features).float()
    y = l2n(moco_features).float()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Bridge pair count mismatch: CLIP {x.shape[0]} vs MoCo {y.shape[0]}")

    device = x.device
    d = x.shape[1]
    eye = torch.eye(d, dtype=x.dtype, device=device)
    lhs = x.t() @ x + float(ridge_lambda) * eye
    rhs = x.t() @ y
    try:
        w = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        w = torch.linalg.pinv(lhs) @ rhs

    with torch.no_grad():
        aligned = l2n(x @ w)
        target = l2n(y)
        cos = (aligned * target).sum(dim=1)
        print("=" * 80)
        print("Training-free CLIP -> MoCo Linear Bridge")
        print("=" * 80)
        print(f"Pairs           : {x.shape[0]}")
        print(f"CLIP dim        : {x.shape[1]}")
        print(f"MoCo dim        : {y.shape[1]}")
        print(f"Ridge lambda    : {ridge_lambda}")
        print(f"Mean train cos  : {cos.mean().item():.4f}")
        print(f"Median train cos: {cos.median().item():.4f}")
        print("=" * 80)

    return w

def project_clip_to_moco(clip_features, bridge):
    return l2n(clip_features.float() @ bridge.float())

def load_meta(args):
    if not args.meta_file:
        raise ValueError("--meta_file is required for support-patch selection because few_paths and selected_indices are needed.")
    if args.meta_file.endswith(".json"):
        meta = load_json(args.meta_file)
    else:
        meta = safe_torch_load(args.meta_file, map_location="cpu")
    if "few_paths" not in meta or "selected_indices" not in meta:
        raise KeyError("meta_file must contain few_paths and selected_indices.")
    return meta

def encode_texts(model, texts, device, batch_size):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            f = model.encode_text(tokens)
            f = F.normalize(f.float(), dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)

def expand_fewshot_labels_for_aux_nodes(few_shot_labels, num_original_classes, aux_to_class):
    if aux_to_class is None or aux_to_class.numel() == 0:
        return few_shot_labels

    device = few_shot_labels.device
    num_fewshot = few_shot_labels.shape[0]
    num_aux = aux_to_class.numel()

    aux_labels = torch.zeros(num_fewshot, num_aux, dtype=few_shot_labels.dtype, device=device)

    for aux_idx, class_idx in enumerate(aux_to_class):
        aux_labels[:, aux_idx] = few_shot_labels[:, class_idx]

    expanded_labels = torch.cat([few_shot_labels, aux_labels], dim=1)

    print("=" * 80)
    print("Few-shot Label Expansion for Auxiliary Nodes")
    print("=" * 80)
    print(f"Original few-shot labels : {few_shot_labels.shape}")
    print(f"Expanded few-shot labels : {expanded_labels.shape}")
    print("=" * 80)

    return expanded_labels

def merge_aux_node_predictions(predictions, num_original_classes, aux_to_class, merge_mode="sum"):
    if aux_to_class is None or aux_to_class.numel() == 0:
        return predictions[:, :num_original_classes]

    original_scores = predictions[:, :num_original_classes]
    aux_scores = predictions[:, num_original_classes:]
    merged_scores = original_scores.clone()

    if merge_mode == "sum":
        for aux_idx, class_idx in enumerate(aux_to_class):
            merged_scores[:, class_idx] += aux_scores[:, aux_idx]

    elif merge_mode == "max":
        for aux_idx, class_idx in enumerate(aux_to_class):
            merged_scores[:, class_idx] = torch.maximum(merged_scores[:, class_idx], aux_scores[:, aux_idx])

    elif merge_mode == "mean":
        count = torch.ones(num_original_classes, dtype=merged_scores.dtype, device=merged_scores.device)
        for aux_idx, class_idx in enumerate(aux_to_class):
            merged_scores[:, class_idx] += aux_scores[:, aux_idx]
            count[class_idx] += 1.0
        merged_scores = merged_scores / count[None, :]

    else:
        raise ValueError(f"Unknown aux_merge_mode: {merge_mode}")

    return merged_scores

def merge_text_aux_to_class_features(text_features, num_original_classes, aux_to_class):
    base = text_features[:num_original_classes].clone()

    if aux_to_class is None or aux_to_class.numel() == 0 or text_features.shape[0] == num_original_classes:
        return F.normalize(base.float(), dim=-1)

    aux = text_features[num_original_classes:]
    count = torch.ones(num_original_classes, 1, device=text_features.device)

    for aux_idx, class_idx in enumerate(aux_to_class):
        base[class_idx] += aux[aux_idx]
        count[class_idx] += 1.0

    class_text = base / count
    return F.normalize(class_text.float(), dim=-1)

def parse_float_list(text):
    if not text:
        return []
    out = []
    for part in str(text).replace("/", ",").split(","):
        item = part.strip()
        if not item:
            continue
        out.append(float(item))
    return out

def aki_thresholds_for_shot(shot):
    configured = SEARCH_CONFIG.get("aki_margin_threshold_by_shot", {}).get(int(shot), [0.0])
    values = configured if isinstance(configured, (list, tuple)) else [configured]
    return [float(value) for value in values] or [0.0]

def synchronize_device(device):
    """Synchronize CUDA only when timing GPU work."""
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)

__all__ = ['expand_dataset_arg', '_first_existing_path', 'resolve_dataset_resource_paths', '_candidate_meta_dirs', 'resolve_dataset_meta_path', '_normalize_dataset_key_for_baseline', 'get_external_baseline', 'load_json', 'normalize_label', 'rewrite_legacy_path', 'same_resolved_path', 'get_name_variants', 'normalize_cupl_prompts', 'class_prompt_name', 'build_global_prompt_class_map', 'resolve_prompt_class_key', 'safe_torch_load', 'dataset_pt_name', 'dataset_pt_candidates', 'resolve_dataset_pt_path', 'l2n', 'collapse_encoder_output', 'load_image', 'solve_clip_to_moco_bridge', 'project_clip_to_moco', 'load_meta', 'encode_texts', 'expand_fewshot_labels_for_aux_nodes', 'merge_aux_node_predictions', 'merge_text_aux_to_class_features', 'parse_float_list', 'aki_thresholds_for_shot', 'synchronize_device', 'torch', 'nn', 'F', 'Image', 'tqdm', 'clip', 'nx', 'ilp_module', 'iterative_label_propagation']
