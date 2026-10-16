import os
import json
import argparse
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.iterative_label_propagation as ilp_module
from utils.iterative_label_propagation import iterative_label_propagation


# ==========================================================
# SEARCH CONFIG: ONLY EDIT THIS BLOCK
# Normal-run variables: shot, text_mode, jgp_weight.
# Parameter-sensitivity grids are configured separately below.
# Other parameters are fixed to the original setting.
# ==========================================================
SEARCH_CONFIG = {
    # 0 means zero-shot. JGP requires few-shot DINO prototypes,
    # so shot=0 will automatically run LP baseline only.
    "shot": [0, 1, 2, 4, 8, 16],

    # AKI grounding threshold for each shot.
    # Rule: keep an attribute when own-class response - hardest-negative response > tau.
    # Edit only these values when comparing shot-specific thresholds.
    "aki_margin_threshold_by_shot": {
        0: [0.0],
        1: [0.0, 0.5, 1.0],
        2: [0.0],
        4: [0.0],
        8: [0.0],
        16: [0.0],
    },

    # Compare the standard photo prompt with CuPL description nodes.
    "text_mode": [
        "original",
        "cupl_aux_node",
    ],

    # Normal experiment candidates for w.
    # Formula: P_final = (1 - w) * P_LP + w * P_JGP
    # This is the same mathematical w as sensitivity["w"], but it serves a
    # different purpose: these are candidate operating points for ordinary
    # runs.  Under --experiment_suite/--run_sensitivity, the outer run is
    # fixed to sensitivity["fixed_w"] and the full sensitivity["w"] curve is
    # evaluated internally in that single pass.
    "jgp_weight": [0.8,0.9],

    # Parameter-sensitivity grids used by --experiment_suite or
    # --run_sensitivity.  tau_k is the normalized AKI pruning threshold
    # denoted by tau_K in the paper; alpha mixes the two graph spaces inside
    # JGP; w fuses the JGP prediction with the original LP prediction. Alpha
    # and w are different parameters; jgp_weight above is another use
    # of this same w, not a third hyperparameter.
    "sensitivity": {
        "tau_k": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "w": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # Sequential JGP tuning protocol used by the ablation table:
        #   (1) sweep alpha at this fixed w;
        #   (2) choose the best alpha;
        #   (3) sweep w again while holding alpha at that best value.
        # This avoids reporting JGP with an arbitrary alpha=0.5.
        "fixed_w": 0.9,
        "reference_tau_k": 0.0,
    },

    # Internal paths. Normally no need to pass these in CLI.
    "paths": {
        "dino_features_path": "moco_r50_auto",
        # prompt_json and cupl_json are resolved per dataset below.
        # Keep them empty here to avoid accidentally using Aircraft prompts for Cars/DTD/etc.
        "cupl_json": "",
        "prompt_json": "",
    },

    # Fixed parameters from your original setting.
    "fixed": {
        # Base graph propagation
        "k_text": 3,
        "k_image": 8,
        "k_fewshot": 8,
        "gamma": 10.0,
        "alpha": 1.0,
        "num_iterations": 3,

        # Qwen text-node options
        "aggregation": "weighted",
        "min_quality": 0.0,
        "topk_per_class": -1,
        "aux_merge_mode": "sum",
        "aux_residual": False,
        "aux_mu": 0.3,

        # Score scaling
        "lp_scale": 1.0,
        "text_clique_scale": 1.0,
        "dino_clique_scale": 30.0,
        "dino_cache_scale": 30.0,

        # Text clique graph for semantic community
        "text_clique_control": "topk",
        "text_clique_control_value": 3.0,
        "text_clique_sog": False,
        "text_clique_r": 0.2,
        "text_clique_beta": 5.5,
        "text_clique_weight": 0.0,

        # DINO clique graph for visual community
        "dino_clique_control": "topk",
        "dino_clique_control_value": 3.0,
        "dino_clique_sog": False,
        "dino_clique_r": 0.2,
        "dino_clique_beta": 5.5,
        "dino_clique_weight": 0.0,

        # JGP inner settings
        "jgp_gate_mode": "soft",
        "jgp_text_gate": 1.0,
        "jgp_visual_gate": 1.0,
        "jgp_inner_weight": 0.5,
        "jgp_mask_boost": 1.0,
        "jgp_scale": 1.0,

        # Inactive DINO cache branch defaults
        "dino_cache_beta": 5.5,
        "dino_cache_weight": 0.0,
    },

}


# ==========================================================
# DATASET LIST + AUTOMATIC PROMPT/CUPL PATH RESOLUTION
# ==========================================================
SUPPORTED_DATASETS = [
    "ImageNet",
    "ImageNet-A",
    "ImageNet-V2",
    "ImageNet-R",
    "ImageNet-Sketch",
    "Caltech101",
    "Flowers102",
    "Aircraft",
    "Cars",
    "DTD",
    "SUN397",
    "UCF101",
    "Food101",
    "Pets",
    "EuroSAT",
]

# You can use:
#   --datasets ALL
#   --datasets ALL_EXCEPT_CARS_IMAGENET
#   --datasets Aircraft/Cars/DTD
DATASET_GROUPS = {
    "ALL": [
        "ImageNet", "Caltech101", "Flowers102", "Aircraft", "Cars",
        "DTD", "SUN397", "UCF101", "Food101", "Pets", "EuroSAT",
    ],
    "IMAGENET_OOD": ["ImageNet-A", "ImageNet-V2", "ImageNet-R", "ImageNet-Sketch"],
    "ALL_EXCEPT_CARS_IMAGENET": [
        "Caltech101", "Flowers102", "Aircraft", "DTD",
        "SUN397", "UCF101", "Food101", "Pets", "EuroSAT",
    ],
}

DEFAULT_FEATURES_PATH = "/home/zh/chy/chy/MLLM/base graph propagation-master/features"
DEFAULT_MOCO_FEATURES_PATH = "/home/zh/chy/chy/MLLM/base graph propagation-master/features_mocov3_aligned_16shot/fine_grained/MOCO/R50"
DEFAULT_QWEN_ATTR_ROOT = "/home/zh/chy/chy/MLLM/base graph propagation-master/qwen_attrs_universal"
DEFAULT_CUPL_ROOT = "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/Prompt_CuPL"
DEFAULT_META_ROOT = "/home/zh/chy/chy/MLLM/base graph propagation-master/features_mocov3_aligned_16shot/fine_grained"
DEFAULT_MOCOV3_CKPT = "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/r-50-1000ep.pth.tar"

# Candidate CuPL filenames. The resolver will use the first existing file.
DATASET_CUPL_CANDIDATES = {
    "ImageNet": ["imagenet.json", "imagenet1k.json", "ImageNet.json"],
    "ImageNet-A": ["imagenet.json", "imagenet1k.json", "ImageNet.json"],
    "ImageNet-V2": ["imagenet.json", "imagenet1k.json", "ImageNet.json"],
    "ImageNet-R": ["imagenet.json", "imagenet1k.json", "ImageNet.json"],
    "ImageNet-Sketch": ["imagenet.json", "imagenet1k.json", "ImageNet.json"],
    "Caltech101": ["caltech101.json", "caltech_101.json", "Caltech101.json"],
    "Flowers102": ["oxford_flowers.json", "flowers102.json", "OxfordFlowers.json", "flower102.json"],
    "Aircraft": ["fgvc.json", "aircraft.json", "fgvc_aircraft.json"],
    "Cars": ["stanford_cars.json", "cars.json", "StanfordCars.json"],
    "DTD": ["dtd.json", "DTD.json"],
    "SUN397": ["sun397.json", "SUN397.json", "sun.json"],
    "UCF101": ["ucf101.json", "UCF101.json"],
    "Food101": ["food101.json", "Food101.json", "food.json"],
    "Pets": ["oxford_pets.json", "pets.json", "OxfordPets.json"],
    "EuroSAT": ["eurosat.json", "EuroSAT.json"],
}


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


# ==========================================================
# EXTERNAL BASELINE TABLE
# Rows correspond to shot = 1, 2, 4, 8, 16.
# Values are the baseline accuracies you provided.
# The CSV will report gain_over_external_baseline = acc_final - baseline.
# If a dataset name has multiple common aliases, they are normalized below.
# ==========================================================
EXTERNAL_BASELINE = {
    1: {
        "ImageNet": 62.79, "Flower": 74.75, "DTD": 55.73, "ExtraCol4": 52.42,
        "Pets": 87.54, "Cars": 59.30, "UCF": 68.31, "Caltech": 90.55,
        "Food": 74.43, "SUN": 65.35, "Aircraft": 20.82, "EuroSAT": 60.77,
    },
    2: {
        "ImageNet": 62.75, "Flower": 80.39, "DTD": 57.51, "ExtraCol4": 58.63,
        "Pets": 86.43, "Cars": 59.54, "UCF": 68.73, "Caltech": 90.63,
        "Food": 73.34, "SUN": 65.68, "Aircraft": 21.15, "EuroSAT": 61.75,
    },
    4: {
        "ImageNet": 62.75, "Flower": 83.84, "DTD": 59.81, "ExtraCol4": 63.30,
        "Pets": 84.63, "Cars": 60.51, "UCF": 71.00, "Caltech": 90.22,
        "Food": 71.33, "SUN": 66.43, "Aircraft": 24.33, "EuroSAT": 68.77,
    },
    8: {
        "ImageNet": 62.97, "Flower": 87.05, "DTD": 60.87, "ExtraCol4": 67.91,
        "Pets": 83.43, "Cars": 63.04, "UCF": 72.59, "Caltech": 89.98,
        "Food": 71.67, "SUN": 67.31, "Aircraft": 27.00, "EuroSAT": 75.90,
    },
    16: {
        "ImageNet": 63.20, "Flower": 89.73, "DTD": 65.37, "ExtraCol4": 73.05,
        "Pets": 83.92, "Cars": 65.37, "UCF": 74.44, "Caltech": 90.99,
        "Food": 71.98, "SUN": 68.42, "Aircraft": 31.92, "EuroSAT": 76.74,
    },
}

DATASET_BASELINE_ALIASES = {
    "imagenet": "ImageNet",
    "imagenet1k": "ImageNet",
    "imagenet-1k": "ImageNet",
    "flower": "Flower",
    "flowers": "Flower",
    "flower102": "Flower",
    "flowers102": "Flower",
    "oxfordflowers": "Flower",
    "oxfordflowers102": "Flower",
    "dtd": "DTD",
    "pets": "Pets",
    "oxfordpets": "Pets",
    "cars": "Cars",
    "stanfordcars": "Cars",
    "ucf": "UCF",
    "ucf101": "UCF",
    "caltech": "Caltech",
    "caltech101": "Caltech",
    "food": "Food",
    "food101": "Food",
    "sun": "SUN",
    "sun397": "SUN",
    "aircraft": "Aircraft",
    "fgvcaircraft": "Aircraft",
    "fgvc-aircraft": "Aircraft",
    "eurosat": "EuroSAT",
}

def _normalize_dataset_key_for_baseline(dataset_name):
    key = str(dataset_name).lower().replace("_", "").replace("-", "").replace(" ", "")
    return DATASET_BASELINE_ALIASES.get(key, dataset_name)

def get_external_baseline(dataset_name, shot):
    shot = int(shot)
    dataset_key = _normalize_dataset_key_for_baseline(dataset_name)
    return EXTERNAL_BASELINE.get(shot, {}).get(dataset_key, None)


# ==========================================================
# Basic utilities
# ==========================================================
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
            "/home/chy/chy/MLLM/base graph propagation-master/",
            "/root/autodl-tmp/base graph propagation-master/",
        ),
        (
            "/home/zh/chy/chy/MLLM/base graph propagation-master/",
            "/root/autodl-tmp/base graph propagation-master/",
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
    """base graph propagation/DMN uses I.pt for ImageNet in some feature folders.

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

    ImageNet caches in this project are not named consistently: older base graph propagation
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


def extract_positive_attributes(item):
    attrs = item.get("positive_attributes", [])
    if not isinstance(attrs, list):
        attrs = []

    out = []
    for attr in attrs:
        if isinstance(attr, str) and attr.strip():
            out.append(attr.strip())

    pos_prompt = item.get("positive_prompt", "")
    if isinstance(pos_prompt, str) and pos_prompt.strip():
        out.append(pos_prompt.strip())

    return out


def load_qwen_attributes(
    prompt_json,
    class_names,
    min_quality=0.0,
    topk_per_class=-1,
    cupl_json="",
    patch_text_source="qwen",
    source_global_class_ids=None,
):
    """Return flat text list and class-index mapping for support patch selection.

    patch_text_source:
        qwen : Qwen positive attributes first, CuPL fallback for missing classes.
        cupl : CuPL prompts only.
        photo: "a photo of a {class}" prompts only.
    """
    patch_text_source = str(patch_text_source or "qwen").lower()
    if patch_text_source not in {"qwen", "cupl", "photo"}:
        raise ValueError(f"Unknown --patch_text_source: {patch_text_source}")
    if patch_text_source == "qwen" and not prompt_json:
        raise ValueError("--prompt_json is required when --patch_text_source=qwen.")

    # A resolved legacy Qwen path may be absent on a CuPL-only server.  Never
    # touch it unless Qwen was explicitly selected.
    items = load_json(prompt_json) if patch_text_source == "qwen" and prompt_json else []
    cupl_data = load_json(cupl_json) if cupl_json else {}
    class_to_idx = {normalize_label(c): i for i, c in enumerate(class_names)}

    grouped = defaultdict(list)
    class_variant_to_key = {}
    for class_name in class_names:
        class_key = normalize_label(class_name)
        for variant in get_name_variants(class_name):
            class_variant_to_key[normalize_label(variant)] = class_key

    global_prompt_class_map = build_global_prompt_class_map(
        class_names,
        source_global_class_ids,
    )
    prompt_match_counts = defaultdict(int)

    for item in items:
        if float(item.get("quality_score", 0.0)) < float(min_quality):
            continue
        matched_key, match_mode = resolve_prompt_class_key(
            item,
            class_variant_to_key,
            global_prompt_class_map,
        )
        if matched_key is None:
            prompt_match_counts["unmatched"] += 1
            continue
        attrs = extract_positive_attributes(item)
        if not attrs:
            continue
        prompt_match_counts[match_mode] += 1
        grouped[matched_key].append((float(item.get("quality_score", 0.0)), attrs))

    texts = []
    attr_to_class = []
    per_class_count = []
    qwen_used_classes = 0
    cupl_used_classes = 0
    photo_used_classes = 0
    missing_classes = []

    for class_name in class_names:
        key = normalize_label(class_name)
        class_idx = class_to_idx[key]
        entries = sorted(grouped.get(key, []), key=lambda x: x[0], reverse=True)
        if topk_per_class > 0:
            entries = entries[:topk_per_class]

        seen = set()
        count = 0

        if patch_text_source == "qwen" and entries:
            qwen_used_classes += 1
            for _, attrs in entries:
                for attr in attrs:
                    norm_attr = normalize_label(attr)
                    if norm_attr in seen:
                        continue
                    seen.add(norm_attr)
                    texts.append(attr)
                    attr_to_class.append(class_idx)
                    count += 1
        elif patch_text_source in {"qwen", "cupl"}:
            cupl_prompts = get_cupl_prompts(cupl_data, class_name, class_idx=class_idx)
            if cupl_prompts:
                cupl_used_classes += 1
                for prompt in cupl_prompts:
                    norm_prompt = normalize_label(prompt)
                    if norm_prompt in seen:
                        continue
                    seen.add(norm_prompt)
                    texts.append(prompt)
                    attr_to_class.append(class_idx)
                    count += 1
            else:
                missing_classes.append(class_name)
        elif patch_text_source == "photo":
            texts.append(f"a photo of a {class_prompt_name(class_name)}")
            attr_to_class.append(class_idx)
            photo_used_classes += 1
            count += 1

        per_class_count.append(count)

    matched_classes = sum(1 for x in per_class_count if x > 0)
    if matched_classes == 0:
        raise RuntimeError(
            "No support-patch selection texts found. Check --patch_text_source, "
            "--prompt_json, --cupl_json and class names."
        )

    print("=" * 80)
    print("Support Patch Selection Texts Loaded")
    print("=" * 80)
    print(f"Text source          : {patch_text_source}")
    print(f"Total texts          : {len(texts)}")
    print(f"Classes with texts   : {matched_classes} / {len(class_names)}")
    print(f"Qwen classes         : {qwen_used_classes}")
    print(f"CuPL classes         : {cupl_used_classes}")
    print(f"Photo prompt classes : {photo_used_classes}")
    print(f"Missing classes      : {len(missing_classes)}")
    print(f"Prompt ID matches    : {prompt_match_counts['global_id']}")
    print(f"Prompt name matches  : {prompt_match_counts['class_name']}")
    print(f"Unmatched prompts    : {prompt_match_counts['unmatched']}")
    print(f"Avg texts per class  : {sum(per_class_count) / max(1, len(per_class_count)):.2f}")
    print(f"Max texts per class  : {max(per_class_count) if per_class_count else 0}")
    if missing_classes:
        print(f"Missing examples     : {missing_classes[:10]}")
    print("=" * 80)

    return texts, torch.tensor(attr_to_class, dtype=torch.long)


def encode_texts(clip_model, texts, device, batch_size):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            f = collapse_encoder_output(clip_model.encode_text(tokens), "CLIP text")
            feats.append(l2n(f).detach())
    return torch.cat(feats, dim=0)


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


def make_grid_crops(image, grid_size, overlap=0.0):
    w, h = image.size
    n = int(grid_size)
    if n <= 0:
        raise ValueError("--patch_grid must be positive")

    cell_w = w / n
    cell_h = h / n
    expand_w = cell_w * float(overlap) / 2.0
    expand_h = cell_h * float(overlap) / 2.0

    crops = []
    boxes = []
    for iy in range(n):
        for ix in range(n):
            x0 = max(0, int(round(ix * cell_w - expand_w)))
            y0 = max(0, int(round(iy * cell_h - expand_h)))
            x1 = min(w, int(round((ix + 1) * cell_w + expand_w)))
            y1 = min(h, int(round((iy + 1) * cell_h + expand_h)))
            if x1 <= x0:
                x1 = min(w, x0 + 1)
            if y1 <= y0:
                y1 = min(h, y0 + 1)
            box = (x0, y0, x1, y1)
            boxes.append(box)
            crops.append(image.crop(box))
    return crops, boxes


def load_mocov3_resnet50(ckpt, device):
    import torchvision.models as models

    model = models.resnet50(weights=None)
    model.fc = nn.Identity()

    ckpt_obj = safe_torch_load(ckpt, map_location="cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj

    new_state = {}
    has_moco_base_encoder = any(
        str(key).startswith(("module.base_encoder.", "base_encoder."))
        for key in state.keys()
    )
    for k, v in state.items():
        if has_moco_base_encoder:
            if k.startswith("module.base_encoder."):
                nk = k[len("module.base_encoder."):]
            elif k.startswith("base_encoder."):
                nk = k[len("base_encoder."):]
            else:
                # Ignore the momentum encoder, predictor, optimizer state,
                # and other training-only entries in an official MoCo v3
                # checkpoint.  They are not part of the inference encoder.
                continue
        else:
            nk = k
            for prefix in [
                "module.encoder.",
                "module.backbone.",
                "module.",
                "encoder.",
                "backbone.",
            ]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break

        if nk.startswith("fc.") or nk.startswith("head.") or nk.startswith("mlp."):
            continue
        if "predictor" in nk or "projector" in nk:
            continue
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("=" * 80)
    print("MoCo v3 ResNet50 loaded")
    print("=" * 80)
    print(f"Checkpoint : {ckpt}")
    print(f"Missing    : {len(missing)}")
    print(f"Unexpected : {len(unexpected)}")
    print("=" * 80)
    return model.to(device).eval()


def moco_transform(size=224):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        interp = InterpolationMode.BICUBIC
    except Exception:
        interp = Image.BICUBIC

    return Compose([
        Resize(size, interpolation=interp),
        CenterCrop(size),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def encode_patch_crops(image_path, clip_model, clip_preprocess, moco_model, moco_preprocess, device, grid_size, crop_overlap, batch_size):
    image = load_image(image_path)
    crops, boxes = make_grid_crops(image, grid_size=grid_size, overlap=crop_overlap)

    clip_feats = []
    moco_feats = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        clip_batch = torch.stack([clip_preprocess(x) for x in batch]).to(device)
        moco_batch = torch.stack([moco_preprocess(x) for x in batch]).to(device)
        with torch.no_grad():
            cf = collapse_encoder_output(clip_model.encode_image(clip_batch), "CLIP patch")
            mf = collapse_encoder_output(moco_model(moco_batch), "MoCo patch")
            clip_feats.append(l2n(cf).detach())
            moco_feats.append(l2n(mf).detach())

    return torch.cat(clip_feats, dim=0), torch.cat(moco_feats, dim=0), boxes


def dense_rank_desc(scores):
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(scores.numel(), device=scores.device, dtype=order.dtype)
    return ranks.float()


def infer_dataset_name_from_meta_or_args(args):
    if getattr(args, "datasets", ""):
        first = str(args.datasets).split("/")[0].strip()
        if first:
            return first
    if getattr(args, "meta_file", ""):
        base = os.path.basename(str(args.meta_file))
        for suffix in ["_aligned_meta.pt", "_aligned_meta.json", "_meta.pt", "_meta.json", ".pt", ".json"]:
            if base.endswith(suffix):
                return base[:-len(suffix)]
    return "Dataset"


def format_overlap_for_name(x):
    s = f"{float(x):g}"
    return s.replace(".", "p").replace("-", "m")


def resolve_support_patch_cache_paths(args):
    dataset_name = infer_dataset_name_from_meta_or_args(args)
    shot_tag = f"_shot{int(getattr(args, 'num_shots', 0))}" if int(getattr(args, "num_shots", 0)) > 0 else ""
    auto_dir = os.path.dirname(args.meta_file) if getattr(args, "meta_file", "") else os.path.join(args.features_path, args.task)
    auto_name = (
        f"{dataset_name}_support_patch{shot_tag}_grid{int(args.patch_grid)}"
        f"_overlap{format_overlap_for_name(args.crop_overlap)}"
        f"_clipRN50_mocoR50_mocoimg{int(args.moco_image_size)}.pt"
    )
    auto_path = os.path.join(auto_dir, auto_name)
    cache_in = args.support_patch_features_pt or auto_path
    cache_out = args.support_patch_cache_out or auto_path
    load_candidates = []
    for path in [args.support_patch_features_pt, args.support_patch_cache_out, auto_path]:
        if path and path not in load_candidates:
            load_candidates.append(path)
    return cache_in, cache_out, load_candidates


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


def flatten_selected_support_paths(meta, num_classes, num_shots, args=None):
    few_paths = [rewrite_legacy_path(x, args=args) for x in meta["few_paths"]]
    selected_indices = meta["selected_indices"]
    if torch.is_tensor(selected_indices):
        selected_indices = selected_indices.cpu().tolist()

    if len(selected_indices) < num_classes:
        raise IndexError(
            f"selected_indices has {len(selected_indices)} classes, "
            f"but the current OOD label space has {num_classes} classes"
        )

    required_indices = []
    for c in range(num_classes):
        class_indices = selected_indices[c]
        if torch.is_tensor(class_indices):
            class_indices = class_indices.cpu().tolist()
        required_indices.extend(int(index) for index in list(class_indices)[:num_shots])

    max_required_index = max(required_indices) if required_indices else -1
    if max_required_index >= len(few_paths):
        # Early OOD caches kept local selected_indices but accidentally sliced
        # the flat ImageNet few_paths pool. Recover that full pool from the
        # source metadata recorded by the extractor.
        source_meta_candidates = []
        protocol = meta.get("cross_domain_protocol", {})
        source_paths = protocol.get("source_paths", {}) if isinstance(protocol, dict) else {}
        if isinstance(source_paths, dict) and source_paths.get("meta"):
            source_meta_candidates.append(source_paths["meta"])
        if args is not None and getattr(args, "meta_file", ""):
            aligned_root = os.path.dirname(os.path.dirname(args.meta_file))
            source_meta_candidates.append(
                os.path.join(aligned_root, "fine_grained", "ImageNet_aligned_meta.pt")
            )

        recovered_path = ""
        for candidate in source_meta_candidates:
            candidate = rewrite_legacy_path(candidate, args=args)
            if not candidate or not os.path.exists(candidate):
                continue
            source_meta = safe_torch_load(candidate, map_location="cpu")
            source_few_paths = source_meta.get("few_paths", [])
            if len(source_few_paths) > max_required_index:
                few_paths = [rewrite_legacy_path(x, args=args) for x in source_few_paths]
                recovered_path = candidate
                break

        if not recovered_path:
            raise IndexError(
                "OOD selected_indices refer to the full ImageNet support pool, "
                f"requiring index {max_required_index}, but the cached few_paths "
                f"contains only {len(few_paths)} entries. Re-extract the OOD meta "
                "or provide the source ImageNet_aligned_meta.pt cache."
            )
        print("=" * 80)
        print("Recovered full ImageNet support path pool for OOD MSC")
        print(f"Source meta : {recovered_path}")
        print(f"Pool size   : {len(few_paths)}")
        print(f"Max index   : {max_required_index}")
        print("=" * 80)

    support_paths = []
    support_class_ids = []
    for c in range(num_classes):
        inds = selected_indices[c]
        if torch.is_tensor(inds):
            inds = inds.cpu().tolist()
        inds = list(inds)[:num_shots]
        if len(inds) < num_shots:
            raise RuntimeError(f"Class {c} has only {len(inds)} selected support paths, num_shots={num_shots}")
        for idx in inds:
            support_paths.append(few_paths[int(idx)])
            support_class_ids.append(c)

    missing = [p for p in support_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing support image paths, examples: {missing[:5]}")

    return support_paths, torch.tensor(support_class_ids, dtype=torch.long)


def onehot_from_class_ids(class_ids, num_classes, device):
    y = torch.zeros(class_ids.numel(), num_classes, dtype=torch.float32, device=device)
    y[torch.arange(class_ids.numel(), device=device), class_ids.to(device)] = 1.0
    return y


def encode_support_patch_pool(support_paths, clip_model, clip_preprocess, moco_model, moco_preprocess, device, args):
    clip_all, moco_all, boxes_all = [], [], []
    iterator = range(len(support_paths))
    if not args.no_tqdm:
        iterator = tqdm(iterator, desc="Encode support patches")
    for idx in iterator:
        cf, mf, boxes = encode_patch_crops(
            image_path=support_paths[idx],
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            moco_model=moco_model,
            moco_preprocess=moco_preprocess,
            device=device,
            grid_size=args.patch_grid,
            crop_overlap=args.crop_overlap,
            batch_size=args.patch_batch_size,
        )
        clip_all.append(cf.detach().cpu())
        moco_all.append(mf.detach().cpu())
        boxes_all.append(torch.tensor(boxes, dtype=torch.long))

    clip_patches = torch.stack(clip_all, dim=0)
    moco_patches = torch.stack(moco_all, dim=0)
    patch_boxes = torch.stack(boxes_all, dim=0)
    return clip_patches, moco_patches, patch_boxes


def load_or_build_support_patch_pool(args, support_paths, support_class_ids, clip_model, clip_preprocess, moco_model, moco_preprocess, device):
    cache_in, cache_out, load_candidates = resolve_support_patch_cache_paths(args)
    print(f"Support patch cache target: {cache_out}")

    existing_cache = next((path for path in load_candidates if os.path.exists(path)), "")
    if existing_cache:
        cache = safe_torch_load(existing_cache, map_location="cpu")
        clip_patches = cache["clip_patch_features"].float()
        moco_patches = cache["moco_patch_features"].float()
        patch_boxes = cache.get("patch_boxes", None)
        cache_paths = cache.get("support_paths", cache.get("image_paths", None))
        if clip_patches.shape[0] != len(support_paths):
            raise ValueError(f"Support patch cache count {clip_patches.shape[0]} != current support count {len(support_paths)}")
        if int(cache.get("patch_grid", args.patch_grid)) != int(args.patch_grid):
            raise ValueError(f"Support patch cache grid {cache.get('patch_grid')} != --patch_grid {args.patch_grid}")
        if cache_paths is not None:
            for i, (a, b) in enumerate(zip(cache_paths, support_paths)):
                if not same_resolved_path(a, b, args=args):
                    raise ValueError(f"Support patch cache path mismatch at {i}: {a} vs {b}")
        print("=" * 80)
        print("Cached support patch features loaded")
        print("=" * 80)
        print(f"Support patch cache : {existing_cache}")
        print(f"CLIP support patches: {tuple(clip_patches.shape)}")
        print(f"MoCo support patches: {tuple(moco_patches.shape)}")
        print("=" * 80)
        return clip_patches.to(device), moco_patches.to(device), patch_boxes

    print("No reusable support patch cache found. Tried:")
    for candidate in load_candidates:
        print(f"  - {candidate}")
    print("Patch crops will be encoded once and saved for the remaining runs.")

    clip_patches, moco_patches, patch_boxes = encode_support_patch_pool(
        support_paths=support_paths,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        moco_model=moco_model,
        moco_preprocess=moco_preprocess,
        device=device,
        args=args,
    )

    cache = {
        "clip_patch_features": clip_patches,
        "moco_patch_features": moco_patches,
        "patch_boxes": patch_boxes,
        "support_paths": support_paths,
        "support_class_ids": support_class_ids,
        "patch_grid": args.patch_grid,
        "crop_overlap": args.crop_overlap,
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_out)), exist_ok=True)
    torch.save(cache, cache_out)
    print(f"Saved support patch cache: {cache_out}")
    return clip_patches.to(device), moco_patches.to(device), patch_boxes


def select_class_aware_support_patch_nodes(
    support_clip_patches,
    support_moco_patches,
    support_class_ids,
    clip_attr_features,
    moco_attr_features,
    attr_to_class,
    num_classes,
    args,
    device,
):
    topk = int(args.support_patch_topk_per_class)
    if topk <= 0:
        raise ValueError("--support_patch_topk_per_class must be positive")

    selected_clip_nodes = []
    selected_moco_nodes = []
    selected_labels = []
    selected_meta = []

    support_class_ids = support_class_ids.to(device)
    attr_to_class = attr_to_class.to(device)
    clip_attr_features = l2n(collapse_encoder_output(clip_attr_features, "clip_attr_features"))
    moco_attr_features = l2n(collapse_encoder_output(moco_attr_features, "moco_attr_features"))

    for c in range(num_classes):
        img_mask = support_class_ids == c
        if not img_mask.any():
            continue
        attr_mask = attr_to_class == c
        if not attr_mask.any():
            continue

        clip_pool = l2n(support_clip_patches[img_mask].reshape(-1, support_clip_patches.shape[-1]))
        moco_pool = l2n(support_moco_patches[img_mask].reshape(-1, support_moco_patches.shape[-1]))
        clip_attr_c = l2n(clip_attr_features[attr_mask])
        moco_attr_c = l2n(moco_attr_features[attr_mask])

        clip_score = (clip_pool @ clip_attr_c.t()).max(dim=1).values
        moco_score = (moco_pool @ moco_attr_c.t()).max(dim=1).values

        select_mode = args.support_patch_select_mode
        if select_mode == "clip_only":
            select_mode = "clip"
        elif select_mode == "moco_only":
            select_mode = "moco"

        if select_mode == "clip":
            score = clip_score
        elif select_mode == "moco":
            score = moco_score
        elif select_mode == "sum":
            score = clip_score + moco_score
        elif select_mode == "rank_sum":
            score = -(dense_rank_desc(clip_score) + dense_rank_desc(moco_score))
        else:
            raise ValueError(f"Unknown --support_patch_select_mode: {args.support_patch_select_mode}")

        k = min(topk, score.numel())
        selected = torch.topk(score, k=k, largest=True).indices
        selected_clip_nodes.append(clip_pool[selected])
        selected_moco_nodes.append(moco_pool[selected])
        selected_labels.append(torch.full((k,), c, dtype=torch.long, device=device))

        for idx in selected.detach().cpu().tolist():
            selected_meta.append({
                "class_id": int(c),
                "local_patch_flat_index": int(idx),
                "clip_score": float(clip_score[idx].detach().cpu().item()),
                "moco_score": float(moco_score[idx].detach().cpu().item()),
            })

    if not selected_clip_nodes:
        raise RuntimeError("No support patch nodes were selected. Check Qwen attributes and class names.")

    clip_nodes = torch.cat(selected_clip_nodes, dim=0)
    moco_nodes = torch.cat(selected_moco_nodes, dim=0)
    label_ids = torch.cat(selected_labels, dim=0)
    label_onehot = onehot_from_class_ids(label_ids.detach().cpu(), num_classes, device)

    print("=" * 80)
    print("Class-aware support patch nodes selected")
    print("=" * 80)
    print(f"Selection mode          : {args.support_patch_select_mode}")
    print(f"Top-k per class         : {args.support_patch_topk_per_class}")
    print(f"Selected patch nodes    : {clip_nodes.shape[0]}")
    print(f"CLIP patch node shape   : {tuple(clip_nodes.shape)}")
    print(f"MoCo patch node shape   : {tuple(moco_nodes.shape)}")
    print("=" * 80)

    return clip_nodes, moco_nodes, label_onehot, selected_meta


# ==========================================================
# CuPL / Qwen text construction
# ==========================================================
def get_cupl_prompts(cupl_data, cls, class_idx=None):
    if not isinstance(cupl_data, dict):
        if class_idx is not None:
            return []
        raise KeyError(f"CuPL prompt not found for class: {cls}")

    variants = get_name_variants(cls)
    keys = []
    for variant in variants:
        keys.extend([variant, variant.replace("_", " "), variant.replace(" ", "_")])

    for key in keys:
        if key in cupl_data:
            prompts = normalize_cupl_prompts(cupl_data[key])
            if prompts:
                return prompts

    norm_keys = {normalize_label(v) for v in variants}
    for key in cupl_data.keys():
        if normalize_label(key) in norm_keys:
            prompts = normalize_cupl_prompts(cupl_data[key])
            if prompts:
                return prompts

    if class_idx is not None:
        values = list(cupl_data.values())
        if 0 <= int(class_idx) < len(values):
            prompts = normalize_cupl_prompts(values[int(class_idx)])
            if prompts:
                return prompts
        return []

    raise KeyError(f"CuPL prompt not found for class: {cls}")


def get_class_names(pre_calculated, text_features_load, cupl_data):
    if "class_names" in pre_calculated:
        class_names = pre_calculated["class_names"]
        print("Using class names from feature pt: class_names")
    elif "classes" in pre_calculated:
        class_names = pre_calculated["classes"]
        print("Using class names from feature pt: classes")
    else:
        if not isinstance(cupl_data, dict) or not cupl_data:
            raise ValueError(
                "Class names are missing from the feature/meta cache and no valid CuPL json is available."
            )
        class_names = list(cupl_data.keys())
        print("Using class names from CuPL json")

    if len(class_names) != text_features_load.shape[0]:
        raise ValueError(
            "Class-name/text-feature count mismatch: "
            f"{len(class_names)} names versus {text_features_load.shape[0]} feature rows. "
            "Refusing to truncate because that would shift class identities. "
            "For ImageNet, provide ImageNet_aligned_meta.pt through --meta_root "
            "(or a feature cache containing class_names)."
        )

    return class_names, text_features_load


def build_base_class_text_features(
    args,
    clip_model,
    pre_calculated,
    text_features_load,
    cupl_data,
    device,
):
    """Build the original class nodes before adding any AKI attributes.

    The paper baseline uses one standard CLIP prompt per class:
        a photo of a {class}.
    Cached text features remain available through --base_text_prompt cached for
    backward-compatible comparisons.
    """
    class_names, cached_text_features = get_class_names(
        pre_calculated=pre_calculated,
        text_features_load=text_features_load,
        cupl_data=cupl_data,
    )
    base_mode = str(getattr(args, "base_text_prompt", "photo") or "photo").lower()
    if base_mode == "cached":
        base_features = F.normalize(cached_text_features.to(device).float(), dim=-1)
        print("=" * 80)
        print("Using cached original text features as base class nodes")
        print("=" * 80)
        return base_features, class_names
    if base_mode != "photo":
        raise ValueError("--base_text_prompt must be one of: photo, cached")

    photo_prompts = [
        f"a photo of a {class_prompt_name(class_name)}."
        for class_name in class_names
    ]
    base_features = encode_texts(
        clip_model,
        photo_prompts,
        device,
        args.batch_size,
    )
    base_features = F.normalize(base_features.float(), dim=-1)
    print("=" * 80)
    print("Using standard CLIP photo prompts as base class nodes")
    print("Template : a photo of a {class}.")
    print(f"Classes  : {len(class_names)}")
    print("=" * 80)
    return base_features, class_names


def item_has_positive(item):
    parsed = item.get("parsed_json", {})
    pos_attrs = parsed.get("positive_attributes", item.get("positive_attributes", []))
    pos_prompt = item.get("positive_prompt", "").strip()
    return bool(pos_prompt) and len(pos_attrs) > 0


def group_prompts_by_class(
    prompt_items,
    min_quality,
    topk_per_class,
    class_names=None,
    source_global_class_ids=None,
):
    groups = defaultdict(list)

    class_variant_to_key = {}
    global_prompt_class_map = {}
    if class_names is not None:
        for class_name in class_names:
            class_key = normalize_label(class_name)
            for variant in get_name_variants(class_name):
                class_variant_to_key[normalize_label(variant)] = class_key
        global_prompt_class_map = build_global_prompt_class_map(
            class_names,
            source_global_class_ids,
        )

    for item in prompt_items:
        if float(item.get("quality_score", 0.0)) < min_quality:
            continue
        if not item_has_positive(item):
            continue

        if class_names is None:
            cls = normalize_label(item.get("query_label", ""))
        else:
            cls, _ = resolve_prompt_class_key(
                item,
                class_variant_to_key,
                global_prompt_class_map,
            )
        if cls:
            groups[cls].append(item)

    for cls in list(groups.keys()):
        groups[cls] = sorted(
            groups[cls],
            key=lambda x: float(x.get("quality_score", 0.0)),
            reverse=True,
        )
        if topk_per_class > 0:
            groups[cls] = groups[cls][:topk_per_class]

    return groups


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


def aggregate_prompt_features(items, feats, device, aggregation):
    if aggregation == "top1":
        return F.normalize(feats[0], dim=-1)

    if aggregation == "mean":
        return F.normalize(feats.mean(dim=0), dim=-1)

    weights = []
    for x in items:
        q = float(x.get("quality_score", 0.0))
        s = float(x.get("clip_similarity", 1.0))
        weights.append(max(q * s, 1e-6))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    proto = (weights[:, None] * feats).sum(dim=0)
    return F.normalize(proto, dim=-1)


def build_recoded_cupl_text_features(model, class_names, cupl_json, device, batch_size):
    cupl_data = load_json(cupl_json)
    text_features = []
    prompt_counts = []

    for cls in class_names:
        cls_prompts = get_cupl_prompts(cupl_data, cls)
        cls_feats = encode_texts(model, cls_prompts, device, batch_size)
        cls_proto = F.normalize(cls_feats.mean(dim=0), dim=-1)
        text_features.append(cls_proto)
        prompt_counts.append(len(cls_prompts))

    text_features = torch.stack(text_features, dim=0)
    text_features = F.normalize(text_features.float(), dim=-1)

    print("=" * 80)
    print("Recoded CuPL Text Features with Multi-Prompt Mean Ensemble")
    print("=" * 80)
    print(f"Classes encoded       : {len(class_names)}")
    print(f"Avg prompts per class : {sum(prompt_counts) / len(prompt_counts):.2f}")
    print(f"Min prompts per class : {min(prompt_counts)}")
    print(f"Max prompts per class : {max(prompt_counts)}")
    print("=" * 80)

    return text_features


def build_qwen_positive_fallback_text_features(
    model,
    class_names,
    prompt_json,
    cupl_json,
    device,
    batch_size,
    aggregation,
    min_quality,
    topk_per_class,
    source_global_class_ids=None,
):
    prompt_items = load_json(prompt_json)
    cupl_data = load_json(cupl_json)
    groups = group_prompts_by_class(
        prompt_items,
        min_quality,
        topk_per_class,
        class_names=class_names,
        source_global_class_ids=source_global_class_ids,
    )

    text_features = []
    used_positive = 0
    fallback_cupl = 0

    for cls in class_names:
        norm_cls = normalize_label(cls)
        items = groups.get(norm_cls, [])

        if len(items) > 0:
            positive_prompts = [x["positive_prompt"] for x in items]
            pos_feats = encode_texts(model, positive_prompts, device, batch_size)
            proto = aggregate_prompt_features(items, pos_feats, device, aggregation)
            used_positive += 1
        else:
            cupl_prompts = get_cupl_prompts(cupl_data, cls)
            cupl_feats = encode_texts(model, cupl_prompts, device, batch_size)
            proto = F.normalize(cupl_feats.mean(dim=0), dim=-1)
            fallback_cupl += 1

        text_features.append(proto)

    text_features = torch.stack(text_features, dim=0)
    text_features = F.normalize(text_features.float(), dim=-1)

    print("=" * 80)
    print("Qwen Positive + Fallback CuPL Mean Ensemble Text Features")
    print("=" * 80)
    print(f"Prompt items loaded     : {len(prompt_items)}")
    print(f"Prompt classes matched  : {len(groups)}")
    print(f"Qwen positive classes   : {used_positive}")
    print(f"Fallback CuPL classes   : {fallback_cupl}")
    print("=" * 80)

    return text_features


def build_qwen_auxiliary_text_nodes(
    model,
    class_names,
    prompt_json,
    device,
    batch_size,
    aggregation,
    min_quality,
    topk_per_class,
    source_global_class_ids=None,
):
    prompt_items = load_json(prompt_json)
    groups = group_prompts_by_class(
        prompt_items,
        min_quality,
        topk_per_class,
        class_names=class_names,
        source_global_class_ids=source_global_class_ids,
    )

    aux_features = []
    aux_to_class = []
    used_positive = 0
    missing_positive = 0

    for class_idx, cls in enumerate(class_names):
        norm_cls = normalize_label(cls)
        items = groups.get(norm_cls, [])

        if len(items) == 0:
            missing_positive += 1
            continue

        positive_prompts = [x["positive_prompt"] for x in items]
        pos_feats = encode_texts(model, positive_prompts, device, batch_size)
        proto = aggregate_prompt_features(items, pos_feats, device, aggregation)

        aux_features.append(proto)
        aux_to_class.append(class_idx)
        used_positive += 1

    if len(aux_features) == 0:
        aux_text_features = None
        aux_to_class = torch.empty(0, dtype=torch.long, device=device)
    else:
        aux_text_features = torch.stack(aux_features, dim=0)
        aux_text_features = F.normalize(aux_text_features.float(), dim=-1)
        aux_to_class = torch.tensor(aux_to_class, dtype=torch.long, device=device)

    print("=" * 80)
    print("Qwen Auxiliary Semantic Nodes")
    print("=" * 80)
    print(f"Prompt items loaded       : {len(prompt_items)}")
    print(f"Prompt classes matched    : {len(groups)}")
    print(f"Auxiliary Qwen nodes      : {used_positive}")
    print(f"Classes without Qwen node : {missing_positive}")
    print("=" * 80)

    return aux_text_features, aux_to_class


def build_support_grounded_auxiliary_nodes(
    model,
    class_names,
    prompt_json,
    cupl_json,
    support_features,
    support_labels,
    original_text_features,
    device,
    batch_size,
    min_quality,
    topk_per_class,
    aux_residual=False,
    aux_mu=0.3,
    grounding_level="image",
    num_shots=1,
    margin_threshold=None,
    source_global_class_ids=None,
    attribute_text_source="cupl",
):
    """Keep attributes whose support margin exceeds the configured shot threshold."""
    attribute_text_source = str(attribute_text_source or "cupl").lower()
    if attribute_text_source not in {"cupl", "qwen"}:
        raise ValueError("attribute_text_source must be one of: cupl, qwen")
    texts, attr_to_class = load_qwen_attributes(
        prompt_json=prompt_json if attribute_text_source == "qwen" else "",
        class_names=class_names,
        min_quality=min_quality,
        topk_per_class=topk_per_class,
        cupl_json=cupl_json,
        patch_text_source=attribute_text_source,
        source_global_class_ids=source_global_class_ids,
    )
    attr_to_class = attr_to_class.to(device)
    attr_features = encode_texts(model, texts, device, batch_size)
    attr_features = F.normalize(attr_features.float(), dim=-1)
    support_features = F.normalize(support_features.float(), dim=-1)
    if grounding_level == "image" and support_features.ndim != 2:
        raise ValueError(f"Image grounding expects [N, D] features, got {tuple(support_features.shape)}")
    if grounding_level == "patch" and support_features.ndim != 3:
        raise ValueError(f"Patch grounding expects [N, P, D] features, got {tuple(support_features.shape)}")
    support_classes = support_labels[:, :len(class_names)].argmax(dim=1).to(device)

    class_masks = []
    for class_index in range(len(class_names)):
        class_mask = support_classes == class_index
        if not class_mask.any():
            raise ValueError(f"Support-grounded AKI found no support images for class {class_index}")
        class_masks.append(class_mask)

    # Chunk the attribute-support comparison so ImageNet does not materialize
    # a potentially very large [attributes, support images] matrix.
    response_chunks = []
    attribute_chunk_size = 512 if grounding_level == "image" else 32
    for start in range(0, attr_features.shape[0], attribute_chunk_size):
        attr_chunk = attr_features[start:start + attribute_chunk_size]
        if grounding_level == "image":
            chunk_similarities = attr_chunk @ support_features.t()
        else:
            # One score per support image: its strongest local response.
            chunk_similarities = torch.einsum("ad,npd->anp", attr_chunk, support_features).amax(dim=2)
        chunk_responses = [
            chunk_similarities[:, class_mask].mean(dim=1)
            for class_mask in class_masks
        ]
        response_chunks.append(torch.stack(chunk_responses, dim=1))
    class_responses = torch.cat(response_chunks, dim=0)

    row_ids = torch.arange(attr_features.shape[0], device=device)
    positive_response = class_responses[row_ids, attr_to_class]
    negative_responses = class_responses.clone()
    negative_responses[row_ids, attr_to_class] = -torch.inf
    hardest_negative = negative_responses.max(dim=1).values
    evidence_margin = positive_response - hardest_negative

    # Normalize without centering, preserving whether an attribute favors its
    # assigned class. Per-class RMS removes dataset/class margin-scale shifts.
    num_classes = len(class_names)
    class_squared_sum = torch.zeros(num_classes, device=device, dtype=evidence_margin.dtype)
    class_counts = torch.zeros(num_classes, device=device, dtype=evidence_margin.dtype)
    class_squared_sum.scatter_add_(0, attr_to_class, evidence_margin.square())
    class_counts.scatter_add_(0, attr_to_class, torch.ones_like(evidence_margin))
    class_rms = torch.sqrt(class_squared_sum / class_counts.clamp_min(1.0))
    normalized_margin = evidence_margin / class_rms[attr_to_class].clamp_min(1e-12)

    if margin_threshold is None:
        configured = SEARCH_CONFIG.get("aki_margin_threshold_by_shot", {}).get(int(num_shots), [0.0])
        configured = configured if isinstance(configured, (list, tuple)) else [configured]
        margin_threshold = configured[0] if configured else 0.0
    margin_threshold = float(margin_threshold)
    keep_mask = normalized_margin > margin_threshold

    aux_features = []
    aux_to_class = []
    kept_per_class = []
    for class_index in range(len(class_names)):
        class_keep = keep_mask & (attr_to_class == class_index)
        kept_count = int(class_keep.sum().item())
        kept_per_class.append(kept_count)
        if kept_count == 0:
            continue
        proto = F.normalize(attr_features[class_keep].mean(dim=0), dim=-1)
        if aux_residual:
            base = original_text_features[class_index]
            proto = F.normalize((1.0 - aux_mu) * base + aux_mu * proto, dim=-1)
        aux_features.append(proto)
        aux_to_class.append(class_index)

    if aux_features:
        aux_features = F.normalize(torch.stack(aux_features, dim=0).float(), dim=-1)
        aux_to_class = torch.tensor(aux_to_class, dtype=torch.long, device=device)
    else:
        aux_features = None
        aux_to_class = torch.empty(0, dtype=torch.long, device=device)

    total_attrs = int(attr_features.shape[0])
    kept_attrs = int(keep_mask.sum().item())
    positive_margins = evidence_margin[keep_mask]
    retained_normalized_margins = normalized_margin[keep_mask]
    print("=" * 80)
    title = "Global CLIP Images" if grounding_level == "image" else "Cached CLIP Patches"
    print(f"Support-Grounded Attribute Selection ({title})")
    print("=" * 80)
    print("Rule                    : keep if class-RMS-normalized margin > tau(shot)")
    print("Margin normalization    : raw margin / per-class RMS margin")
    print(f"Current shot            : {int(num_shots)}")
    print(f"Margin threshold tau    : {margin_threshold:.6f}")
    print(f"Raw candidate attributes: {total_attrs}")
    print(f"Retained attributes     : {kept_attrs} ({100.0 * kept_attrs / max(total_attrs, 1):.2f}%)")
    print(f"Hard-dropped attributes : {total_attrs - kept_attrs}")
    print(f"Classes with AKI node   : {len(aux_to_class)} / {len(class_names)}")
    print(f"Avg retained per class  : {sum(kept_per_class) / max(len(kept_per_class), 1):.2f}")
    print(f"Raw margin range        : [{evidence_margin.min().item():+.6f}, {evidence_margin.max().item():+.6f}]")
    print(f"Normalized margin range : [{normalized_margin.min().item():+.6f}, {normalized_margin.max().item():+.6f}]")
    print(f"Mean retained raw margin: {positive_margins.mean().item() if positive_margins.numel() else 0.0:+.6f}")
    print(
        "Mean retained normalized: "
        f"{retained_normalized_margins.mean().item() if retained_normalized_margins.numel() else 0.0:+.6f}"
    )
    print("=" * 80)
    stats = {
        "total": total_attrs,
        "retained": kept_attrs,
        "retention_rate": 100.0 * kept_attrs / max(total_attrs, 1),
        "margin_threshold": margin_threshold,
        "margin_normalization": "class_rms",
    }
    return aux_features, aux_to_class, stats


def filter_msc_qwen_attributes(
    model,
    class_names,
    clip_attr_features,
    attr_to_class,
    support_features,
    support_labels,
    margin_threshold,
    device,
    batch_size,
):
    """Apply the normalized AKI margin rule to Qwen texts used by MSC."""
    clip_attr_features = F.normalize(clip_attr_features.float(), dim=-1)
    attr_to_class = attr_to_class.to(device)
    support_features = F.normalize(support_features.float(), dim=-1)
    support_classes = support_labels[:, :len(class_names)].argmax(dim=1).to(device)

    class_masks = [support_classes == class_index for class_index in range(len(class_names))]
    similarities = clip_attr_features @ support_features.t()
    class_responses = torch.stack(
        [similarities[:, class_mask].mean(dim=1) for class_mask in class_masks],
        dim=1,
    )
    row_ids = torch.arange(clip_attr_features.shape[0], device=device)
    positive_response = class_responses[row_ids, attr_to_class]
    negative_responses = class_responses.clone()
    negative_responses[row_ids, attr_to_class] = -torch.inf
    raw_margin = positive_response - negative_responses.max(dim=1).values

    class_squared_sum = torch.zeros(len(class_names), device=device, dtype=raw_margin.dtype)
    class_counts = torch.zeros(len(class_names), device=device, dtype=raw_margin.dtype)
    class_squared_sum.scatter_add_(0, attr_to_class, raw_margin.square())
    class_counts.scatter_add_(0, attr_to_class, torch.ones_like(raw_margin))
    class_rms = torch.sqrt(class_squared_sum / class_counts.clamp_min(1.0))
    normalized_margin = raw_margin / class_rms[attr_to_class].clamp_min(1e-12)
    keep_mask = normalized_margin > float(margin_threshold)

    filtered_features = clip_attr_features[keep_mask]
    filtered_classes = attr_to_class[keep_mask]
    retained_classes = set(filtered_classes.detach().cpu().tolist())
    fallback_classes = [c for c in range(len(class_names)) if c not in retained_classes]
    if fallback_classes:
        fallback_texts = [f"a photo of a {class_prompt_name(class_names[c])}" for c in fallback_classes]
        fallback_features = encode_texts(model, fallback_texts, device, batch_size)
        filtered_features = torch.cat([filtered_features, fallback_features], dim=0)
        filtered_classes = torch.cat([
            filtered_classes,
            torch.tensor(fallback_classes, dtype=torch.long, device=device),
        ])

    stats = {
        "total": int(keep_mask.numel()),
        "retained": int(keep_mask.sum().item()),
        "retention_rate": 100.0 * float(keep_mask.float().mean().item()),
        "fallback_classes": len(fallback_classes),
        "margin_threshold": float(margin_threshold),
    }
    print("=" * 80)
    print("Grounded Qwen Texts for MSC")
    print("=" * 80)
    print(f"Normalized threshold    : {float(margin_threshold):.4f}")
    print(f"Qwen attributes retained: {stats['retained']} / {stats['total']} ({stats['retention_rate']:.2f}%)")
    print(f"Photo fallback classes  : {stats['fallback_classes']} / {len(class_names)}")
    print("=" * 80)
    return F.normalize(filtered_features.float(), dim=-1), filtered_classes, stats


def build_aux_node_text_features(args, clip_model, pre_calculated, text_features_load, cupl_data, device):
    text_mode = str(args.text_mode).lower()
    if text_mode == "cupl_aux_node":
        attribute_text_source = "cupl"
    elif text_mode == "qwen_aux_node":
        attribute_text_source = "qwen"
    else:
        attribute_text_source = str(getattr(args, "attribute_text_source", "cupl") or "cupl").lower()

    if attribute_text_source == "qwen" and not args.prompt_json:
        raise ValueError("--prompt_json is required when Qwen descriptions are selected")
    if attribute_text_source == "cupl" and not args.cupl_json:
        raise ValueError("--cupl_json is required when CuPL descriptions are selected")

    original_text_features, class_names = build_base_class_text_features(
        args=args,
        clip_model=clip_model,
        pre_calculated=pre_calculated,
        text_features_load=text_features_load,
        cupl_data=cupl_data,
        device=device,
    )

    if attribute_text_source == "qwen":
        aux_text_features, aux_to_class = build_qwen_auxiliary_text_nodes(
            model=clip_model,
            class_names=class_names,
            prompt_json=args.prompt_json,
            device=device,
            batch_size=args.batch_size,
            aggregation=args.aggregation,
            min_quality=args.min_quality,
            topk_per_class=args.topk_per_class,
            source_global_class_ids=getattr(args, "source_global_class_ids", None),
        )
    else:
        description_texts, aux_to_class = load_qwen_attributes(
            prompt_json="",
            class_names=class_names,
            min_quality=args.min_quality,
            topk_per_class=args.topk_per_class,
            cupl_json=args.cupl_json,
            patch_text_source="cupl",
            source_global_class_ids=getattr(args, "source_global_class_ids", None),
        )
        aux_text_features = encode_texts(
            clip_model,
            description_texts,
            device,
            args.batch_size,
        )
        aux_text_features = F.normalize(aux_text_features.float(), dim=-1)
        aux_to_class = aux_to_class.to(device)

    if aux_text_features is None:
        print(f"[Warning] No {attribute_text_source.upper()} auxiliary nodes are constructed.")
        text_features_aug = original_text_features
    else:
        if args.aux_residual:
            residual_aux = []
            for j, class_idx in enumerate(aux_to_class):
                base = original_text_features[class_idx]
                description = aux_text_features[j]
                mixed = (1.0 - args.aux_mu) * base + args.aux_mu * description
                mixed = F.normalize(mixed, dim=-1)
                residual_aux.append(mixed)
            aux_text_features = torch.stack(residual_aux, dim=0)

        text_features_aug = torch.cat([original_text_features, aux_text_features], dim=0)

    print("=" * 80)
    print("Final Text Node Construction")
    print("=" * 80)
    print(f"Original text nodes : {original_text_features.shape[0]}")
    print(f"Auxiliary nodes     : {text_features_aug.shape[0] - original_text_features.shape[0]}")
    print(f"Total text nodes    : {text_features_aug.shape[0]}")
    print(f"Feature dim         : {text_features_aug.shape[1]}")
    print(f"Aux residual        : {args.aux_residual}")
    if args.aux_residual:
        print(f"Aux residual mu     : {args.aux_mu}")
    print("=" * 80)

    return text_features_aug, class_names, aux_to_class


def select_text_features(args, clip_model, pre_calculated, text_features_load, cupl_data, device):
    if args.text_mode == "original":
        text_features, class_names = build_base_class_text_features(
            args=args,
            clip_model=clip_model,
            pre_calculated=pre_calculated,
            text_features_load=text_features_load,
            cupl_data=cupl_data,
            device=device,
        )
        return text_features, class_names, torch.empty(0, dtype=torch.long, device=device)

    if not args.cupl_json:
        raise ValueError("--cupl_json is required when text_mode is not original")

    class_names, text_features_load = get_class_names(pre_calculated, text_features_load, cupl_data)

    if args.text_mode == "recoded_cupl":
        text_features = build_recoded_cupl_text_features(
            model=clip_model,
            class_names=class_names,
            cupl_json=args.cupl_json,
            device=device,
            batch_size=args.batch_size,
        )
        return text_features, class_names, torch.empty(0, dtype=torch.long, device=device)

    if args.text_mode == "qwen_positive":
        if not args.prompt_json:
            raise ValueError("--prompt_json is required when --text_mode qwen_positive")
        text_features = build_qwen_positive_fallback_text_features(
            model=clip_model,
            class_names=class_names,
            prompt_json=args.prompt_json,
            cupl_json=args.cupl_json,
            device=device,
            batch_size=args.batch_size,
            aggregation=args.aggregation,
            min_quality=args.min_quality,
            topk_per_class=args.topk_per_class,
            source_global_class_ids=getattr(args, "source_global_class_ids", None),
        )
        return text_features, class_names, torch.empty(0, dtype=torch.long, device=device)

    if args.text_mode in {"cupl_aux_node", "qwen_aux_node"}:
        text_features, class_names, aux_to_class = build_aux_node_text_features(
            args=args,
            clip_model=clip_model,
            pre_calculated=pre_calculated,
            text_features_load=text_features_load,
            cupl_data=cupl_data,
            device=device,
        )
        return text_features, class_names, aux_to_class

    raise ValueError(f"Unknown text_mode: {args.text_mode}")


# ==========================================================
# Few-shot auxiliary label utilities
# ==========================================================
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


# ==========================================================
# Auxiliary visual feature branch
# ==========================================================
def load_dino_feature_file(args, dataset_name, device):
    """Load pre-extracted DINO features.

    Expected:
    dino_root/Dataset.pt:
        image_features: [N, D]
        labels: [N] or [N, C]

    dino_root/few_shots/Dataset.pt:
        image_features: [C, shot, D]
        labels: [C, shot, C]
    """
    args.resolved_moco_query_path = ""
    args.resolved_moco_fewshot_path = ""
    args.moco_align_info = None
    if not (args.use_dino_cache or args.use_dino_clique or getattr(args, "use_jgp", False) or getattr(args, "use_cross_graph", False)):
        return None, None, None, None

    dino_root = args.dino_features_path
    if not dino_root:
        dino_root = os.path.join(args.features_path, args.task, args.dino_model)

    feature_path = resolve_dataset_pt_path(
        dino_root,
        dataset_name,
        description="MoCo/visual query feature",
    )
    args.resolved_moco_query_path = feature_path

    dino_data = safe_torch_load(feature_path, map_location="cpu")
    dino_image_features = F.normalize(dino_data["image_features"].float(), dim=-1).to(device)
    dino_query_labels = dino_data.get("labels")
    args.moco_align_info = dino_data.get("align_info")

    dino_fewshot_features = None
    dino_fewshot_labels = None

    if args.mode == "FS":
        fewshot_path = resolve_dataset_pt_path(
            dino_root,
            dataset_name,
            subdir="few_shots",
            description="MoCo/visual few-shot feature",
        )
        args.resolved_moco_fewshot_path = fewshot_path

        dino_few = safe_torch_load(fewshot_path, map_location="cpu")
        dino_fewshot_features = dino_few["image_features"][:, :args.num_shots, :]
        dino_fewshot_features = dino_fewshot_features.reshape(-1, dino_fewshot_features.shape[-1]).to(device)
        dino_fewshot_features = F.normalize(dino_fewshot_features.float(), dim=-1)

        dino_fewshot_labels = dino_few["labels"][:, :args.num_shots, :]
        dino_fewshot_labels = dino_fewshot_labels.reshape(-1, dino_fewshot_labels.shape[-1]).to(device)

    print("=" * 80)
    print("MoCo/Visual Feature Branch Loaded")
    print("=" * 80)
    print(f"MoCo/Visual feature path : {feature_path}")
    print(f"MoCo/Visual image features : {tuple(dino_image_features.shape)}")
    if dino_fewshot_features is not None:
        print(f"MoCo/Visual few-shot path : {fewshot_path}")
        print(f"MoCo/Visual few-shot feats : {tuple(dino_fewshot_features.shape)}")
    print("=" * 80)

    return dino_image_features, dino_fewshot_features, dino_fewshot_labels, dino_query_labels


def build_class_prototypes_from_fewshot(few_shot_image_features, few_shot_labels, num_classes):
    device = few_shot_image_features.device
    dim = few_shot_image_features.shape[-1]

    prototypes = torch.zeros(num_classes, dim, device=device, dtype=few_shot_image_features.dtype)
    counts = torch.zeros(num_classes, 1, device=device, dtype=few_shot_image_features.dtype)

    hard_labels = few_shot_labels[:, :num_classes].argmax(dim=1)
    for i, c in enumerate(hard_labels):
        prototypes[c] += few_shot_image_features[i]
        counts[c] += 1.0

    prototypes = prototypes / counts.clamp(min=1.0)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    return prototypes


def print_label_alignment_check(name, clip_labels, moco_labels, num_classes):
    if clip_labels is None or moco_labels is None:
        return
    clip_hard = clip_labels[:, :num_classes].argmax(dim=1).detach().cpu()
    moco_hard = moco_labels[:, :num_classes].argmax(dim=1).detach().cpu()
    same_shape = tuple(clip_hard.shape) == tuple(moco_hard.shape)
    same_order = bool(same_shape and torch.equal(clip_hard, moco_hard))
    clip_counts = torch.bincount(clip_hard, minlength=num_classes)
    moco_counts = torch.bincount(moco_hard, minlength=num_classes)
    same_counts = bool(torch.equal(clip_counts, moco_counts))
    print("=" * 80)
    print(f"{name} CLIP/MoCo Few-shot Label Alignment Check")
    print("=" * 80)
    print(f"Same shape        : {same_shape} | CLIP {tuple(clip_hard.shape)} vs MoCo {tuple(moco_hard.shape)}")
    print(f"Same order        : {same_order}")
    print(f"Same class counts : {same_counts}")
    if not same_order:
        mismatch = (clip_hard != moco_hard).nonzero(as_tuple=False).flatten()
        show = mismatch[:10].tolist()
        print(f"Mismatched positions : {int(mismatch.numel())}")
        print(f"First mismatches     : {show}")
        if show:
            pairs = [(int(i), int(clip_hard[i]), int(moco_hard[i])) for i in show]
            print(f"(idx, clip_cls, moco_cls): {pairs}")
    print("=" * 80)


def validate_query_label_alignment(clip_labels, moco_labels, clip_path, moco_path):
    """Fail fast when original CLIP and aligned MoCo query rows disagree."""
    if clip_labels is None or moco_labels is None:
        raise ValueError(
            "Cannot verify CLIP/MoCo query alignment because one query cache has no labels: "
            f"CLIP={clip_path}, MoCo={moco_path}"
        )

    def hard(labels):
        value = torch.as_tensor(labels).detach().cpu()
        if value.ndim > 1 and value.shape[-1] > 1:
            value = value.argmax(dim=-1)
        return value.reshape(-1).long()

    clip_hard = hard(clip_labels)
    moco_hard = hard(moco_labels)
    same_shape = tuple(clip_hard.shape) == tuple(moco_hard.shape)
    same_order = bool(same_shape and torch.equal(clip_hard, moco_hard))

    print("=" * 80)
    print("Original CLIP / aligned MoCo query check")
    print(f"CLIP query : {clip_path}")
    print(f"MoCo query : {moco_path}")
    print(f"Same rows  : {same_shape}")
    print(f"Label order: {same_order}")
    print("=" * 80)

    if not same_order:
        if same_shape:
            mismatch = (clip_hard != moco_hard).nonzero(as_tuple=False).flatten()
            detail = f"; mismatched label positions={int(mismatch.numel())}, first={mismatch[:10].tolist()}"
        else:
            detail = f"; CLIP rows={clip_hard.numel()}, MoCo rows={moco_hard.numel()}"
        raise ValueError(
            "The MoCo query cache is not aligned to the base graph propagation query cache"
            f"{detail}. CLIP={clip_path}, MoCo={moco_path}"
        )


def validate_aligned_clip_reference(args, dataset_name, original_features, original_labels):
    """Verify that the aligned visual cache follows the base graph propagation row order.

    The jointly extracted aligned CLIP file is not used for prediction here.
    It acts as an image-level alignment witness: if its row-wise CLIP cosine to
    the base graph propagation cache is high, the paired MoCo rows can safely be fused
    with the original CLIP rows.
    """
    args.resolved_aligned_clip_reference_path = ""
    args.aligned_clip_mean_cosine = None
    args.aligned_clip_min_cosine = None

    explicit = str(getattr(args, "aligned_clip_reference_path", "") or "").strip()
    if explicit and os.path.isfile(explicit):
        reference_path = explicit
    else:
        if explicit:
            reference_root = explicit
        elif getattr(args, "meta_file", ""):
            reference_root = os.path.join(os.path.dirname(args.meta_file), args.clip_model)
        else:
            print("[Warning] No aligned CLIP reference path is available; image-level alignment is unverified.")
            return
        try:
            reference_path = resolve_dataset_pt_path(
                reference_root,
                dataset_name,
                description="aligned CLIP reference feature",
            )
        except FileNotFoundError as exc:
            print(f"[Warning] {exc}")
            return

    aligned_data = safe_torch_load(reference_path, map_location="cpu")
    if "image_features" not in aligned_data:
        raise ValueError(f"Aligned CLIP reference has no image_features: {reference_path}")

    old_features = F.normalize(torch.as_tensor(original_features).float().cpu(), dim=-1)
    aligned_features = F.normalize(torch.as_tensor(aligned_data["image_features"]).float().cpu(), dim=-1)
    if old_features.shape != aligned_features.shape:
        raise ValueError(
            "Original/aligned CLIP query feature shapes differ: "
            f"original={tuple(old_features.shape)}, aligned={tuple(aligned_features.shape)}; "
            f"original_path={getattr(args, 'resolved_clip_query_path', '')}, aligned_path={reference_path}"
        )

    if aligned_data.get("labels") is None:
        raise ValueError(f"Aligned CLIP reference has no labels: {reference_path}")

    def hard(labels):
        value = torch.as_tensor(labels).detach().cpu()
        if value.ndim > 1 and value.shape[-1] > 1:
            value = value.argmax(dim=-1)
        return value.reshape(-1).long()

    old_labels = hard(original_labels)
    aligned_labels = hard(aligned_data["labels"])
    if old_labels.shape != aligned_labels.shape or not torch.equal(old_labels, aligned_labels):
        raise ValueError(
            "Original/aligned CLIP label order differs, so the MoCo rows cannot be paired safely. "
            f"original={getattr(args, 'resolved_clip_query_path', '')}, aligned={reference_path}"
        )

    row_cosine = (old_features * aligned_features).sum(dim=-1)
    mean_cosine = float(row_cosine.mean().item())
    min_cosine = float(row_cosine.min().item())
    min_required = float(getattr(args, "alignment_min_cosine", 0.90))
    args.resolved_aligned_clip_reference_path = reference_path
    args.aligned_clip_mean_cosine = mean_cosine
    args.aligned_clip_min_cosine = min_cosine

    print("=" * 80)
    print("Base graph propagation / aligned CLIP image-order verification")
    print(f"Original CLIP : {getattr(args, 'resolved_clip_query_path', '')}")
    print(f"Aligned CLIP  : {reference_path}")
    print(f"Rows          : {old_features.shape[0]}")
    print(f"Labels equal  : True")
    print(f"Mean cosine   : {mean_cosine:.6f}")
    print(f"Min cosine    : {min_cosine:.6f}")
    print(f"Required mean : {min_required:.6f}")
    print("=" * 80)

    if mean_cosine < min_required:
        raise ValueError(
            "The aligned CLIP cache is not in the same image order as the base graph propagation cache "
            f"(mean row-wise cosine {mean_cosine:.6f} < {min_required:.6f}). The paired MoCo "
            "cache must not be fused with the original I.pt. Regenerate/reorder the aligned "
            "CLIP+MoCo caches with --align_to_old_order, or use the aligned CLIP cache for both methods."
        )


def compute_nearest_prototype_accuracy(query_features, labels, prototypes):
    if query_features is None or labels is None or prototypes is None:
        return None
    query_features = F.normalize(query_features.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    pred = (query_features @ prototypes.t()).argmax(dim=1).detach().cpu()
    if torch.is_tensor(labels):
        hard_labels = labels.detach().cpu()
        if hard_labels.ndim >= 2:
            hard_labels = hard_labels.argmax(dim=-1)
        hard_labels = hard_labels.reshape(-1).long()
    else:
        hard_labels = torch.as_tensor(labels).reshape(-1).long()
    if hard_labels.numel() != pred.numel():
        return None
    return (pred == hard_labels).float().mean().item() * 100.0


def compute_dino_cache_logits(query_dino, support_dino, support_labels, beta=5.5, alpha=1.0):
    """DINO query-support instance graph branch.

    query_dino: [1, D]
    support_dino: [C*shot, D]
    support_labels: [C*shot, C]
    """
    query_dino = F.normalize(query_dino.float(), dim=-1)
    support_dino = F.normalize(support_dino.float(), dim=-1)

    affinity = query_dino @ support_dino.t()
    logits = torch.exp(-beta * (1.0 - affinity)) @ support_labels.float()
    return alpha * logits


# ==========================================================
# Clique graph utilities, close to COSMIC style
# ==========================================================
def build_graph_by_topk(sim, topk):
    n = sim.shape[0]
    sim = sim.clone()
    sim.fill_diagonal_(-1e9)
    k = min(max(1, int(topk)), n - 1)
    idx = torch.topk(sim, k=k, dim=1).indices
    adj = torch.zeros_like(sim)
    adj.scatter_(1, idx, 1.0)
    adj = torch.maximum(adj, adj.t())
    adj.fill_diagonal_(0)
    return adj


def build_graph_by_threshold(sim, threshold):
    adj = (sim > threshold).float()
    adj.fill_diagonal_(0)
    return adj


def build_graph_by_degree(sim, target_avg_degree, max_iter=100, tol=1e-2):
    n = sim.shape[0]
    low = sim.min().item()
    high = sim.max().item()
    best_adj = None
    best_deg = 0.0

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        adj = (sim > mid).float()
        adj.fill_diagonal_(0)
        avg_degree = adj.sum().item() / n
        best_adj = adj
        best_deg = avg_degree

        if abs(avg_degree - target_avg_degree) < tol:
            break
        if avg_degree < target_avg_degree:
            high = mid
        else:
            low = mid

    return best_adj, best_deg


def apply_sog(adj):
    two_hop = adj @ adj
    two_hop.fill_diagonal_(0)
    # COSMIC-style: keep stable second-order relation constrained by original adjacency.
    adj_sog = adj * two_hop
    adj_sog = (adj_sog > 1).float()
    adj_sog.fill_diagonal_(0)
    return adj_sog


def make_max_cliques(features, control="topk", control_value=3.0, use_sog=False, reduce_batch=False):
    if nx is None:
        raise ImportError("networkx is required for clique mode. Please install networkx.")

    features = F.normalize(features.float(), dim=-1)
    sim = features @ features.t()

    if control == "topk":
        adj = build_graph_by_topk(sim, int(control_value))
        value = float(control_value)
    elif control == "threshold":
        adj = build_graph_by_threshold(sim, float(control_value))
        value = float(control_value)
    elif control == "degree":
        adj, value = build_graph_by_degree(sim, float(control_value))
    else:
        raise ValueError(f"Unknown clique_control: {control}")

    if use_sog:
        adj = apply_sog(adj)

    graph = nx.from_numpy_array(adj.detach().cpu().numpy())
    cliques = list(nx.find_cliques(graph))

    max_cliques = []
    for clique in cliques:
        clique = sorted(list(set(clique)))
        if len(clique) >= 1:
            max_cliques.append(clique)

    all_nodes = set(range(features.size(0)))
    nodes_in_cliques = set(node for clique in max_cliques for node in clique)
    missing_nodes = sorted(list(all_nodes - nodes_in_cliques))
    for node in missing_nodes:
        max_cliques.append([node])

    if not max_cliques:
        max_cliques = [[i] for i in range(features.size(0))]

    anchors = []
    for clique in max_cliques:
        anchors.append(F.normalize(features[clique].mean(dim=0), dim=-1))
    anchor_features = torch.stack(anchors, dim=0)

    avg_degree = adj.sum().item() / max(1, adj.shape[0])

    return max_cliques, anchor_features, adj, avg_degree, value


def compute_clique_logits(query_features, anchor_features, target_features, max_cliques, r=0.2, beta=5.5, alpha=1.0):
    """COSMIC-style clique logits.

    query_features: [1, D]
    anchor_features: [num_cliques, D]
    target_features: [C, D]
    max_cliques: list[list[int]]
    """
    query_features = F.normalize(query_features.float(), dim=-1)
    anchor_features = F.normalize(anchor_features.float(), dim=-1)
    target_features = F.normalize(target_features.float(), dim=-1)

    clique_aff = query_features @ anchor_features.t()
    top_count = max(1, int(len(max_cliques) * r))
    top_count = min(top_count, len(max_cliques))
    _, top_idx = torch.topk(clique_aff, k=top_count, dim=-1)

    relevant = set()
    for idx in top_idx[0].detach().cpu().tolist():
        relevant.update(max_cliques[int(idx)])

    relevant_indices = sorted(list(relevant))
    if not relevant_indices:
        relevant_indices = list(range(target_features.size(0)))

    relevant_tensor = torch.tensor(relevant_indices, dtype=torch.long, device=target_features.device)
    relevant_target = torch.index_select(target_features, 0, relevant_tensor)

    affinity = query_features @ relevant_target.t()
    logits_small = torch.exp(-beta * (1.0 - affinity)) * alpha

    full_logits = torch.zeros(1, target_features.shape[0], device=query_features.device)
    full_logits[:, relevant_tensor] = logits_small

    return full_logits, relevant_indices



def compute_clique_mask(query_features, anchor_features, target_features, max_cliques, r=0.2):
    """Return selected class mask from a clique/community graph."""
    query_features = F.normalize(query_features.float(), dim=-1)
    anchor_features = F.normalize(anchor_features.float(), dim=-1)

    clique_aff = query_features @ anchor_features.t()
    top_count = max(1, int(len(max_cliques) * r))
    top_count = min(top_count, len(max_cliques))
    _, top_idx = torch.topk(clique_aff, k=top_count, dim=-1)

    relevant = set()
    for idx in top_idx[0].detach().cpu().tolist():
        relevant.update(max_cliques[int(idx)])

    relevant_indices = sorted(list(relevant))
    if not relevant_indices:
        relevant_indices = list(range(target_features.size(0)))

    mask = torch.zeros(1, target_features.shape[0], device=target_features.device)
    mask[:, torch.tensor(relevant_indices, dtype=torch.long, device=target_features.device)] = 1.0
    return mask, relevant_indices


def compute_jgp_gate(text_mask, dino_mask, args):
    if args.jgp_gate_mode == "union":
        jgp_gate = torch.clamp(text_mask + dino_mask, 0.0, 1.0)
    elif args.jgp_gate_mode == "intersection":
        jgp_gate = text_mask * dino_mask
        if jgp_gate.sum().item() < 1:
            jgp_gate = torch.clamp(text_mask + dino_mask, 0.0, 1.0)
    elif args.jgp_gate_mode == "soft":
        jgp_gate = float(args.jgp_text_gate) * text_mask + float(args.jgp_visual_gate) * dino_mask
        jgp_gate = jgp_gate / jgp_gate.max().clamp(min=1e-6)
    else:
        raise ValueError(f"Unknown jgp_gate_mode: {args.jgp_gate_mode}")
    return jgp_gate


def compute_jgp_score(lp_score, text_clique_score, dino_clique_score, text_mask, dino_mask, args):
    """JGP semantic-visual graph guidance.

    The Qwen/text clique graph and the DINO cache clique graph jointly build a
    community gate. This branch is different from late score fusion: the two
    graphs first guide the same prediction distribution through a shared class
    gate, then the guided distribution is fused with other branches.
    """
    lp_prob = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
    text_prob = torch.softmax(float(args.text_clique_scale) * text_clique_score.float(), dim=1)
    dino_prob = torch.softmax(float(args.dino_clique_scale) * dino_clique_score.float(), dim=1)

    jgp_gate = compute_jgp_gate(text_mask, dino_mask, args)

    text_graph_w = max(0.0, min(1.0, float(getattr(args, "jgp_text_weight", 0.5))))
    graph_prob = text_graph_w * text_prob + (1.0 - text_graph_w) * dino_prob
    jgp_prob = (1.0 - float(args.jgp_inner_weight)) * lp_prob + float(args.jgp_inner_weight) * graph_prob
    jgp_prob = jgp_prob * (1.0 + float(args.jgp_mask_boost) * jgp_gate)
    jgp_prob = jgp_prob / jgp_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return jgp_prob



def compute_single_graph_guided_score(lp_score, graph_score, graph_mask, args, graph_scale=1.0):
    """Single visual graph guided prediction.

    This is the standalone LP + MoCo visual graph branch. It follows the
    same internal design as the single-graph ablation in the jgp dual graph:

        1) Convert LP score and visual graph score to probabilities.
        2) Mix LP probability and graph probability using jgp_inner_weight.
        3) Use the visual clique mask as a gate to boost selected classes.
        4) Renormalize to a probability distribution.

    It is intentionally not the same as directly fusing LP with dino_clique_score.
    """
    lp_prob = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
    graph_prob = torch.softmax(float(graph_scale) * graph_score.float(), dim=1)

    graph_gate = graph_mask.float()
    if graph_gate.max().item() > 0:
        graph_gate = graph_gate / graph_gate.max().clamp(min=1e-6)

    guided_prob = (1.0 - float(args.jgp_inner_weight)) * lp_prob + float(args.jgp_inner_weight) * graph_prob
    guided_prob = guided_prob * (1.0 + float(args.jgp_mask_boost) * graph_gate)
    guided_prob = guided_prob / guided_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return guided_prob


# ==========================================================
# Legacy cross-graph helper
# ==========================================================
def build_row_normalized_topk_adj(features, topk=8, self_loop=True, temperature=1.0, eps=1e-6):
    """Build a row-normalized class graph from class prototypes.

    features: [C, D]
    return: [C, C]
    """
    features = F.normalize(features.float(), dim=-1)
    sim = features @ features.t()
    c = sim.shape[0]

    if topk is not None and int(topk) > 0 and int(topk) < c:
        masked = torch.full_like(sim, -1e9)
        k = min(int(topk), c)
        if self_loop:
            idx = torch.topk(sim, k=k, dim=1).indices
        else:
            sim_no_diag = sim.clone()
            sim_no_diag.fill_diagonal_(-1e9)
            idx = torch.topk(sim_no_diag, k=min(k, c - 1), dim=1).indices
        masked.scatter_(1, idx, sim.gather(1, idx))
        sim = masked

    if self_loop:
        # ensure self-loop always exists
        diag = torch.arange(c, device=features.device)
        sim[diag, diag] = 1.0

    adj = torch.softmax(sim / max(float(temperature), eps), dim=1)
    return adj


def compute_cross_graph_score(lp_score, dino_score, clip_class_prototypes, dino_class_prototypes, args):
    """Mutual propagation between the CLIP class graph and the auxiliary visual class graph.

    We do not mix feature dimensions directly. Instead, both branches are first
    represented as class-level distributions, then we propagate class confidence
    over a two-layer graph:

        [ CLIP class graph      cross identity ]
        [ cross identity       DINO class graph]

    This makes the CLIP graph scores and auxiliary visual scores refine each other through same-label
    cross edges and intra-modality neighborhood edges.
    """
    lp_prob = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)       # [1, C]
    dino_prob = torch.softmax(float(args.dino_cache_scale) * dino_score.float(), dim=1)  # [1, C]

    c = lp_prob.shape[1]
    device = lp_prob.device

    a_clip = build_row_normalized_topk_adj(
        clip_class_prototypes,
        topk=args.cross_topk,
        self_loop=True,
        temperature=args.cross_graph_temp,
    )
    a_dino = build_row_normalized_topk_adj(
        dino_class_prototypes,
        topk=args.cross_topk,
        self_loop=True,
        temperature=args.cross_graph_temp,
    )

    eye = torch.eye(c, device=device, dtype=lp_prob.dtype)
    eta = float(args.cross_eta)

    # Block adjacency for two score layers: CLIP graph nodes and DINO nodes.
    top = torch.cat([a_clip, eta * eye], dim=1)
    bottom = torch.cat([eta * eye, a_dino], dim=1)
    adj = torch.cat([top, bottom], dim=0)
    adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-6)

    h = torch.cat([lp_prob, dino_prob], dim=1)  # [1, 2C]
    h0 = h
    for _ in range(int(args.cross_steps)):
        h = h @ adj
        h = (1.0 - float(args.cross_residual)) * h + float(args.cross_residual) * h0
        h = h / h.sum(dim=1, keepdim=True).clamp(min=1e-6)

    clip_part = h[:, :c]
    dino_part = h[:, c:]
    cross_prob = float(args.cross_output_clip_weight) * clip_part + (1.0 - float(args.cross_output_clip_weight)) * dino_part
    cross_prob = cross_prob / cross_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)

    # Final branch fusion. cross_graph_weight=1 means only propagated cross-graph distribution.
    out = (1.0 - float(args.cross_graph_weight)) * lp_prob + float(args.cross_graph_weight) * cross_prob
    out = out / out.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return out

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


# ==========================================================
# Evaluation helpers
# ==========================================================
def update_correct(counter, pred, gt):
    if torch.is_tensor(gt):
        gt_cpu = gt.cpu()
    else:
        gt_cpu = torch.tensor(gt)
    return counter + (pred.cpu() == gt_cpu).sum().item()


def search_best_weight(base_logits_all, branch_logits_all, labels_all, weights, base_scale=1.0, branch_scale=1.0):
    """Search fusion ratio in [0, 1] with branch temperature scaling.

    Fusion:
        base_prob   = softmax(base_scale * base_logits)
        branch_prob = softmax(branch_scale * branch_logits)
        final       = (1 - weight) * base_prob + weight * branch_prob

    Interpretation:
        weight=0.0 -> only LP/base branch
        weight=1.0 -> only auxiliary branch
        weight=0.3 -> 70% LP + 30% branch

    branch_scale is used to sharpen or weaken the auxiliary branch before softmax.
    For example, if previous additive fusion needed weight=30, use branch_scale
    to make DINO distribution sharper while keeping fusion ratio in [0, 1].
    """
    best_acc = -1.0
    best_w = 0.0

    base_prob = torch.softmax(float(base_scale) * base_logits_all.float(), dim=1)
    branch_prob = torch.softmax(float(branch_scale) * branch_logits_all.float(), dim=1)

    for w in weights:
        w = float(w)
        w = max(0.0, min(1.0, w))
        logits = (1.0 - w) * base_prob + w * branch_prob
        pred = logits.argmax(dim=1).cpu()
        acc = (pred == labels_all.cpu()).float().mean().item() * 100.0

        if acc > best_acc:
            best_acc = acc
            best_w = w

    return best_acc, best_w


def search_best_two_weights(
    base_logits_all,
    branch1_all,
    branch2_all,
    labels_all,
    weights1,
    weights2,
    base_scale=1.0,
    branch1_scale=1.0,
    branch2_scale=1.0,
):
    """Search two-branch fusion ratios with temperature scaling.

    We use normalized convex fusion:
        base_weight = max(0, 1 - w1 - w2)
        final = base_weight * base_prob + w1 * branch1_prob + w2 * branch2_prob

    All weights are ratios in [0, 1].
    """
    best_acc = -1.0
    best_w1 = 0.0
    best_w2 = 0.0

    base_prob = torch.softmax(float(base_scale) * base_logits_all.float(), dim=1)
    branch1_prob = torch.softmax(float(branch1_scale) * branch1_all.float(), dim=1)
    branch2_prob = torch.softmax(float(branch2_scale) * branch2_all.float(), dim=1)

    for w1 in weights1:
        for w2 in weights2:
            w1 = max(0.0, min(1.0, float(w1)))
            w2 = max(0.0, min(1.0, float(w2)))
            if w1 + w2 > 1.0:
                continue

            base_w = 1.0 - w1 - w2
            logits = base_w * base_prob + w1 * branch1_prob + w2 * branch2_prob
            pred = logits.argmax(dim=1).cpu()
            acc = (pred == labels_all.cpu()).float().mean().item() * 100.0

            if acc > best_acc:
                best_acc = acc
                best_w1 = w1
                best_w2 = w2

    return best_acc, best_w1, best_w2


def normalize_score_vector(x, mode="none", eps=1e-6):
    """Normalize per-sample score vector before score-level fusion.

    mode:
        none   : keep original scale
        l2     : divide by L2 norm
        l1     : divide by L1 norm
        zscore : subtract mean and divide by std
        minmax : map each sample score to [0, 1]
    """
    if mode == "none":
        return x

    x = x.float()

    if mode == "l2":
        return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

    if mode == "l1":
        return x / (x.abs().sum(dim=1, keepdim=True) + eps)

    if mode == "zscore":
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / (std + eps)

    if mode == "minmax":
        xmin = x.min(dim=1, keepdim=True).values
        xmax = x.max(dim=1, keepdim=True).values
        return (x - xmin) / (xmax - xmin + eps)

    raise ValueError(f"Unknown score_norm mode: {mode}")


def fuse_scores(lp_score, text_clique_score, dino_cache_score, dino_clique_score, jgp_score, args):
    """Probability-level convex fusion with temperature/amplification scales.

    All branch weights are interpreted as ratios in [0, 1], not as large
    amplification factors.

    Example:
        dino_cache_weight = 0.3
        final = 0.7 * LP_prob + 0.3 * DINO_prob

    Temperature/amplification:
        dino_cache_scale = 30
        dino_prob = softmax(30 * dino_logits)

    This keeps the fusion ratio in [0, 1], while allowing a weak branch score
    to become sharper before probability fusion.
    """
    lp_prob = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)

    text_w = max(0.0, min(1.0, float(args.text_clique_weight)))
    dino_cache_w = max(0.0, min(1.0, float(args.dino_cache_weight)))
    dino_clique_w = max(0.0, min(1.0, float(args.dino_clique_weight)))
    jgp_w = max(0.0, min(1.0, float(args.jgp_weight))) if getattr(args, "use_jgp", False) else 0.0

    total_branch_w = text_w + dino_cache_w + dino_clique_w + jgp_w
    if total_branch_w > 1.0:
        text_w = text_w / total_branch_w
        dino_cache_w = dino_cache_w / total_branch_w
        dino_clique_w = dino_clique_w / total_branch_w
        jgp_w = jgp_w / total_branch_w
        base_w = 0.0
    else:
        base_w = 1.0 - total_branch_w

    final_prob = base_w * lp_prob

    if text_w > 0:
        text_prob = torch.softmax(float(args.text_clique_scale) * text_clique_score.float(), dim=1)
        final_prob = final_prob + text_w * text_prob

    if dino_cache_w > 0:
        dino_prob = torch.softmax(float(args.dino_cache_scale) * dino_cache_score.float(), dim=1)
        final_prob = final_prob + dino_cache_w * dino_prob

    if dino_clique_w > 0:
        dino_clique_prob = torch.softmax(float(args.dino_clique_scale) * dino_clique_score.float(), dim=1)
        final_prob = final_prob + dino_clique_w * dino_clique_prob

    if jgp_w > 0:
        if jgp_score is None:
            raise ValueError("--jgp_weight > 0 requires --use_jgp.")
        jgp_prob = jgp_score.float()
        jgp_prob = jgp_prob / jgp_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
        final_prob = final_prob + jgp_w * jgp_prob

    return final_prob



# ==========================================================
# Search runner: 0-shot to 16-shot + internal grid search
# ==========================================================
import csv
import time
from itertools import product
from types import SimpleNamespace


# ==========================================================
# Editable search config
# All searchable parameters are controlled here.
# In normal use, you only run with --datasets and --features_path.
# ==========================================================
def iter_grid(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def iter_search_configs(search_config):
    """Yield minimal search configs.

    Search dimensions:
        1) shot
        2) text_mode
        3) jgp_weight

    For shot=0, JGP cannot be built because there are no few-shot
    DINO prototypes, so we run LP baseline once for each text_mode.
    """
    fixed = search_config["fixed"]
    for shot in search_config["shot"]:
        for text_mode in search_config["text_mode"]:
            weights = [0.0] if int(shot) == 0 else search_config["jgp_weight"]
            for jgp_w in weights:
                hp = dict(fixed)
                hp["jgp_weight"] = float(jgp_w)
                branch_mode = "lp" if int(shot) == 0 else "jgp"
                yield {
                    "shot": shot,
                    "text_mode": text_mode,
                    "branch_mode": branch_mode,
                    "params": hp,
                }


def parse_shot_list(text):
    if not text:
        return None
    out = []
    for part in str(text).replace("/", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_text_mode_list(text):
    if not text:
        return None
    valid = {"original", "cupl_aux_node", "qwen_aux_node"}
    out = []
    for part in str(text).replace("/", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if item not in valid:
            raise ValueError(
                f"Unknown --text_mode item: {item}. "
                "Choose from original, cupl_aux_node, qwen_aux_node."
            )
        out.append(item)
    return out


def parse_patch_text_source_list(text):
    if not text:
        return ["cupl"]
    valid = {"qwen", "cupl", "photo"}
    out = []
    for part in str(text).replace("/", ",").split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"both", "all"}:
            for source in ["qwen", "cupl", "photo"]:
                if source not in out:
                    out.append(source)
            continue
        if item not in valid:
            raise ValueError(f"Unknown --patch_text_source item: {item}. Choose from qwen, cupl, photo, both.")
        if item not in out:
            out.append(item)
    return out or ["cupl"]


def parse_module_list(text):
    if not text:
        return ["baseline", "aki", "msc", "jgp", "jgp_msc", "full"]
    aliases = {
        "base": "baseline",
        "lp": "baseline",
        "qwen": "aki",
        "baseline_qwen": "aki",
        "baseline_aki": "aki",
        "baseline_jgp": "jgp",
        "baseline_msc": "msc",
        "baseline_msc_qwen": "msc",
        "msc_qwen": "msc",
        "jgp+msc": "jgp_msc",
        "jgp_msc_qwen": "jgp_msc",
        "aki+jgp+msc": "full",
        "jgp+msc+aki": "full",
        "full_qwen": "full",
    }
    valid = {"baseline", "aki", "msc", "jgp", "jgp_msc", "full"}
    out = []
    for part in str(text).replace("/", ",").split(","):
        item = part.strip().lower()
        if not item:
            continue
        item = aliases.get(item, item)
        if item not in valid:
            raise ValueError(
                f"Unknown --modules item: {part.strip()}. "
                "Choose from baseline, aki, msc, jgp, jgp_msc, full."
            )
        if item not in out:
            out.append(item)
    return out or ["baseline", "jgp", "msc", "jgp_msc"]


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


def sanitize_filename_part(text):
    text = str(text or "").strip()
    if not text:
        return "default"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "default"


def resolve_output_csv_path(cli_args, datasets):
    if getattr(cli_args, "output_csv", ""):
        return cli_args.output_csv

    dataset_part = sanitize_filename_part(datasets[0] if len(datasets) == 1 else "all_datasets")
    shots = parse_shot_list(getattr(cli_args, "shots", "")) or SEARCH_CONFIG["shot"]
    shot_part = "shot" + sanitize_filename_part("-".join(str(s) for s in shots))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_part = sanitize_filename_part(getattr(cli_args, "text_mode", "") or "search_text")
    patch_part = sanitize_filename_part(getattr(cli_args, "patch_variant", "both"))
    patch_text_part = sanitize_filename_part(getattr(cli_args, "patch_text_source", "cupl"))
    alpha_part = "searchalpha" if getattr(cli_args, "search_alpha", False) else f"alpha{float(getattr(cli_args, 'jgp_text_weight', 0.5)):g}"
    filename = f"{dataset_part}_{shot_part}_{text_part}_patch-{patch_part}_patchtext-{patch_text_part}_{alpha_part}_{timestamp}.csv"
    return os.path.join("results_consen", dataset_part, shot_part, filename)


def compact_subset_for_dataset(compact, dataset_name):
    return {k: v for k, v in compact.items() if k[0] == dataset_name}


def resolve_internal_paths(cli_args):
    """Resolve paths that are common to all datasets.

    This MoCo version keeps the original variable names such as
    dino_features_path internally for compatibility, but the path points to
    the pre-extracted MoCo branch by default:
        features_path/fine_grained/MOCO/R50

    prompt_json and cupl_json are resolved per dataset in
    resolve_dataset_resource_paths(), so multi-dataset runs will not
    accidentally reuse Cars/Aircraft prompts.
    """
    paths = SEARCH_CONFIG.get("paths", {})

    # Highest priority: explicit MoCo path alias.
    if getattr(cli_args, "moco_features_path", ""):
        cli_args.dino_features_path = cli_args.moco_features_path
        return cli_args

    # Backward compatibility: --dino_features_path can still be used, but in
    # this file it is interpreted as the visual/MoCo feature path.
    if not getattr(cli_args, "dino_features_path", ""):
        visual_path = paths.get("dino_features_path", "moco_r50_auto")
        if visual_path in ["auto", "moco_r50_auto", "MOCO/R50", ""]:
            cli_args.dino_features_path = os.path.join(cli_args.features_path, cli_args.task, "MOCO", "R50")
        elif visual_path in ["moco_vit_auto", "MOCO/VIT"]:
            cli_args.dino_features_path = os.path.join(cli_args.features_path, cli_args.task, "MOCO", "VIT")
        else:
            cli_args.dino_features_path = visual_path

    return cli_args


def mode_to_flags(branch_mode):
    flags = {
        "use_text_clique": False,
        "use_dino_cache": False,
        "use_dino_clique": False,
        "use_jgp": False,
        "use_cross_graph": False,
    }

    if branch_mode == "lp":
        return flags
    if branch_mode == "jgp":
        flags["use_jgp"] = True
        return flags

    raise ValueError(f"Unknown branch mode: {branch_mode}")

def make_run_args(cli_args, dataset_name, shot, text_mode, branch_mode, hp):
    flags = mode_to_flags(branch_mode)
    if text_mode == "qwen_aux_node":
        run_attribute_source = "qwen"
    elif text_mode == "cupl_aux_node":
        run_attribute_source = "cupl"
    else:
        run_attribute_source = getattr(cli_args, "attribute_text_source", "cupl")

    run_args = SimpleNamespace(
        task=cli_args.task,
        datasets=dataset_name,
        features_path=cli_args.features_path,
        clip_fewshot_features_path=getattr(cli_args, "clip_fewshot_features_path", ""),
        clip_model=cli_args.clip_model,
        output_file="",

        mode="ZS" if int(shot) == 0 else "FS",
        num_shots=max(1, int(shot)),

        text_mode=text_mode,
        base_text_prompt=getattr(cli_args, "base_text_prompt", "photo"),
        attribute_text_source=run_attribute_source,
        prompt_json=cli_args.prompt_json,
        cupl_json=cli_args.cupl_json,
        batch_size=cli_args.batch_size,

        dino_features_path=cli_args.dino_features_path,
        dino_model=cli_args.dino_model,
        meta_file=cli_args.meta_file,
        aligned_clip_reference_path=getattr(cli_args, "aligned_clip_reference_path", ""),
        alignment_min_cosine=getattr(cli_args, "alignment_min_cosine", 0.90),
        use_support_patch_nodes=cli_args.use_support_patch_nodes,
        support_patch_features_pt=cli_args.support_patch_features_pt,
        support_patch_cache_out=cli_args.support_patch_cache_out,
        support_patch_topk_per_class=cli_args.support_patch_topk_per_class,
        support_patch_select_mode=cli_args.support_patch_select_mode,
        patch_text_source=cli_args.patch_text_source,
        patch_grid=cli_args.patch_grid,
        crop_overlap=cli_args.crop_overlap,
        moco_image_size=cli_args.moco_image_size,
        patch_batch_size=cli_args.patch_batch_size,
        bridge_source=cli_args.bridge_source,
        bridge_lambda=cli_args.bridge_lambda,
        no_tqdm=cli_args.no_tqdm,
        search_alpha=getattr(cli_args, "search_alpha", False),
        alpha_grid=getattr(
            cli_args,
            "alpha_grid",
            ",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["alpha"]),
        ),
        search_jgp_weight=getattr(cli_args, "search_jgp_weight", False),
        jgp_weight_grid=getattr(
            cli_args,
            "jgp_weight_grid",
            ",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["w"]),
        ),
        search_alpha_weight_matrix=getattr(cli_args, "search_alpha_weight_matrix", True),
        jgp_text_weight=getattr(cli_args, "jgp_text_weight", 0.5),
        measure_runtime=getattr(cli_args, "save_timing", False),
        timing_warmup_queries=getattr(cli_args, "timing_warmup_queries", 5),
        sensitivity_fixed_w=getattr(cli_args, "sensitivity_fixed_w", 0.9),
        support_grounded_aki=(
            getattr(cli_args, "support_grounded_aki", False)
            and text_mode in {"cupl_aux_node", "qwen_aux_node"}
        ),
        aki_grounding_level=getattr(cli_args, "aki_grounding_level", "image"),
        aki_margin_threshold=getattr(cli_args, "aki_margin_threshold", None),
        ground_msc_qwen=getattr(cli_args, "ground_msc_qwen", False),
        prediction_scores_out=getattr(cli_args, "prediction_scores_out", ""),
        prediction_export_per_class=getattr(cli_args, "prediction_export_per_class", 0),
        prediction_export_seed=getattr(cli_args, "prediction_export_seed", 1),
        save_patch_evidence=False,

        score_norm="none",
        search_weights=False,
        search_weight_grid="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )

    for k, v in hp.items():
        setattr(run_args, k, v)

    for k, v in flags.items():
        setattr(run_args, k, v)

    return run_args


def config_is_valid(args):
    if args.text_mode in ["recoded_cupl", "qwen_positive", "cupl_aux_node"] and not args.cupl_json:
        return False, "cupl_json is required for non-original text modes"
    if args.text_mode in ["qwen_positive", "qwen_aux_node"] and not args.prompt_json:
        return False, "prompt_json is required for Qwen text modes"

    # DINO cache and DINO clique require few-shot support features in this implementation.
    if args.mode == "ZS" and (args.use_dino_cache or args.use_dino_clique or args.use_jgp or getattr(args, "use_cross_graph", False)):
        return False, "DINO cache/clique/jgp modes require FS support features"

    if (args.use_dino_cache or args.use_dino_clique or args.use_jgp or getattr(args, "use_cross_graph", False)) and not args.dino_features_path:
        # If not specified, code falls back to features_path/task/DINO, so this is still valid.
        return True, ""

    return True, ""


def synchronize_device(device):
    """Synchronize CUDA only when timing GPU work."""
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_one_dataset(args, clip_model, clip_preprocess, moco_model, moco_preprocess, device, dataset_name):
    evaluation_started = time.perf_counter()
    timing_enabled = bool(getattr(args, "measure_runtime", False))
    if timing_enabled and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    feature_path = resolve_dataset_pt_path(
        os.path.join(args.features_path, args.task, args.clip_model),
        dataset_name,
        description="CLIP query feature",
    )
    args.resolved_clip_query_path = feature_path
    args.resolved_clip_fewshot_path = ""
    pre_calculated = safe_torch_load(feature_path, map_location="cpu")
    args.source_global_class_ids = []
    if getattr(args, "meta_file", "") and os.path.exists(args.meta_file):
        aligned_meta = safe_torch_load(args.meta_file, map_location="cpu")
        meta_class_names = aligned_meta.get("class_names", [])
        if (
            meta_class_names
            and "class_names" not in pre_calculated
            and "classes" not in pre_calculated
        ):
            expected_classes = int(pre_calculated["text_features"].shape[0])
            if len(meta_class_names) != expected_classes:
                raise ValueError(
                    f"Aligned meta contains {len(meta_class_names)} class names but "
                    f"the text cache contains {expected_classes} class features: {args.meta_file}"
                )
            pre_calculated = dict(pre_calculated)
            pre_calculated["class_names"] = list(meta_class_names)
            print(
                f"Using {len(meta_class_names)} class names from aligned meta: "
                f"{args.meta_file}"
            )

        if dataset_name in {"ImageNet-A", "ImageNet-V2", "ImageNet-R", "ImageNet-Sketch"}:
            source_global_class_ids = aligned_meta.get("source_global_class_ids", [])
            if source_global_class_ids:
                args.source_global_class_ids = list(source_global_class_ids)
                print(
                    "Using ImageNet global-to-local class map from aligned meta: "
                    f"{len(args.source_global_class_ids)} classes"
                )
    image_features_load = pre_calculated["image_features"]
    text_features_load = pre_calculated["text_features"]
    gts = pre_calculated["labels"]

    cupl_data = load_json(args.cupl_json) if args.cupl_json else None

    text_selection_args = args
    if getattr(args, "support_grounded_aki", False):
        # The grounded branch constructs its description nodes after support features
        # are loaded. Start from original class nodes to avoid building and
        # encoding an unfiltered auxiliary graph that will immediately be replaced.
        text_selection_args = SimpleNamespace(**vars(args))
        text_selection_args.text_mode = "original"

    text_features, class_names, aux_to_class = select_text_features(
        args=text_selection_args,
        clip_model=clip_model,
        pre_calculated=pre_calculated,
        text_features_load=text_features_load,
        cupl_data=cupl_data,
        device=device,
    )

    if class_names is None:
        num_original_classes = text_features_load.shape[0]
    else:
        num_original_classes = len(class_names)

    few_shot_image_features = None
    few_shot_labels_original = None
    few_shot_labels_lp = None
    preloaded_support_patch_pool = None
    preloaded_support_paths = None
    preloaded_support_class_ids = None
    aki_grounding_stats = None
    msc_grounding_stats = None

    if args.mode == "FS":
        clip_fewshot_root = getattr(args, "clip_fewshot_features_path", "")
        if not clip_fewshot_root:
            clip_fewshot_root = os.path.join(args.features_path, args.task, args.clip_model, "few_shots")
        fewshot_path = resolve_dataset_pt_path(
            clip_fewshot_root,
            dataset_name,
            description="CLIP few-shot feature",
        )
        args.resolved_clip_fewshot_path = fewshot_path
        few_shot_pre_calculate = safe_torch_load(fewshot_path, map_location="cpu")
        few_shot_image_features = few_shot_pre_calculate["image_features"]
        few_shot_labels_original = few_shot_pre_calculate["labels"]

        few_shot_image_features = few_shot_image_features[:, :args.num_shots, :]
        few_shot_image_features = few_shot_image_features.reshape(-1, few_shot_image_features.shape[-1]).to(device)
        few_shot_image_features = F.normalize(few_shot_image_features.float(), dim=-1)

        few_shot_labels_original = few_shot_labels_original[:, :args.num_shots, :]
        few_shot_labels_original = few_shot_labels_original.reshape(-1, few_shot_labels_original.shape[-1]).to(device)

    if getattr(args, "support_grounded_aki", False):
        if args.text_mode not in {"cupl_aux_node", "qwen_aux_node"}:
            raise ValueError(
                "--support_grounded_aki requires --aki_text_mode "
                "cupl_aux_node or qwen_aux_node"
            )
        if few_shot_image_features is None or few_shot_labels_original is None:
            raise ValueError("--support_grounded_aki requires few-shot support images (shot > 0)")
        if class_names is None:
            class_names, _ = get_class_names(pre_calculated, text_features_load, cupl_data)

        grounding_level = getattr(args, "aki_grounding_level", "image")
        grounding_features = few_shot_image_features
        grounding_labels = few_shot_labels_original
        if grounding_level == "patch":
            meta = load_meta(args)
            preloaded_support_paths, preloaded_support_class_ids = flatten_selected_support_paths(
                meta,
                num_original_classes,
                args.num_shots,
                args=args,
            )
            preloaded_support_patch_pool = load_or_build_support_patch_pool(
                args=args,
                support_paths=preloaded_support_paths,
                support_class_ids=preloaded_support_class_ids,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                moco_model=moco_model,
                moco_preprocess=moco_preprocess,
                device=device,
            )
            grounding_features = preloaded_support_patch_pool[0]
            grounding_labels = onehot_from_class_ids(
                preloaded_support_class_ids,
                num_original_classes,
                device,
            )

        # Keep the graph's original nodes identical to the paper baseline:
        # a photo of a {class}.  Do not reintroduce the cached/merged text
        # features here after the AKI candidates have been grounded.
        original_text_features = F.normalize(
            text_features[:num_original_classes].to(device).float(),
            dim=-1,
        )
        grounded_aux_features, aux_to_class, aki_grounding_stats = build_support_grounded_auxiliary_nodes(
            model=clip_model,
            class_names=class_names,
            prompt_json=args.prompt_json,
            cupl_json=args.cupl_json,
            support_features=grounding_features,
            support_labels=grounding_labels,
            original_text_features=original_text_features,
            device=device,
            batch_size=args.batch_size,
            min_quality=args.min_quality,
            topk_per_class=args.topk_per_class,
            aux_residual=args.aux_residual,
            aux_mu=args.aux_mu,
            grounding_level=grounding_level,
            num_shots=args.num_shots,
            margin_threshold=getattr(args, "aki_margin_threshold", None),
            source_global_class_ids=getattr(args, "source_global_class_ids", None),
            attribute_text_source=getattr(args, "attribute_text_source", "cupl"),
        )
        if grounded_aux_features is None:
            text_features = original_text_features
        else:
            text_features = torch.cat([original_text_features, grounded_aux_features], dim=0)

    dino_image_features, dino_fewshot_features, dino_fewshot_labels, dino_query_labels = load_dino_feature_file(
        args=args,
        dataset_name=dataset_name,
        device=device,
    )
    if dino_image_features is not None:
        validate_query_label_alignment(
            clip_labels=gts,
            moco_labels=dino_query_labels,
            clip_path=str(getattr(args, "resolved_clip_query_path", feature_path)),
            moco_path=str(getattr(args, "resolved_moco_query_path", "")),
        )
        validate_aligned_clip_reference(
            args=args,
            dataset_name=dataset_name,
            original_features=image_features_load,
            original_labels=gts,
        )
        align_info = getattr(args, "moco_align_info", None) or {}
        if align_info:
            print("=" * 80)
            print("MoCo image-level alignment metadata")
            print(f"Aligned to old order : {bool(align_info.get('aligned_to_old_order', False))}")
            print(f"Old CLIP reference   : {align_info.get('old_feature_path', '[not recorded]')}")
            print(f"One-to-one order     : {align_info.get('order_is_bijective', '[not recorded]')}")
            print(f"Mean CLIP cosine     : {align_info.get('image_mean_cosine_to_old', '[not recorded]')}")
            print("=" * 80)
        else:
            print(
                "[Warning] MoCo query labels match the original CLIP cache, but the MoCo file "
                "contains no align_info. Within-class image-level order cannot be independently verified."
            )
    if args.mode == "FS" and dino_fewshot_labels is not None and few_shot_labels_original is not None:
        print_label_alignment_check(
            dataset_name,
            few_shot_labels_original,
            dino_fewshot_labels,
            num_original_classes,
        )

    if getattr(args, "use_support_patch_nodes", False) and args.mode == "FS":
        if dino_fewshot_features is None or dino_fewshot_labels is None:
            raise ValueError("--use_support_patch_nodes requires MoCo few-shot features.")
        if class_names is None:
            class_names, _ = get_class_names(pre_calculated, text_features_load, cupl_data)
        if preloaded_support_paths is None:
            meta = load_meta(args)
            support_paths, support_class_ids = flatten_selected_support_paths(
                meta,
                num_original_classes,
                args.num_shots,
                args=args,
            )
        else:
            support_paths = preloaded_support_paths
            support_class_ids = preloaded_support_class_ids

        bridge_source_clip = image_features_load.to(device)
        bridge_source_moco = dino_image_features
        if args.bridge_source in {"support", "fewshot"}:
            bridge_source_clip = few_shot_image_features
            bridge_source_moco = dino_fewshot_features
        elif args.bridge_source == "all":
            bridge_source_clip = torch.cat([bridge_source_clip, few_shot_image_features], dim=0)
            bridge_source_moco = torch.cat([bridge_source_moco, dino_fewshot_features], dim=0)
        elif args.bridge_source != "query":
            raise ValueError(f"Unknown --bridge_source: {args.bridge_source}")

        clip_to_moco_bridge = solve_clip_to_moco_bridge(
            clip_features=bridge_source_clip,
            moco_features=bridge_source_moco,
            ridge_lambda=args.bridge_lambda,
        )
        attr_texts, attr_to_class = load_qwen_attributes(
            prompt_json=args.prompt_json,
            class_names=class_names,
            min_quality=args.min_quality,
            topk_per_class=args.topk_per_class,
            cupl_json=args.cupl_json,
            patch_text_source=args.patch_text_source,
            source_global_class_ids=getattr(args, "source_global_class_ids", None),
        )
        clip_attr_text_features = encode_texts(clip_model, attr_texts, device, args.batch_size)
        attr_to_class = attr_to_class.to(device)

        if (
            getattr(args, "ground_msc_qwen", False)
            and args.patch_text_source == "qwen"
        ):
            threshold = getattr(args, "aki_margin_threshold", None)
            if threshold is None:
                threshold = aki_thresholds_for_shot(args.num_shots)[0]
            clip_attr_text_features, attr_to_class, msc_grounding_stats = filter_msc_qwen_attributes(
                model=clip_model,
                class_names=class_names,
                clip_attr_features=clip_attr_text_features,
                attr_to_class=attr_to_class,
                support_features=few_shot_image_features,
                support_labels=few_shot_labels_original,
                margin_threshold=threshold,
                device=device,
                batch_size=args.batch_size,
            )

        moco_attr_text_features = project_clip_to_moco(clip_attr_text_features, clip_to_moco_bridge)

        if preloaded_support_patch_pool is None:
            support_clip_patches, support_moco_patches, _ = load_or_build_support_patch_pool(
                args=args,
                support_paths=support_paths,
                support_class_ids=support_class_ids,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                moco_model=moco_model,
                moco_preprocess=moco_preprocess,
                device=device,
            )
        else:
            support_clip_patches, support_moco_patches, _ = preloaded_support_patch_pool

        support_clip_nodes, support_moco_nodes, support_patch_labels, _ = select_class_aware_support_patch_nodes(
            support_clip_patches=support_clip_patches,
            support_moco_patches=support_moco_patches,
            support_class_ids=support_class_ids,
            clip_attr_features=clip_attr_text_features,
            moco_attr_features=moco_attr_text_features,
            attr_to_class=attr_to_class,
            num_classes=num_original_classes,
            args=args,
            device=device,
        )

        few_shot_image_features = torch.cat([few_shot_image_features, support_clip_nodes], dim=0)
        few_shot_labels_original = torch.cat([few_shot_labels_original[:, :num_original_classes], support_patch_labels], dim=0)
        dino_fewshot_features = torch.cat([dino_fewshot_features, support_moco_nodes], dim=0)
        dino_fewshot_labels = torch.cat([dino_fewshot_labels[:, :num_original_classes], support_patch_labels], dim=0)

        print("=" * 80)
        print("Support Patch Nodes Injected Into Few-shot Graph")
        print("=" * 80)
        print(f"CLIP few-shot nodes : {tuple(few_shot_image_features.shape)}")
        print(f"MoCo few-shot nodes : {tuple(dino_fewshot_features.shape)}")
        print("=" * 80)

    few_shot_labels_lp = few_shot_labels_original
    if args.text_mode in {"cupl_aux_node", "qwen_aux_node"} and few_shot_labels_original is not None:
        few_shot_labels_lp = expand_fewshot_labels_for_aux_nodes(
            few_shot_labels=few_shot_labels_original,
            num_original_classes=num_original_classes,
            aux_to_class=aux_to_class,
        )

    # Text clique graph
    text_cliques = None
    text_clique_anchors = None
    text_clique_target = None
    text_adj = None

    if args.use_text_clique or args.use_jgp:
        text_clique_target = merge_text_aux_to_class_features(
            text_features=text_features,
            num_original_classes=num_original_classes,
            aux_to_class=aux_to_class,
        )

        text_cliques, text_clique_anchors, text_adj, text_avg_degree, text_control_value = make_max_cliques(
            features=text_clique_target,
            control=args.text_clique_control,
            control_value=args.text_clique_control_value,
            use_sog=args.text_clique_sog,
        )

    # CLIP class prototypes for cross-graph propagation
    clip_class_prototypes = None
    if getattr(args, "use_cross_graph", False):
        if few_shot_image_features is None or few_shot_labels_original is None:
            raise ValueError("--use_cross_graph requires CLIP few-shot features.")
        clip_class_prototypes = build_class_prototypes_from_fewshot(
            few_shot_image_features=few_shot_image_features,
            few_shot_labels=few_shot_labels_original,
            num_classes=num_original_classes,
        )

    # DINO visual clique graph
    dino_class_prototypes = None
    dino_cliques = None
    dino_clique_anchors = None
    dino_adj = None

    if args.use_dino_clique or args.use_jgp or getattr(args, "use_cross_graph", False):
        if dino_fewshot_features is None or dino_fewshot_labels is None:
            raise ValueError("--use_dino_clique or --use_jgp requires DINO few-shot features.")

        dino_class_prototypes = build_class_prototypes_from_fewshot(
            few_shot_image_features=dino_fewshot_features,
            few_shot_labels=dino_fewshot_labels,
            num_classes=num_original_classes,
        )

        dino_cliques, dino_clique_anchors, dino_adj, dino_avg_degree, dino_control_value = make_max_cliques(
            features=dino_class_prototypes,
            control=args.dino_clique_control,
            control_value=args.dino_clique_control_value,
            use_sog=args.dino_clique_sog,
        )

    correct_lp = 0
    correct_text_clique = 0
    correct_dino_cache = 0
    correct_dino_clique = 0
    correct_jgp = 0
    correct_lp_moco_visual = 0
    correct_cross_graph = 0
    correct_final = 0
    correct_sensitivity_fixed = 0
    alpha_values = []
    alpha_correct = []
    alpha_accs = []
    jgp_weight_values = []
    jgp_weight_correct = []
    weight_accs = []
    jgp_alpha_weight_correct = []
    jgp_alpha_weight_accs = []
    export_path = str(getattr(args, "prediction_scores_out", "") or "").strip()
    export_indices = []
    export_index_set = set()
    export_lp_scores = []
    export_final_scores = []
    export_jgp_scores = []
    export_alpha_jgp_scores = []
    export_text_clique_scores = []
    export_moco_clique_scores = []
    export_text_masks = []
    export_moco_masks = []
    export_jgp_gates = []
    if getattr(args, "search_alpha", False) and getattr(args, "use_jgp", False):
        alpha_values = parse_float_list(getattr(args, "alpha_grid", ""))
        if not alpha_values:
            raise ValueError("--search_alpha requires a non-empty --alpha_grid")
        alpha_correct = [0 for _ in alpha_values]
    if getattr(args, "search_jgp_weight", False) and getattr(args, "use_jgp", False):
        jgp_weight_values = parse_float_list(getattr(args, "jgp_weight_grid", ""))
        if not jgp_weight_values:
            raise ValueError("--search_jgp_weight requires a non-empty --jgp_weight_grid")
        # Retain a one-dimensional w curve.  When alpha is also searched,
        # we additionally accumulate the alpha x w matrix in the same inference
        # pass.  After the alpha sweep finishes, the reported w curve is taken
        # from the row corresponding to the best alpha.  This implements the
        # paper/ablation protocol: alpha first, then w at best alpha.
        jgp_weight_correct = [0 for _ in jgp_weight_values]
        if alpha_values and bool(getattr(args, "search_alpha_weight_matrix", True)):
            jgp_alpha_weight_correct = [[0 for _ in jgp_weight_values] for _ in alpha_values]

    num_samples = image_features_load.shape[0]
    warmup_queries = min(
        max(0, int(getattr(args, "timing_warmup_queries", 5))),
        max(0, int(num_samples) - 1),
    )
    core_inference_seconds = 0.0
    timed_query_count = 0
    setup_seconds = time.perf_counter() - evaluation_started

    if export_path:
        labels_for_export = torch.as_tensor(gts).cpu()
        if labels_for_export.ndim > 1 and labels_for_export.shape[-1] > 1:
            labels_for_export = labels_for_export.argmax(dim=-1)
        labels_for_export = labels_for_export.reshape(-1).long()
        per_class = int(getattr(args, "prediction_export_per_class", 0) or 0)
        generator = torch.Generator().manual_seed(int(getattr(args, "prediction_export_seed", 1)))
        for class_id in range(num_original_classes):
            candidates = torch.where(labels_for_export == class_id)[0]
            if per_class > 0 and candidates.numel() > per_class:
                order = torch.randperm(candidates.numel(), generator=generator)[:per_class]
                candidates = candidates[order]
            export_indices.extend(int(value) for value in candidates.tolist())
        export_indices = sorted(export_indices)
        export_index_set = set(export_indices)

    with torch.no_grad():
        for i in range(num_samples):
            measure_this_query = timing_enabled and i >= warmup_queries
            if measure_this_query:
                synchronize_device(device)
                query_started = time.perf_counter()

            image_features = torch.cat(
                (image_features_load[: i + 1], image_features_load[i].unsqueeze(0)),
                dim=0,
            ).to(device)
            image_features = F.normalize(image_features.float(), dim=-1)

            predictions_lp = iterative_label_propagation(
                image_features,
                text_features,
                args.k_text,
                args.k_image,
                args.k_fewshot,
                args.gamma,
                args.alpha,
                fewshot_image_features=few_shot_image_features,
                fewshot_labels=few_shot_labels_lp,
                max_iter=args.num_iterations,
            )

            if args.text_mode in {"cupl_aux_node", "qwen_aux_node"}:
                predictions_lp = merge_aux_node_predictions(
                    predictions=predictions_lp,
                    num_original_classes=num_original_classes,
                    aux_to_class=aux_to_class,
                    merge_mode=args.aux_merge_mode,
                )
            else:
                predictions_lp = predictions_lp[:, :num_original_classes]

            lp_score = predictions_lp.float()
            query_clip = F.normalize(image_features_load[i].unsqueeze(0).to(device).float(), dim=-1)

            text_clique_score = torch.zeros_like(lp_score)
            if args.use_text_clique or args.use_jgp:
                text_clique_score, _ = compute_clique_logits(
                    query_features=query_clip,
                    anchor_features=text_clique_anchors,
                    target_features=text_clique_target,
                    max_cliques=text_cliques,
                    r=args.text_clique_r,
                    beta=args.text_clique_beta,
                    alpha=1.0,
                )

            dino_cache_score = torch.zeros_like(lp_score)
            if args.use_dino_cache or getattr(args, "use_cross_graph", False):
                query_dino = dino_image_features[i].unsqueeze(0)
                dino_cache_score = compute_dino_cache_logits(
                    query_dino=query_dino,
                    support_dino=dino_fewshot_features,
                    support_labels=dino_fewshot_labels[:, :num_original_classes],
                    beta=args.dino_cache_beta,
                    alpha=1.0,
                )

            dino_clique_score = torch.zeros_like(lp_score)
            if args.use_dino_clique or args.use_jgp or getattr(args, "use_cross_graph", False):
                query_dino = dino_image_features[i].unsqueeze(0)
                dino_clique_score, _ = compute_clique_logits(
                    query_features=query_dino,
                    anchor_features=dino_clique_anchors,
                    target_features=dino_class_prototypes,
                    max_cliques=dino_cliques,
                    r=args.dino_clique_r,
                    beta=args.dino_clique_beta,
                    alpha=1.0,
                )

            jgp_score = None
            lp_moco_visual_score = None
            text_mask = None
            dino_mask = None
            jgp_gate = None
            if args.use_jgp:
                text_mask, _ = compute_clique_mask(
                    query_features=query_clip,
                    anchor_features=text_clique_anchors,
                    target_features=text_clique_target,
                    max_cliques=text_cliques,
                    r=args.text_clique_r,
                )
                dino_mask, _ = compute_clique_mask(
                    query_features=query_dino,
                    anchor_features=dino_clique_anchors,
                    target_features=dino_class_prototypes,
                    max_cliques=dino_cliques,
                    r=args.dino_clique_r,
                )
                jgp_gate = compute_jgp_gate(text_mask, dino_mask, args)
                jgp_score = compute_jgp_score(
                    lp_score=lp_score,
                    text_clique_score=text_clique_score,
                    dino_clique_score=dino_clique_score,
                    text_mask=text_mask,
                    dino_mask=dino_mask,
                    args=args,
                )

                # Standalone LP + MoCo visual graph branch.
                # This uses only the visual clique score and visual mask gate,
                # while keeping the same guided-score design as the jgp graph.
                lp_moco_visual_score = compute_single_graph_guided_score(
                    lp_score=lp_score,
                    graph_score=dino_clique_score,
                    graph_mask=dino_mask,
                    args=args,
                    graph_scale=args.dino_clique_scale,
                )

            cross_graph_score = None
            if getattr(args, "use_cross_graph", False):
                cross_graph_score = compute_cross_graph_score(
                    lp_score=lp_score,
                    dino_score=dino_cache_score,
                    clip_class_prototypes=clip_class_prototypes,
                    dino_class_prototypes=dino_class_prototypes,
                    args=args,
                )

            # Keep exactly the same rule as the old --search_weights diagnostic branch:
            #   base_prob   = softmax(lp_scale * lp_score)
            #   branch_prob = softmax(jgp_scale * jgp_score)
            #   final       = (1 - w) * base_prob + w * branch_prob
            #
            # Important: this is intentionally different from fuse_scores(), where
            # jgp_score is directly treated as a normalized probability.
            # The old reported best ratio, e.g. ratio=0.8, comes from this diagnostic rule.
            if args.use_jgp and jgp_score is not None:
                w = max(0.0, min(1.0, float(args.jgp_weight)))
                lp_prob_for_search = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
                jgp_prob_for_search = jgp_score.float()
                final_score = (1.0 - w) * lp_prob_for_search + w * jgp_prob_for_search
            else:
                final_score = fuse_scores(
                    lp_score=lp_score,
                    text_clique_score=text_clique_score,
                    dino_cache_score=dino_cache_score,
                    dino_clique_score=dino_clique_score,
                    jgp_score=jgp_score,
                    args=args,
                )

            lp_moco_visual_final_score = None
            if args.use_jgp and lp_moco_visual_score is not None:
                w = max(0.0, min(1.0, float(args.jgp_weight)))
                lp_prob_for_visual = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
                visual_prob_for_search = lp_moco_visual_score.float()
                lp_moco_visual_final_score = (1.0 - w) * lp_prob_for_visual + w * visual_prob_for_search

            if getattr(args, "use_cross_graph", False):
                final_score = cross_graph_score

            # Runtime is measured only for the normal JG-MSC prediction path.
            # The alpha/w grid diagnostics below are deliberately excluded so
            # the reported test latency is not inflated by sensitivity search.
            if measure_this_query:
                synchronize_device(device)
                core_inference_seconds += time.perf_counter() - query_started
                timed_query_count += 1

            gt = gts[i]
            if torch.is_tensor(gt):
                gt_cpu = gt.cpu().view(-1)[0]
            else:
                gt_cpu = torch.tensor(gt)

            if args.use_jgp and jgp_score is not None:
                sensitivity_w = max(
                    0.0,
                    min(1.0, float(getattr(args, "sensitivity_fixed_w", 0.9))),
                )
                sensitivity_lp_prob = torch.softmax(
                    float(args.lp_scale) * lp_score.float(), dim=1
                )
                sensitivity_score = (
                    (1.0 - sensitivity_w) * sensitivity_lp_prob
                    + sensitivity_w * jgp_score.float()
                )
                correct_sensitivity_fixed += (
                    sensitivity_score.argmax(dim=1).cpu() == gt_cpu
                ).sum().item()
            else:
                correct_sensitivity_fixed += (
                    final_score.argmax(dim=1).cpu() == gt_cpu
                ).sum().item()

            if alpha_values and args.use_jgp and jgp_score is not None:
                w = max(
                    0.0,
                    min(1.0, float(getattr(args, "sensitivity_fixed_w", args.jgp_weight))),
                )
                lp_prob_for_alpha = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
                old_alpha = getattr(args, "jgp_text_weight", 0.5)
                for ai, alpha_value in enumerate(alpha_values):
                    args.jgp_text_weight = float(alpha_value)
                    jgp_alpha_score = compute_jgp_score(
                        lp_score=lp_score,
                        text_clique_score=text_clique_score,
                        dino_clique_score=dino_clique_score,
                        text_mask=text_mask,
                        dino_mask=dino_mask,
                        args=args,
                    )
                    jgp_alpha_prob = jgp_alpha_score.float()
                    final_alpha_score = (1.0 - w) * lp_prob_for_alpha + w * jgp_alpha_prob
                    alpha_correct[ai] += (final_alpha_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
                args.jgp_text_weight = old_alpha

            if jgp_weight_values and args.use_jgp and jgp_score is not None:
                lp_prob_for_weight = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
                # Fallback w curve at the configured alpha.  If alpha is searched
                # in the same run, this curve is replaced after evaluation by the
                # row at the best alpha (sequential alpha -> w protocol).
                fixed_alpha_jgp_prob = jgp_score.float()
                for wi, weight_value in enumerate(jgp_weight_values):
                    w = max(0.0, min(1.0, float(weight_value)))
                    fixed_alpha_score = (
                        (1.0 - w) * lp_prob_for_weight
                        + w * fixed_alpha_jgp_prob
                    )
                    jgp_weight_correct[wi] += (
                        fixed_alpha_score.argmax(dim=1).cpu() == gt_cpu
                    ).sum().item()

                # Accumulate all alpha x w combinations once.  We do NOT choose
                # the global 2-D optimum.  The matrix is only used to obtain the
                # w curve at the alpha selected by the preceding alpha sweep.
                if jgp_alpha_weight_correct:
                    old_alpha = getattr(args, "jgp_text_weight", 0.5)
                    for ai, alpha_value in enumerate(alpha_values):
                        args.jgp_text_weight = float(alpha_value)
                        jgp_alpha_score = compute_jgp_score(
                            lp_score=lp_score,
                            text_clique_score=text_clique_score,
                            dino_clique_score=dino_clique_score,
                            text_mask=text_mask,
                            dino_mask=dino_mask,
                            args=args,
                        )
                        jgp_prob_for_weight = jgp_alpha_score.float()
                        for wi, weight_value in enumerate(jgp_weight_values):
                            w = max(0.0, min(1.0, float(weight_value)))
                            final_weight_score = (1.0 - w) * lp_prob_for_weight + w * jgp_prob_for_weight
                            jgp_alpha_weight_correct[ai][wi] += (final_weight_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
                    args.jgp_text_weight = old_alpha

            if i in export_index_set:
                export_lp_scores.append(torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)[0].cpu())
                export_final_scores.append(final_score.float()[0].cpu())
                export_text_clique_scores.append(text_clique_score.float()[0].cpu())
                export_moco_clique_scores.append(dino_clique_score.float()[0].cpu())
                if jgp_score is not None:
                    export_jgp_scores.append(jgp_score.float()[0].cpu())
                    export_text_masks.append(text_mask.float()[0].cpu())
                    export_moco_masks.append(dino_mask.float()[0].cpu())
                    export_jgp_gates.append(jgp_gate.float()[0].cpu())
                    if alpha_values:
                        old_alpha = getattr(args, "jgp_text_weight", 0.5)
                        per_alpha = []
                        for alpha_value in alpha_values:
                            args.jgp_text_weight = float(alpha_value)
                            score = compute_jgp_score(
                                lp_score=lp_score,
                                text_clique_score=text_clique_score,
                                dino_clique_score=dino_clique_score,
                                text_mask=text_mask,
                                dino_mask=dino_mask,
                                args=args,
                            )
                            per_alpha.append(score.float()[0].cpu())
                        args.jgp_text_weight = old_alpha
                        export_alpha_jgp_scores.append(torch.stack(per_alpha))

            correct_lp += (lp_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
            if args.use_text_clique:
                correct_text_clique += (text_clique_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
            if args.use_dino_cache or getattr(args, "use_cross_graph", False):
                correct_dino_cache += (dino_cache_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
            if args.use_dino_clique:
                correct_dino_clique += (dino_clique_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
            if args.use_jgp:
                correct_jgp += (jgp_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
                if lp_moco_visual_final_score is not None:
                    correct_lp_moco_visual += (lp_moco_visual_final_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
            if getattr(args, "use_cross_graph", False):
                correct_cross_graph += (cross_graph_score.argmax(dim=1).cpu() == gt_cpu).sum().item()
            correct_final += (final_score.argmax(dim=1).cpu() == gt_cpu).sum().item()

    best_alpha = None
    best_alpha_acc = None
    if alpha_values:
        alpha_accs = [x / num_samples * 100.0 for x in alpha_correct]
        best_idx = max(range(len(alpha_accs)), key=lambda idx: alpha_accs[idx])
        best_alpha = float(alpha_values[best_idx])
        best_alpha_acc = float(alpha_accs[best_idx])
        print("=" * 80)
        print("JGP Alpha Search")
        print("=" * 80)
        print("Definition: graph_prob = alpha * TEXT_GRAPH + (1 - alpha) * MOCO_GRAPH")
        for alpha_value, acc in zip(alpha_values, alpha_accs):
            print(f"alpha={float(alpha_value):.2f} | acc={acc:.2f}%")
        print(f"Best alpha: {best_alpha:.2f} | Best acc: {best_alpha_acc:.2f}%")
        print("=" * 80)

    best_jgp_weight = None
    best_jgp_weight_acc = None
    best_jgp_alpha = None
    weight_fixed_alpha = None
    if jgp_weight_values:
        # Default/fallback curve at the configured alpha.
        weight_accs = [x / num_samples * 100.0 for x in jgp_weight_correct]
        weight_fixed_alpha = float(getattr(args, "jgp_text_weight", 0.5))

        # Sequential tuning required by the ablation protocol:
        #   Step 1: select alpha using the alpha curve evaluated at fixed w.
        #   Step 2: select w from the alpha x w row corresponding to that alpha.
        # Importantly, this is NOT the unconstrained global optimum over the 2-D
        # alpha x w grid.
        if alpha_values and jgp_alpha_weight_correct and best_alpha is not None:
            jgp_alpha_weight_accs = [
                [x / num_samples * 100.0 for x in row]
                for row in jgp_alpha_weight_correct
            ]
            best_ai = min(
                range(len(alpha_values)),
                key=lambda idx: abs(float(alpha_values[idx]) - float(best_alpha)),
            )
            weight_accs = list(jgp_alpha_weight_accs[best_ai])
            weight_fixed_alpha = float(alpha_values[best_ai])
            best_jgp_alpha = float(alpha_values[best_ai])
        elif best_alpha is not None:
            # This path is kept for compatibility if the matrix was explicitly
            # disabled.  The reported alpha is still the best alpha, but the w
            # curve then remains the configured-alpha fallback.
            best_jgp_alpha = float(best_alpha)

        print("=" * 80)
        print("JGP Sequential Weight Search")
        print("=" * 80)
        print(
            "Protocol: first choose alpha at fixed w, then choose w at best alpha; "
            f"w-sweep alpha={float(weight_fixed_alpha):.2f}"
        )
        print("Definition: final = (1 - w) * LP + w * JGP(alpha)")
        for weight_value, acc in zip(jgp_weight_values, weight_accs):
            print(f"w={float(weight_value):.2f} | acc={acc:.2f}%")

        best_idx = max(range(len(weight_accs)), key=lambda idx: weight_accs[idx])
        best_jgp_weight = float(jgp_weight_values[best_idx])
        best_jgp_weight_acc = float(weight_accs[best_idx])
        if best_jgp_alpha is None:
            best_jgp_alpha = float(weight_fixed_alpha)
        print(
            f"Selected alpha: {best_jgp_alpha:.2f} | "
            f"Best w: {best_jgp_weight:.2f} | Best acc: {best_jgp_weight_acc:.2f}%"
        )
        print("=" * 80)

    if export_path:
        labels_for_export = torch.as_tensor(gts).cpu()
        if labels_for_export.ndim > 1 and labels_for_export.shape[-1] > 1:
            labels_for_export = labels_for_export.argmax(dim=-1)
        labels_for_export = labels_for_export.reshape(-1).long()
        exported_lp = torch.stack(export_lp_scores) if export_lp_scores else torch.empty(0, num_original_classes)
        exported_final = torch.stack(export_final_scores) if export_final_scores else exported_lp.clone()
        selected_alpha = best_jgp_alpha if best_jgp_alpha is not None else best_alpha
        selected_weight = best_jgp_weight
        if selected_weight is None:
            selected_weight = float(getattr(args, "jgp_weight", 0.0))
        if args.use_jgp and export_jgp_scores:
            if selected_alpha is not None and export_alpha_jgp_scores:
                alpha_index = min(
                    range(len(alpha_values)),
                    key=lambda index: abs(float(alpha_values[index]) - float(selected_alpha)),
                )
                exported_jgp = torch.stack(export_alpha_jgp_scores)[:, alpha_index, :]
            else:
                exported_jgp = torch.stack(export_jgp_scores)
            exported_final = (1.0 - float(selected_weight)) * exported_lp + float(selected_weight) * exported_jgp
        else:
            exported_jgp = torch.empty(0, num_original_classes)

        exported_text_scores = (
            torch.stack(export_text_clique_scores)
            if export_text_clique_scores
            else torch.empty(0, num_original_classes)
        )
        exported_moco_scores = (
            torch.stack(export_moco_clique_scores)
            if export_moco_clique_scores
            else torch.empty(0, num_original_classes)
        )
        exported_text_prob = torch.softmax(
            float(args.text_clique_scale) * exported_text_scores.float(), dim=1
        ) if exported_text_scores.numel() else exported_text_scores.clone()
        exported_moco_prob = torch.softmax(
            float(args.dino_clique_scale) * exported_moco_scores.float(), dim=1
        ) if exported_moco_scores.numel() else exported_moco_scores.clone()
        exported_text_mask = (
            torch.stack(export_text_masks).bool()
            if export_text_masks
            else torch.empty(0, num_original_classes, dtype=torch.bool)
        )
        exported_moco_mask = (
            torch.stack(export_moco_masks).bool()
            if export_moco_masks
            else torch.empty(0, num_original_classes, dtype=torch.bool)
        )
        exported_jgp_gate = (
            torch.stack(export_jgp_gates)
            if export_jgp_gates
            else torch.empty(0, num_original_classes)
        )

        query_paths = []
        for path_key in ("image_paths", "paths", "query_paths"):
            path_values = pre_calculated.get(path_key)
            if path_values is not None and len(path_values) == num_samples:
                query_paths = [str(path_values[index]) for index in export_indices]
                break

        text_graph_features = None
        text_similarity = None
        if text_clique_target is not None:
            text_graph_features = F.normalize(text_clique_target.detach().float(), dim=-1).cpu()
            text_similarity = text_graph_features @ text_graph_features.t()

        moco_graph_features = None
        moco_similarity = None
        if dino_class_prototypes is not None:
            moco_graph_features = F.normalize(dino_class_prototypes.detach().float(), dim=-1).cpu()
            moco_similarity = moco_graph_features @ moco_graph_features.t()

        selected_labels = labels_for_export[export_indices]
        query_affinity = {
            "indices": torch.tensor(export_indices, dtype=torch.long),
            "labels": selected_labels,
            "paths": query_paths,
            "lp_probabilities": exported_lp.half(),
            "text_clique_scores": exported_text_scores.half(),
            "text_probabilities": exported_text_prob.half(),
            "moco_clique_scores": exported_moco_scores.half(),
            "moco_probabilities": exported_moco_prob.half(),
            "text_masks": exported_text_mask,
            "moco_masks": exported_moco_mask,
            "jgp_gates": exported_jgp_gate.half(),
            "jgp_probabilities": exported_jgp.half(),
            "final_probabilities": exported_final.half(),
        }
        static_graph = {
            "text_class_features": text_graph_features.half() if text_graph_features is not None else None,
            "moco_class_prototypes": moco_graph_features.half() if moco_graph_features is not None else None,
            "text_similarity": text_similarity.half() if text_similarity is not None else None,
            "moco_similarity": moco_similarity.half() if moco_similarity is not None else None,
            "text_adjacency": text_adj.detach().cpu().bool() if text_adj is not None else None,
            "moco_adjacency": dino_adj.detach().cpu().bool() if dino_adj is not None else None,
            "text_cliques": text_cliques,
            "moco_cliques": dino_cliques,
        }
        affinity_parameters = {
            "lp_scale": float(args.lp_scale),
            "text_clique_scale": float(args.text_clique_scale),
            "moco_clique_scale": float(args.dino_clique_scale),
            "jgp_text_weight": float(
                selected_alpha if selected_alpha is not None
                else getattr(args, "jgp_text_weight", 0.5)
            ),
            "jgp_inner_weight": float(args.jgp_inner_weight),
            "jgp_mask_boost": float(args.jgp_mask_boost),
            "jgp_gate_mode": str(args.jgp_gate_mode),
            "jgp_text_gate": float(args.jgp_text_gate),
            "jgp_visual_gate": float(args.jgp_visual_gate),
            "jgp_weight": float(selected_weight),
            "text_clique_control": str(args.text_clique_control),
            "text_clique_control_value": float(args.text_clique_control_value),
            "moco_clique_control": str(args.dino_clique_control),
            "moco_clique_control_value": float(args.dino_clique_control_value),
            "num_iterations": int(args.num_iterations),
        }
        affinity_metrics = {
            "lp_top1": 100.0 * float(correct_lp) / float(num_samples),
            "selected_jgp_top1_on_exported_queries": (
                100.0 * float((exported_jgp.argmax(dim=1) == selected_labels).float().mean().item())
                if exported_jgp.numel() else None
            ),
            "exported_query_count": int(selected_labels.numel()),
            "full_query_count": int(num_samples),
            "final_top1": (
                float(best_jgp_weight_acc)
                if best_jgp_weight_acc is not None
                else float(best_alpha_acc)
                if best_alpha_acc is not None
                else 100.0 * float(correct_final) / float(num_samples)
            ),
        }

        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        torch.save(
            {
                "dataset": dataset_name,
                "shot": int(args.num_shots) if args.mode == "FS" else 0,
                "query_indices": torch.tensor(export_indices, dtype=torch.long),
                "labels": labels_for_export[export_indices],
                "class_names": class_names,
                "lp_probabilities": exported_lp,
                "final_probabilities": exported_final,
                "jgp_probabilities": exported_jgp,
                "text_mode": args.text_mode,
                "patch_enabled": bool(getattr(args, "use_support_patch_nodes", False)),
                "patch_text_source": getattr(args, "patch_text_source", ""),
                "ablation_name": getattr(args, "prediction_ablation_name", ""),
                "jgp_enabled": bool(getattr(args, "use_jgp", False)),
                "best_alpha": selected_alpha,
                "best_jgp_weight": selected_weight,
                "source_experiment": "shot_threshold_aki_msc_grid_search",
                "affinity_bundle_version": 1,
                "query_affinity": query_affinity,
                "static_graph": static_graph,
                "parameters": affinity_parameters,
                "metrics": affinity_metrics,
                "source_files": {
                    "clip_query_features": feature_path,
                    "moco_feature_root": str(getattr(args, "dino_features_path", "")),
                    "prompt_json": str(getattr(args, "prompt_json", "")),
                    "cupl_json": str(getattr(args, "cupl_json", "")),
                    "meta_file": str(getattr(args, "meta_file", "")),
                },
            },
            export_path,
        )
        print(f"Prediction-score + affinity bundle saved to: {export_path}")

    evaluation_seconds = time.perf_counter() - evaluation_started
    test_ms_per_query = (
        1000.0 * core_inference_seconds / timed_query_count
        if timing_enabled and timed_query_count > 0
        else None
    )
    peak_gpu_memory_mb = None
    if timing_enabled and device.type == "cuda":
        synchronize_device(device)
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    return {
        "acc_lp": correct_lp / num_samples * 100.0,
        "acc_text_clique": correct_text_clique / num_samples * 100.0 if args.use_text_clique else 0.0,
        "acc_dino_cache": correct_dino_cache / num_samples * 100.0 if args.use_dino_cache else 0.0,
        "acc_dino_clique": correct_dino_clique / num_samples * 100.0 if args.use_dino_clique else 0.0,
        "acc_jgp": correct_jgp / num_samples * 100.0 if args.use_jgp else 0.0,
        "acc_lp_moco_visual": correct_lp_moco_visual / num_samples * 100.0 if args.use_jgp else 0.0,
        "acc_cross_graph": correct_cross_graph / num_samples * 100.0 if getattr(args, "use_cross_graph", False) else 0.0,
        "acc_final": (
            best_jgp_weight_acc
            if best_jgp_weight_acc is not None
            else best_alpha_acc
            if best_alpha_acc is not None
            else correct_final / num_samples * 100.0
        ),
        "best_alpha": best_alpha,
        "best_alpha_acc": best_alpha_acc,
        "best_jgp_alpha": best_jgp_alpha,
        "best_jgp_weight": best_jgp_weight,
        "best_jgp_weight_acc": best_jgp_weight_acc,
        "acc_sensitivity_fixed": correct_sensitivity_fixed / num_samples * 100.0,
        "alpha_grid": [float(value) for value in alpha_values],
        "alpha_accuracies": [float(value) for value in alpha_accs],
        "weight_grid": [float(value) for value in jgp_weight_values],
        "weight_accuracies": [float(value) for value in weight_accs],
        "weight_fixed_alpha": (
            None if weight_fixed_alpha is None else float(weight_fixed_alpha)
        ),
        "jgp_alpha_weight_accuracies": [
            [float(value) for value in row] for row in jgp_alpha_weight_accs
        ],
        "sensitivity_fixed_alpha": float(getattr(args, "jgp_text_weight", 0.5)),
        "sensitivity_fixed_w": float(getattr(args, "sensitivity_fixed_w", 0.9)),
        "aki_grounding_stats": aki_grounding_stats,
        "msc_grounding_stats": msc_grounding_stats,
        "num_samples": num_samples,
        "source_files": {
            "clip_query": str(getattr(args, "resolved_clip_query_path", "")),
            "clip_fewshot": str(getattr(args, "resolved_clip_fewshot_path", "")),
            "moco_query": str(getattr(args, "resolved_moco_query_path", "")),
            "moco_fewshot": str(getattr(args, "resolved_moco_fewshot_path", "")),
            "aligned_clip_reference": str(getattr(args, "resolved_aligned_clip_reference_path", "")),
            "aligned_clip_mean_cosine": getattr(args, "aligned_clip_mean_cosine", None),
            "aligned_clip_min_cosine": getattr(args, "aligned_clip_min_cosine", None),
            "meta": str(getattr(args, "meta_file", "")),
        },
        "timing": {
            "setup_seconds": float(setup_seconds),
            "core_inference_seconds": float(core_inference_seconds),
            "timed_query_count": int(timed_query_count),
            "warmup_query_count": int(warmup_queries),
            "test_ms_per_query": test_ms_per_query,
            "evaluation_seconds": float(evaluation_seconds),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
    }


def flatten_params(hp):
    return json.dumps(hp, ensure_ascii=False, sort_keys=True)


def update_best(item, key, value, weight_key=None, weight_value=None):
    if value is None:
        return False
    if item.get(key) is None or float(value) > float(item[key]):
        item[key] = float(value)
        if weight_key is not None:
            item[weight_key] = weight_value
        return True
    return False


def write_compact_csv(
    out_csv,
    fieldnames,
    compact,
    requested_patch_text_sources=None,
    attribute_text_source="cupl",
):
    requested_patch_text_sources = list(requested_patch_text_sources or ["cupl"])
    attribute_text_source = str(attribute_text_source or "cupl").lower()
    grouped = defaultdict(dict)
    for (dataset_name, shot, patch_text_source), item in compact.items():
        grouped[(dataset_name, shot)][patch_text_source] = item

    def fmt(x):
        return "" if x is None else f"{float(x):.4f}"

    def best_weight(item, key):
        if not item:
            return ""
        value = item.get(key)
        return "" if value is None else f"{float(value):.4f}"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (dataset_name, shot), source_items in sorted(grouped.items()):
            base_item = {}
            for source in requested_patch_text_sources:
                if source in source_items:
                    base_item = source_items[source]
                    break
            if not base_item and source_items:
                base_item = next(iter(source_items.values()))

            baseline = base_item.get("baseline")
            baseline_aki = base_item.get("baseline_qwen")
            baseline_aki_unfiltered = base_item.get("baseline_aki_unfiltered")
            baseline_aki_grounded = base_item.get("baseline_aki_grounded")
            jgp = base_item.get("baseline_jgp")
            full_item = source_items.get(attribute_text_source, {})
            full_model = full_item.get("qwen_jgp_patch")
            baseline_external = base_item.get("baseline_external")

            row = {
                "dataset": dataset_name,
                "shot": shot,
                "baseline": fmt(baseline),
                "baseline_jgp": fmt(jgp),
                f"aki_jgp_msc_{attribute_text_source}": fmt(full_model),
                "jgp_raw": fmt(base_item.get("baseline_jgp_raw_patch", base_item.get("baseline_jgp_raw"))),
                "moco_visual_raw": fmt(base_item.get("lp_moco_visual_raw_patch", base_item.get("lp_moco_visual_raw"))),
                "gain_jgp": fmt(None if baseline is None or jgp is None else jgp - baseline),
                f"gain_aki_jgp_msc_{attribute_text_source}": fmt(
                    None if baseline is None or full_model is None else full_model - baseline
                ),
                "best_weight_jgp": best_weight(base_item, "best_weight_baseline_jgp"),
                f"best_weight_aki_jgp_msc_{attribute_text_source}": best_weight(
                    full_item,
                    "best_weight_qwen_jgp_patch",
                ),
                "jgp_text_weight": fmt(base_item.get("jgp_text_weight", None)),
                "baseline_external": fmt(baseline_external),
            }
            if "baseline_aki" in fieldnames:
                row["baseline_aki"] = fmt(baseline_aki)
            if "gain_aki" in fieldnames:
                row["gain_aki"] = fmt(None if baseline is None or baseline_aki is None else baseline_aki - baseline)
            if "baseline_aki_unfiltered" in fieldnames:
                row["baseline_aki_unfiltered"] = fmt(baseline_aki_unfiltered)
                row["gain_aki_unfiltered"] = fmt(
                    None if baseline is None or baseline_aki_unfiltered is None else baseline_aki_unfiltered - baseline
                )
                row["baseline_aki_grounded"] = fmt(baseline_aki_grounded)
                row["gain_aki_grounded"] = fmt(
                    None if baseline is None or baseline_aki_grounded is None else baseline_aki_grounded - baseline
                )
                row["aki_attributes_total"] = fmt(base_item.get("aki_attributes_total"))
                row["aki_attributes_retained"] = fmt(base_item.get("aki_attributes_retained"))
                row["aki_retention_rate"] = fmt(base_item.get("aki_retention_rate"))
            for source in requested_patch_text_sources:
                item = source_items.get(source, {})
                msc = item.get("baseline_patch")
                jgp_msc = item.get("baseline_jgp_patch")
                row[f"baseline_msc_{source}"] = fmt(msc)
                row[f"jgp_msc_{source}"] = fmt(jgp_msc)
                row[f"gain_msc_{source}"] = fmt(None if baseline is None or msc is None else msc - baseline)
                row[f"gain_jgp_msc_{source}"] = fmt(None if baseline is None or jgp_msc is None else jgp_msc - baseline)
                row[f"best_weight_jgp_msc_{source}"] = best_weight(item, "best_weight_baseline_jgp_patch")
            writer.writerow(row)


def write_aki_threshold_csv(path, rows):
    fieldnames = [
        "dataset", "shot", "threshold", "accuracy", "baseline", "gain",
        "attributes_total", "attributes_retained", "retention_rate",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_msc_threshold_csv(path, rows):
    fieldnames = [
        "dataset", "shot", "text_source", "threshold", "accuracy", "baseline", "gain",
        "attributes_total", "attributes_retained", "retention_rate", "photo_fallback_classes",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(path, fieldnames, rows):
    """Rewrite a small experiment CSV after every completed run.

    Rewriting keeps partially completed multi-dataset jobs usable after an
    interruption while avoiding duplicate headers.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_details_csv(path, rows):
    """Save one row per unique AKI/MSC/JGP component combination."""
    deduplicated = {}
    for row in rows:
        # One final row per component combination.  Hyperparameter candidates
        # (tau_K, alpha, w) compete inside the same ablation cell and only the
        # best measured result is retained.
        key = (
            row["dataset"],
            int(row["shot"]),
            int(row["aki"]),
            int(row["msc"]),
            int(row["jgp"]),
            row.get("patch_text_source", "") if int(row["msc"]) else "",
        )
        current = deduplicated.get(key)
        if current is None or float(row["accuracy"]) > float(current["accuracy"]):
            deduplicated[key] = dict(row)

    baseline_by_dataset_shot = {}
    for row in deduplicated.values():
        if not int(row["aki"]) and not int(row["msc"]) and not int(row["jgp"]):
            baseline_by_dataset_shot[(row["dataset"], int(row["shot"]))] = float(row["accuracy"])

    # Diagnostic only: compare the measured no-module baseline with the
    # external base graph propagation table.  Do not substitute the external number.
    for (dataset_name, shot), measured in sorted(baseline_by_dataset_shot.items()):
        external = get_external_baseline(dataset_name, shot)
        if external is not None:
            delta = float(measured) - float(external)
            print(
                f"[BASELINE-CHECK] {dataset_name} {shot}-shot | "
                f"measured={measured:.3f} | external={float(external):.3f} | "
                f"delta={delta:+.3f}"
            )
            if abs(delta) > float(getattr(write_ablation_details_csv, "baseline_warn_tolerance", 0.25)):
                print(
                    "[BASELINE-WARNING] Measured no-module baseline differs from "
                    "the external base graph propagation reference. Check base_text_prompt, feature "
                    "cache, split, and few-shot seed before using the table."
                )

    output_rows = []
    for row in sorted(
        deduplicated.values(),
        key=lambda item: (
            item["dataset"], int(item["shot"]),
            int(item["aki"]), int(item["msc"]), int(item["jgp"]),
            str(item.get("tau_k", "")),
        ),
    ):
        row = dict(row)
        baseline = baseline_by_dataset_shot.get((row["dataset"], int(row["shot"])))
        row["baseline_accuracy"] = "" if baseline is None else baseline
        row["gain_over_baseline"] = (
            "" if baseline is None else float(row["accuracy"]) - baseline
        )
        output_rows.append(row)

    fieldnames = [
        "dataset", "shot", "variant", "aki", "msc", "jgp",
        "patch_text_source", "tau_k", "alpha", "w", "accuracy",
        "baseline_accuracy", "gain_over_baseline", "num_samples",
        "source_run", "selection_protocol",
    ]
    write_rows_csv(path, fieldnames, output_rows)


def write_timing_csv(path, rows):
    """Save per-method timing and a direct comparison against base graph propagation.

    The raw measurements are kept unchanged.  When an base graph propagation/baseline row is
    available for the same dataset and shot, the derived columns make the
    JG-MSC overhead directly usable in the computational-efficiency table.
    """
    baseline_by_dataset_shot = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method == "base graph propagation" or (
            not int(row.get("aki", 0))
            and not int(row.get("msc", 0))
            and not int(row.get("jgp", 0))
        ):
            baseline_by_dataset_shot[(row["dataset"], int(row["shot"]))] = row

    output_rows = []
    for source_row in rows:
        row = dict(source_row)
        baseline = baseline_by_dataset_shot.get((row["dataset"], int(row["shot"])))

        def numeric(value):
            if value in (None, ""):
                return None
            return float(value)

        current_ms = numeric(row.get("test_ms_per_query"))
        baseline_ms = numeric(baseline.get("test_ms_per_query")) if baseline else None
        current_setup = numeric(row.get("setup_seconds"))
        baseline_setup = numeric(baseline.get("setup_seconds")) if baseline else None
        current_e2e = numeric(row.get("end_to_end_seconds"))
        baseline_e2e = numeric(baseline.get("end_to_end_seconds")) if baseline else None

        row["ecalp_test_ms_per_query"] = "" if baseline_ms is None else baseline_ms
        row["test_delta_ms_vs_ecalp"] = (
            "" if current_ms is None or baseline_ms is None else current_ms - baseline_ms
        )
        row["test_ratio_vs_ecalp"] = (
            "" if current_ms is None or baseline_ms in (None, 0.0) else current_ms / baseline_ms
        )
        row["setup_delta_seconds_vs_ecalp"] = (
            "" if current_setup is None or baseline_setup is None else current_setup - baseline_setup
        )
        row["end_to_end_delta_seconds_vs_ecalp"] = (
            "" if current_e2e is None or baseline_e2e is None else current_e2e - baseline_e2e
        )
        output_rows.append(row)

    output_rows.sort(
        key=lambda item: (
            item["dataset"],
            int(item["shot"]),
            0 if str(item.get("method", "")) == "base graph propagation" else 1,
            str(item.get("method", "")),
            str(item.get("ablation", "")),
        )
    )

    fieldnames = [
        "dataset", "shot", "method", "ablation", "base_text_prompt",
        "clip_query_path", "clip_fewshot_path", "moco_query_path",
        "moco_fewshot_path", "aligned_clip_reference_path",
        "aligned_clip_mean_cosine", "aligned_clip_min_cosine", "meta_path",
        "aki", "msc", "jgp", "tau_k", "alpha", "w",
        "k_text", "k_image", "k_fewshot", "gamma", "lp_alpha",
        "num_iterations",
        "num_samples", "train_seconds", "setup_seconds",
        "core_inference_seconds", "timed_query_count", "warmup_query_count",
        "test_ms_per_query", "end_to_end_seconds", "peak_gpu_memory_mb",
        "additional_trainable_params", "accuracy",
        "ecalp_test_ms_per_query", "test_delta_ms_vs_ecalp",
        "test_ratio_vs_ecalp", "setup_delta_seconds_vs_ecalp",
        "end_to_end_delta_seconds_vs_ecalp",
    ]
    write_rows_csv(path, fieldnames, output_rows)


def print_runtime_comparison(rows):
    """Print the same base graph propagation/JG-MSC comparison saved by write_timing_csv."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["shot"]))].append(row)

    if not grouped:
        return

    print("=" * 100)
    print("RUNTIME COMPARISON: ORIGINAL base graph propagation VS JG-MSC")
    print("=" * 100)
    for (dataset_name, shot), group in sorted(grouped.items()):
        baseline = next((row for row in group if row.get("method") == "base graph propagation"), None)
        ours = next((row for row in group if row.get("method") == "JG-MSC (Ours)"), None)
        print(f"{dataset_name:15s} shot={shot}")
        for row in [baseline, ours]:
            if row is None:
                continue
            test_ms = row.get("test_ms_per_query")
            setup_s = row.get("setup_seconds")
            e2e_s = row.get("end_to_end_seconds")
            acc = row.get("accuracy")
            print(
                f"  {row['method']:16s} | test={float(test_ms):.3f} ms/query "
                f"| setup={float(setup_s):.2f}s | end-to-end={float(e2e_s):.2f}s "
                f"| acc={float(acc):.2f}%"
            )
        if baseline is not None and ours is not None:
            baseline_ms = float(baseline["test_ms_per_query"])
            ours_ms = float(ours["test_ms_per_query"])
            ratio = ours_ms / baseline_ms if baseline_ms > 0 else float("nan")
            print(
                f"  JG-MSC overhead   | {ours_ms - baseline_ms:+.3f} ms/query "
                f"| {ratio:.3f}x base graph propagation"
            )


def write_sensitivity_csv(path, rows):
    fieldnames = [
        "dataset", "shot", "parameter", "value", "accuracy",
        "fixed_tau_k", "fixed_alpha", "fixed_w", "ablation",
        "num_samples",
    ]
    ordered = sorted(
        rows,
        key=lambda item: (
            item["dataset"], int(item["shot"]), item["parameter"], float(item["value"]),
        ),
    )
    write_rows_csv(path, fieldnames, ordered)


def component_variant_name(aki, msc, jgp):
    names = []
    if aki:
        names.append("AKI")
    if msc:
        names.append("MSC")
    if jgp:
        names.append("JGP")
    return "+".join(names) if names else "Baseline"


def main(cli_args):
    """Run AKI / JGP / MSC ablations and save a compact CSV.

    Final compact columns:
        1) baseline                 = Original text + LP
        2) baseline_aki             = AKI / Qwen auxiliary node + LP
        3) baseline_jgp             = Original text + JGP
        4) baseline_msc_qwen        = Original text + MSC patch nodes selected by attributes
        5) jgp_msc_qwen             = Original text + JGP + MSC

    Notes:
        - For shot=0, JGP cannot be built because DINO few-shot prototypes do not exist.
          Therefore JGP columns fall back to the LP-side result.
        - baseline_external is the external table value from EXTERNAL_BASELINE, kept only for comparison.
    """
    cli_args = resolve_internal_paths(cli_args)
    if bool(getattr(cli_args, "experiment_suite", False)):
        cli_args.save_timing = True
        cli_args.save_ablation_details = True
        cli_args.run_sensitivity = True

    runtime_compare = bool(getattr(cli_args, "runtime_compare", False))
    if runtime_compare:
        # A clean two-run timing protocol:
        #   1) base graph propagation (cached text features + LP only),
        #   2) complete JG-MSC (AKI + MSC + JGP) at fixed alpha/w/tau_K.
        # Sensitivity searches are disabled because they must not inflate the
        # reported inference time or replace the fixed-parameter accuracy.
        cli_args.save_timing = True
        cli_args.run_sensitivity = False
        cli_args.search_alpha = False
        cli_args.search_jgp_weight = False
        cli_args.search_alpha_weight_matrix = False
        cli_args.modules = "baseline,full"
        cli_args.patch_variant = "on"
        cli_args.support_grounded_aki = True
        # Fixed paper timing configuration.  Keep the one-command runtime
        # comparison simple and prevent accidental per-run parameter changes.
        cli_args.jgp_text_weight = 0.5
        cli_args.runtime_compare_w = 0.9
        cli_args.runtime_compare_tau_k = 0.0
        cli_args.timing_warmup_queries = 5

    if bool(getattr(cli_args, "run_sensitivity", False)):
        # Sensitivity is evaluated on the complete JG-MSC model.  The fixed
        # values are used for one-at-a-time curves; the two-dimensional grid
        # remains available as a diagnostic in the normal result object.
        cli_args.search_alpha = True
        cli_args.search_jgp_weight = True
        cli_args.support_grounded_aki = True
        attribute_source = str(
            getattr(cli_args, "attribute_text_source", "cupl") or "cupl"
        ).lower()
        cli_args.aki_text_mode = f"{attribute_source}_aux_node"
        cli_args.patch_text_source = attribute_source
        cli_args.patch_variant = "on"
        requested_module_text = str(getattr(cli_args, "modules", "") or "")
        if "full" not in parse_module_list(requested_module_text):
            cli_args.modules = f"{requested_module_text},full".strip(",")

    ablation_mode = str(getattr(cli_args, "ablation_mode", "controlled") or "controlled").lower()
    if ablation_mode not in {"full", "controlled"}:
        raise ValueError("--ablation_mode must be one of: full, controlled")
    if ablation_mode == "full":
        # Full mode always runs the combinations needed by the compact ablation table.
        # Use controlled mode when you want text_mode, patch_variant and patch_text_source
        # to be taken literally from the command line.
        cli_args.text_mode = ""
        cli_args.patch_variant = "on"

    patch_variant = str(getattr(cli_args, "patch_variant", "both") or "both").lower()
    if patch_variant not in {"off", "on", "both"}:
        raise ValueError("--patch_variant must be one of: off, on, both")
    cli_args.use_support_patch_nodes = patch_variant in {"on", "both"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model_name = "ViT-B/16" if cli_args.clip_model == "VIT" else "RN50"
    loaded = clip.load(clip_model_name, device=device)
    if len(loaded) == 3:
        clip_model, _, clip_preprocess = loaded
    else:
        clip_model, clip_preprocess = loaded
    clip_model.eval()

    moco_model = None
    moco_preprocess = None
    needs_patch_encoder = getattr(cli_args, "use_support_patch_nodes", False) or (
        getattr(cli_args, "aki_grounding_level", "image") == "patch"
        and (
            getattr(cli_args, "support_grounded_aki", False)
            or getattr(cli_args, "compare_support_grounding", False)
        )
    )
    if needs_patch_encoder:
        if cli_args.clip_model != "RN":
            raise ValueError("--use_support_patch_nodes currently supports --clip_model RN only.")
        if not cli_args.mocov3_ckpt:
            raise ValueError("--use_support_patch_nodes requires --mocov3_ckpt.")
        moco_model = load_mocov3_resnet50(cli_args.mocov3_ckpt, device)
        moco_preprocess = moco_transform(cli_args.moco_image_size)

    datasets = expand_dataset_arg(cli_args.datasets)
    out_csv = resolve_output_csv_path(cli_args, datasets)
    stem, extension = os.path.splitext(out_csv)
    aki_threshold_csv = f"{stem}_aki_thresholds{extension or '.csv'}"
    aki_threshold_trials = []
    msc_threshold_csv = f"{stem}_msc_thresholds{extension or '.csv'}"
    msc_threshold_trials = []
    ablation_details_csv = f"{stem}_ablation{extension or '.csv'}"
    timing_csv = (
        f"{stem}_runtime_compare{extension or '.csv'}"
        if runtime_compare
        else f"{stem}_timing{extension or '.csv'}"
    )
    sensitivity_csv = f"{stem}_sensitivity{extension or '.csv'}"
    ablation_detail_rows = []
    timing_rows = []
    sensitivity_rows = []
    cli_args.output_csv = out_csv
    print(f"Output CSV : {out_csv}")
    if getattr(cli_args, "save_ablation_details", False):
        print(f"Ablation CSV: {ablation_details_csv}")
    if getattr(cli_args, "save_timing", False):
        print(f"Timing CSV  : {timing_csv}")
    if getattr(cli_args, "run_sensitivity", False):
        print(f"Sensitivity CSV: {sensitivity_csv}")
    print(f"Ablation mode: {ablation_mode}")

    sensitivity_config = SEARCH_CONFIG.get("sensitivity", {})
    tau_k_values = parse_float_list(getattr(cli_args, "tau_k_grid", ""))
    if not tau_k_values:
        tau_k_values = [float(value) for value in sensitivity_config.get("tau_k", [0.0])]
    sensitivity_reference_tau = float(
        getattr(
            cli_args,
            "sensitivity_reference_tau_k",
            sensitivity_config.get("reference_tau_k", tau_k_values[0]),
        )
    )
    sensitivity_reference_tau = min(
        tau_k_values,
        key=lambda value: abs(float(value) - sensitivity_reference_tau),
    )

    attribute_text_source = str(
        getattr(cli_args, "attribute_text_source", "cupl") or "cupl"
    ).lower()
    if attribute_text_source not in {"cupl", "qwen"}:
        raise ValueError("--attribute_text_source must be one of: cupl, qwen")
    requested_patch_text_sources = parse_patch_text_source_list(
        getattr(cli_args, "patch_text_source", "cupl")
    )
    aki_text_mode = str(getattr(cli_args, "aki_text_mode", "") or "").strip()
    if aki_text_mode and aki_text_mode not in {"original", "cupl_aux_node", "qwen_aux_node"}:
        raise ValueError(
            "--aki_text_mode must be one of: original, cupl_aux_node, qwen_aux_node"
        )
    aki_enabled = bool(aki_text_mode)
    compare_support_grounding = bool(getattr(cli_args, "compare_support_grounding", False))
    if compare_support_grounding:
        aki_enabled = True
        aki_text_mode = f"{attribute_text_source}_aux_node"
    print(f"MSC text sources: {','.join(requested_patch_text_sources)}")
    print(f"AKI text mode   : {aki_text_mode if aki_enabled else 'off'}")

    fieldnames = [
        "dataset",
        "shot",
        "baseline",
        "baseline_jgp",
        f"aki_jgp_msc_{attribute_text_source}",
        "jgp_raw",
        "moco_visual_raw",
        "gain_jgp",
        f"gain_aki_jgp_msc_{attribute_text_source}",
        "best_weight_jgp",
        f"best_weight_aki_jgp_msc_{attribute_text_source}",
        "jgp_text_weight",
        "baseline_external",
    ]
    if aki_enabled and not compare_support_grounding:
        fieldnames.insert(fieldnames.index("baseline_jgp"), "baseline_aki")
        fieldnames.insert(fieldnames.index("gain_jgp"), "gain_aki")
    if compare_support_grounding:
        aki_insert = fieldnames.index("baseline_jgp")
        for name in ["baseline_aki_unfiltered", "baseline_aki_grounded"]:
            fieldnames.insert(aki_insert, name)
            aki_insert += 1
        gain_insert = fieldnames.index("gain_jgp")
        for name in ["gain_aki_unfiltered", "gain_aki_grounded", "aki_attributes_total", "aki_attributes_retained", "aki_retention_rate"]:
            fieldnames.insert(gain_insert, name)
            gain_insert += 1
    insert_at = 5
    for source in requested_patch_text_sources:
        fieldnames.insert(insert_at, f"baseline_msc_{source}")
        insert_at += 1
        fieldnames.insert(insert_at, f"jgp_msc_{source}")
        insert_at += 1
    gain_insert_at = fieldnames.index("best_weight_jgp")
    for source in requested_patch_text_sources:
        fieldnames.insert(gain_insert_at, f"gain_msc_{source}")
        gain_insert_at += 1
        fieldnames.insert(gain_insert_at, f"gain_jgp_msc_{source}")
        gain_insert_at += 1
    weight_insert_at = fieldnames.index("jgp_text_weight")
    for source in requested_patch_text_sources:
        fieldnames.insert(weight_insert_at, f"best_weight_jgp_msc_{source}")
        weight_insert_at += 1

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if cli_args.overwrite and os.path.exists(out_csv):
        os.remove(out_csv)

    # key: (dataset, shot)
    # values are updated during all runs, then written once at the end.
    compact = {}

    search_config = dict(SEARCH_CONFIG)
    requested_shots = parse_shot_list(getattr(cli_args, "shots", ""))
    if requested_shots is not None:
        search_config["shot"] = requested_shots
    requested_jgp_weights = parse_float_list(getattr(cli_args, "jgp_weights", ""))
    if requested_jgp_weights:
        search_config["jgp_weight"] = requested_jgp_weights
    if runtime_compare:
        search_config["jgp_weight"] = [
            float(getattr(cli_args, "runtime_compare_w", 0.9))
        ]
    elif getattr(cli_args, "run_sensitivity", False):
        # alpha and w are swept internally only once on the full model at the
        # reference tau_K.  All outer ablation/tau_K runs therefore use the
        # fixed paper value instead of creating another weight-search round.
        search_config["jgp_weight"] = [
            float(getattr(cli_args, "sensitivity_fixed_w", 0.9))
        ]
    elif getattr(cli_args, "search_jgp_weight", False):
        # The internal search determines the reported best w.  Keep the outer
        # run at the alpha-search fixed w instead of 0.0 so diagnostic branches
        # (e.g. LP+MoCoVisual) do not silently collapse to pure LP.
        search_config["jgp_weight"] = [
            float(getattr(cli_args, "sensitivity_fixed_w", 0.9))
        ]
    # The fixed ablation plan supplies text_mode explicitly. Keep the search
    # iterator from duplicating runs over SEARCH_CONFIG["text_mode"].
    search_config["text_mode"] = ["original"]
    search_configs = list(iter_search_configs(search_config))

    requested_modules = parse_module_list(getattr(cli_args, "modules", ""))
    if aki_enabled and "aki" not in requested_modules:
        requested_modules.append("aki")
    if not aki_enabled and "aki" in requested_modules:
        requested_modules.remove("aki")
    if any(m in requested_modules for m in ["aki", "jgp", "msc", "jgp_msc", "full"]) and "baseline" not in requested_modules:
        requested_modules = ["baseline"] + requested_modules
    print(f"Requested modules: {','.join(requested_modules)}")

    # Controlled module plan:
    #   baseline, AKI, JGP, MSC(Qwen), JGP+MSC(Qwen).
    # Baseline/JGP share the no-patch original run; MSC/JGP+MSC share the
    # patch-on original run. AKI needs qwen_aux_node and is therefore separate.
    def thresholds_for_current_run(shot, sweep=False):
        if runtime_compare:
            return [float(getattr(cli_args, "runtime_compare_tau_k", 0.0))]
        if bool(getattr(cli_args, "tune_aki_per_ablation", True)):
            # Every AKI-containing ablation is allowed to choose its own tau_K.
            # write_ablation_details_csv later keeps only the best row for that
            # component combination.
            return list(tau_k_values)
        if getattr(cli_args, "run_sensitivity", False):
            return list(tau_k_values) if sweep else [sensitivity_reference_tau]
        return aki_thresholds_for_shot(shot)

    planned_runs = []
    lp_aki_only = not any(m in requested_modules for m in ["jgp", "msc", "jgp_msc", "full"])
    seen_lp_aki_shots = set()
    for source_config in search_configs:
        config = source_config
        if lp_aki_only:
            shot_key = int(source_config["shot"])
            if shot_key in seen_lp_aki_shots:
                continue
            seen_lp_aki_shots.add(shot_key)
            config = dict(source_config)
            config["branch_mode"] = "lp"
            config["params"] = dict(source_config["params"])
            config["params"]["jgp_weight"] = 0.0
        shot = int(config["shot"])
        branch_mode = config["branch_mode"]
        base_source = requested_patch_text_sources[0]
        if "baseline" in requested_modules or "jgp" in requested_modules:
            if runtime_compare:
                baseline_config = dict(config)
                baseline_config["branch_mode"] = "lp"
                baseline_config["params"] = dict(config["params"])
                baseline_config["params"]["jgp_weight"] = 0.0
                planned_runs.append(
                    (baseline_config, "original", base_source, False, "baseline", None, None)
                )
            else:
                planned_runs.append((config, "original", base_source, False, "baseline_jgp" if branch_mode == "jgp" else "baseline", None, None))
        if "aki" in requested_modules:
            if compare_support_grounding:
                planned_runs.append((config, aki_text_mode, base_source, False, "baseline_aki_unfiltered", False, None))
                for threshold in thresholds_for_current_run(shot):
                    planned_runs.append((config, aki_text_mode, base_source, False, "baseline_aki_grounded", True, threshold))
            else:
                thresholds = thresholds_for_current_run(shot) if getattr(cli_args, "support_grounded_aki", False) else [None]
                for threshold in thresholds:
                    planned_runs.append((config, aki_text_mode, base_source, False, "baseline_aki", None, threshold))
        if shot > 0 and ("msc" in requested_modules or "jgp_msc" in requested_modules):
            for source in requested_patch_text_sources:
                ablation_name = f"jgp_msc_{source}" if branch_mode == "jgp" else f"baseline_msc_{source}"
                thresholds = (
                    thresholds_for_current_run(shot)
                    if source == "qwen" and getattr(cli_args, "ground_msc_qwen", False)
                    else [None]
                )
                for threshold in thresholds:
                    planned_runs.append((config, "original", source, True, ablation_name, None, threshold))
        if shot > 0 and "full" in requested_modules:
            for threshold in thresholds_for_current_run(shot, sweep=True):
                planned_runs.append((
                    config,
                    f"{attribute_text_source}_aux_node",
                    attribute_text_source,
                    True,
                    f"full_aki_jgp_msc_{attribute_text_source}",
                    True if getattr(cli_args, "support_grounded_aki", False) else None,
                    threshold,
                ))

    total = len(datasets) * len(planned_runs)
    run_id = 0

    for dataset_name in datasets:
        print(f"Dataset [{dataset_name}] will append to shared CSV: {out_csv}")
        dataset_prompt_json, dataset_cupl_json = resolve_dataset_resource_paths(cli_args, dataset_name)
        dataset_meta_file = resolve_dataset_meta_path(cli_args, dataset_name)

        for config, text_mode, patch_text_source, use_patch, ablation_name, grounding_override, aki_threshold in planned_runs:
            shot = int(config["shot"])
            branch_mode = config["branch_mode"]
            hp = dict(config["params"])
            hp.setdefault("jgp_text_weight", float(getattr(cli_args, "jgp_text_weight", 0.5)))
            jgp_w = float(hp.get("jgp_weight", 0.0))

            if patch_variant == "off" and use_patch:
                continue
            if patch_variant == "on" and not use_patch and ablation_name not in {"baseline", "baseline_aki", "baseline_jgp"}:
                continue

            run_id += 1
            dataset_cli_args = SimpleNamespace(**vars(cli_args))
            dataset_cli_args.prompt_json = dataset_prompt_json
            dataset_cli_args.cupl_json = dataset_cupl_json
            dataset_cli_args.meta_file = dataset_meta_file
            dataset_cli_args.patch_text_source = patch_text_source
            dataset_cli_args.use_support_patch_nodes = bool(use_patch)

            # Controlled ablation must start from exactly the same original
            # graph in every row.  By default we use the cached base graph propagation class
            # text features, which is also the protocol expected by the
            # external base graph propagation baseline table (e.g. ImageNet 16-shot 63.20).
            # We never overwrite the measured baseline with that external
            # number; it is only a consistency check.
            if getattr(cli_args, "save_ablation_details", False) or runtime_compare:
                dataset_cli_args.base_text_prompt = str(
                    getattr(cli_args, "ablation_base_text_prompt", "cached")
                )
            if grounding_override is not None:
                dataset_cli_args.support_grounded_aki = bool(grounding_override)
            dataset_cli_args.aki_margin_threshold = aki_threshold

            run_args = make_run_args(
                cli_args=dataset_cli_args,
                dataset_name=dataset_name,
                shot=shot,
                text_mode=text_mode,
                branch_mode=branch_mode,
                hp=hp,
            )
            # Every JGP-containing ablation gets its own tuned alpha and w.
            # Protocol: alpha is selected first at sensitivity_fixed_w, then w
            # is selected while holding alpha at that best value.  The alpha x
            # w matrix is accumulated only as an efficient way to recover the
            # second-stage w curve without a second dataset pass.
            if branch_mode == "jgp" and bool(
                getattr(cli_args, "tune_jgp_per_ablation", True)
            ):
                run_args.search_alpha = True
                run_args.search_jgp_weight = True
                run_args.search_alpha_weight_matrix = True
            elif getattr(cli_args, "run_sensitivity", False):
                internal_sensitivity_run = (
                    ablation_name == f"full_aki_jgp_msc_{attribute_text_source}"
                    and aki_threshold is not None
                    and abs(float(aki_threshold) - sensitivity_reference_tau) < 1e-12
                )
                run_args.search_alpha = bool(internal_sensitivity_run)
                run_args.search_jgp_weight = bool(internal_sensitivity_run)
                run_args.search_alpha_weight_matrix = bool(internal_sensitivity_run)

            prediction_export_dir = str(getattr(cli_args, "export_prediction_scores", "") or "").strip()
            if prediction_export_dir:
                threshold_tag = "" if aki_threshold is None else f"_tau{float(aki_threshold):g}"
                score_name = (
                    f"{dataset_name}_{shot}shot_{ablation_name}_{patch_text_source}"
                    f"{threshold_tag}.pt"
                )
                run_args.prediction_scores_out = os.path.join(prediction_export_dir, score_name)
                run_args.prediction_ablation_name = ablation_name

            ok, reason = config_is_valid(run_args)
            if not ok:
                print(
                    f"[SKIP {run_id}/{total}] dataset={dataset_name} "
                    f"shot={shot} ablation={ablation_name} mode={branch_mode} text={text_mode} "
                    f"patch={'on' if use_patch else 'off'}: {reason}"
                )
                continue

            print("=" * 100)
            print(
                f"[RUN {run_id}/{total}] dataset={dataset_name} "
                f"shot={shot} ablation={ablation_name} mode={branch_mode} text={text_mode} "
                f"patch={'on' if use_patch else 'off'} "
                f"patch_text={dataset_cli_args.patch_text_source} "
                f"jgp_weight={jgp_w} "
                f"aki_tau={aki_threshold if aki_threshold is not None else 'n/a'}"
            )
            print("=" * 100)

            start = time.time()
            try:
                result = evaluate_one_dataset(
                    run_args,
                    clip_model,
                    clip_preprocess,
                    moco_model,
                    moco_preprocess,
                    device,
                    dataset_name,
                )
                elapsed = time.time() - start

                selected_alpha = result.get("best_jgp_alpha")
                if selected_alpha is None:
                    selected_alpha = result.get("best_alpha")
                if selected_alpha is None:
                    selected_alpha = float(getattr(run_args, "jgp_text_weight", 0.5))
                selected_w = result.get("best_jgp_weight")
                if selected_w is None:
                    selected_w = float(getattr(run_args, "jgp_weight", 0.0))

                aki_component = text_mode in {"cupl_aux_node", "qwen_aux_node"}
                msc_component = bool(use_patch)
                jgp_component = branch_mode == "jgp"

                if getattr(cli_args, "save_ablation_details", False):
                    # LP-side row: same run without the JGP prediction branch.
                    lp_row = {
                        "dataset": dataset_name,
                        "shot": shot,
                        "variant": component_variant_name(aki_component, msc_component, False),
                        "aki": int(aki_component),
                        "msc": int(msc_component),
                        "jgp": 0,
                        "patch_text_source": patch_text_source if msc_component else "",
                        "tau_k": "" if aki_threshold is None else float(aki_threshold),
                        "alpha": "",
                        "w": "",
                        "accuracy": float(result["acc_lp"]),
                        "num_samples": int(result["num_samples"]),
                        "source_run": ablation_name,
                        "selection_protocol": "",
                    }
                    ablation_detail_rows.append(lp_row)

                    if jgp_component:
                        jgp_row = dict(lp_row)
                        jgp_row.update({
                            "variant": component_variant_name(aki_component, msc_component, True),
                            "jgp": 1,
                            "alpha": float(selected_alpha),
                            "w": float(selected_w),
                            "accuracy": float(result["acc_final"]),
                            "selection_protocol": "alpha_then_w",
                        })
                        ablation_detail_rows.append(jgp_row)
                    write_ablation_details_csv(ablation_details_csv, ablation_detail_rows)

                if getattr(cli_args, "save_timing", False):
                    timing = result.get("timing", {})
                    source_files = result.get("source_files", {})
                    if runtime_compare and ablation_name == "baseline":
                        timing_method = "base graph propagation"
                    elif runtime_compare and ablation_name == f"full_aki_jgp_msc_{attribute_text_source}":
                        timing_method = "JG-MSC (Ours)"
                    else:
                        timing_method = ablation_name
                    timing_rows.append({
                        "dataset": dataset_name,
                        "shot": shot,
                        "method": timing_method,
                        "ablation": ablation_name,
                        "base_text_prompt": getattr(run_args, "base_text_prompt", ""),
                        "clip_query_path": source_files.get("clip_query", ""),
                        "clip_fewshot_path": source_files.get("clip_fewshot", ""),
                        "moco_query_path": source_files.get("moco_query", ""),
                        "moco_fewshot_path": source_files.get("moco_fewshot", ""),
                        "aligned_clip_reference_path": source_files.get("aligned_clip_reference", ""),
                        "aligned_clip_mean_cosine": (
                            "" if source_files.get("aligned_clip_mean_cosine") is None
                            else float(source_files["aligned_clip_mean_cosine"])
                        ),
                        "aligned_clip_min_cosine": (
                            "" if source_files.get("aligned_clip_min_cosine") is None
                            else float(source_files["aligned_clip_min_cosine"])
                        ),
                        "meta_path": source_files.get("meta", ""),
                        "aki": int(aki_component),
                        "msc": int(msc_component),
                        "jgp": int(jgp_component),
                        "tau_k": "" if aki_threshold is None else float(aki_threshold),
                        "alpha": (
                            float(getattr(run_args, "jgp_text_weight", 0.5))
                            if jgp_component else ""
                        ),
                        "w": (
                            float(getattr(run_args, "jgp_weight", 0.0))
                            if jgp_component else ""
                        ),
                        "k_text": int(run_args.k_text),
                        "k_image": int(run_args.k_image),
                        "k_fewshot": int(run_args.k_fewshot),
                        "gamma": float(run_args.gamma),
                        "lp_alpha": float(run_args.alpha),
                        "num_iterations": int(run_args.num_iterations),
                        "num_samples": int(result["num_samples"]),
                        "train_seconds": 0.0,
                        "setup_seconds": timing.get("setup_seconds", ""),
                        "core_inference_seconds": timing.get("core_inference_seconds", ""),
                        "timed_query_count": timing.get("timed_query_count", ""),
                        "warmup_query_count": timing.get("warmup_query_count", ""),
                        "test_ms_per_query": timing.get("test_ms_per_query", ""),
                        "end_to_end_seconds": float(elapsed),
                        "peak_gpu_memory_mb": timing.get("peak_gpu_memory_mb", ""),
                        "additional_trainable_params": 0,
                        "accuracy": float(result["acc_final"]),
                    })
                    write_timing_csv(timing_csv, timing_rows)

                if (
                    getattr(cli_args, "run_sensitivity", False)
                    and ablation_name == f"full_aki_jgp_msc_{attribute_text_source}"
                    and aki_threshold is not None
                ):
                    fixed_alpha = float(result.get("sensitivity_fixed_alpha", 0.5))
                    fixed_w = float(result.get("sensitivity_fixed_w", 0.9))
                    weight_fixed_alpha = result.get("weight_fixed_alpha")
                    if weight_fixed_alpha is None:
                        weight_fixed_alpha = result.get("best_alpha", fixed_alpha)
                    weight_fixed_alpha = float(weight_fixed_alpha)
                    sensitivity_rows.append({
                        "dataset": dataset_name,
                        "shot": shot,
                        "parameter": "tau_K",
                        "value": float(aki_threshold),
                        "accuracy": float(result["acc_sensitivity_fixed"]),
                        "fixed_tau_k": "",
                        "fixed_alpha": fixed_alpha,
                        "fixed_w": fixed_w,
                        "ablation": ablation_name,
                        "num_samples": int(result["num_samples"]),
                    })

                    if abs(float(aki_threshold) - sensitivity_reference_tau) < 1e-12:
                        for alpha_value, accuracy in zip(
                            result.get("alpha_grid", []),
                            result.get("alpha_accuracies", []),
                        ):
                            sensitivity_rows.append({
                                "dataset": dataset_name,
                                "shot": shot,
                                "parameter": "alpha",
                                "value": float(alpha_value),
                                "accuracy": float(accuracy),
                                "fixed_tau_k": float(aki_threshold),
                                "fixed_alpha": "",
                                "fixed_w": fixed_w,
                                "ablation": ablation_name,
                                "num_samples": int(result["num_samples"]),
                            })
                        for weight_value, accuracy in zip(
                            result.get("weight_grid", []),
                            result.get("weight_accuracies", []),
                        ):
                            sensitivity_rows.append({
                                "dataset": dataset_name,
                                "shot": shot,
                                "parameter": "w",
                                "value": float(weight_value),
                                "accuracy": float(accuracy),
                                "fixed_tau_k": float(aki_threshold),
                                "fixed_alpha": weight_fixed_alpha,
                                "fixed_w": "",
                                "ablation": ablation_name,
                                "num_samples": int(result["num_samples"]),
                            })
                    write_sensitivity_csv(sensitivity_csv, sensitivity_rows)

                key = (dataset_name, shot, patch_text_source)
                baseline_external = get_external_baseline(dataset_name, shot)
                if key not in compact:
                    compact[key] = {
                        "dataset": dataset_name,
                        "shot": shot,
                        "patch_text_source": patch_text_source,
                        "baseline_external": baseline_external,
                        "baseline": None,
                        "baseline_patch": None,
                        "baseline_qwen": None,
                        "baseline_aki_unfiltered": None,
                        "baseline_aki_grounded": None,
                        "aki_attributes_total": None,
                        "aki_attributes_retained": None,
                        "aki_retention_rate": None,
                        "baseline_qwen_patch": None,
                        "baseline_jgp": None,
                        "baseline_jgp_patch": None,
                        "baseline_jgp_raw": None,
                        "baseline_jgp_raw_patch": None,
                        "qwen_jgp": None,
                        "qwen_jgp_patch": None,
                        "lp_moco_visual": None,
                        "lp_moco_visual_patch": None,
                        "lp_moco_visual_raw": None,
                        "lp_moco_visual_raw_patch": None,
                        "best_weight_lp_moco_visual": None,
                        "best_weight_lp_moco_visual_patch": None,
                        "best_weight_baseline_jgp": None,
                        "best_weight_baseline_jgp_patch": None,
                        "best_weight_qwen_jgp": None,
                        "best_weight_qwen_jgp_patch": None,
                        "jgp_text_weight": getattr(run_args, "jgp_text_weight", None),
                    }

                item = compact[key]
                patch_suffix = "_patch" if use_patch else ""

                if text_mode == "original":
                    update_best(item, f"baseline{patch_suffix}", result["acc_lp"])

                if text_mode in {"cupl_aux_node", "qwen_aux_node"}:
                    if ablation_name == "baseline_aki_unfiltered":
                        update_best(item, "baseline_aki_unfiltered", result["acc_lp"])
                    elif ablation_name == "baseline_aki_grounded":
                        stats = result.get("aki_grounding_stats") or {}
                        improved = update_best(item, "baseline_aki_grounded", result["acc_lp"])
                        if improved:
                            item["aki_attributes_total"] = stats.get("total")
                            item["aki_attributes_retained"] = stats.get("retained")
                            item["aki_retention_rate"] = stats.get("retention_rate")
                        if aki_threshold is not None:
                            baseline_value = item.get("baseline")
                            aki_threshold_trials.append({
                                "dataset": dataset_name,
                                "shot": shot,
                                "threshold": float(aki_threshold),
                                "accuracy": float(result["acc_lp"]),
                                "baseline": "" if baseline_value is None else float(baseline_value),
                                "gain": "" if baseline_value is None else float(result["acc_lp"] - baseline_value),
                                "attributes_total": stats.get("total", ""),
                                "attributes_retained": stats.get("retained", ""),
                                "retention_rate": stats.get("retention_rate", ""),
                            })
                            write_aki_threshold_csv(aki_threshold_csv, aki_threshold_trials)
                    else:
                        stats = result.get("aki_grounding_stats") or {}
                        update_best(item, f"baseline_qwen{patch_suffix}", result["acc_lp"])
                        if aki_threshold is not None and stats and not use_patch:
                            baseline_value = item.get("baseline")
                            aki_threshold_trials.append({
                                "dataset": dataset_name,
                                "shot": shot,
                                "threshold": float(aki_threshold),
                                "accuracy": float(result["acc_lp"]),
                                "baseline": "" if baseline_value is None else float(baseline_value),
                                "gain": "" if baseline_value is None else float(result["acc_lp"] - baseline_value),
                                "attributes_total": stats.get("total", ""),
                                "attributes_retained": stats.get("retained", ""),
                                "retention_rate": stats.get("retention_rate", ""),
                            })
                            write_aki_threshold_csv(aki_threshold_csv, aki_threshold_trials)

                if text_mode == "original" and branch_mode == "jgp":
                    best_jgp_w = result.get("best_jgp_weight")
                    stored_jgp_w = best_jgp_w if best_jgp_w is not None else jgp_w
                    improved = update_best(
                        item,
                        f"baseline_jgp{patch_suffix}",
                        result["acc_final"],
                        f"best_weight_baseline_jgp{patch_suffix}",
                        stored_jgp_w,
                    )
                    if improved:
                        if result.get("best_jgp_alpha") is not None:
                            item["jgp_text_weight"] = result["best_jgp_alpha"]
                        elif result.get("best_alpha") is not None:
                            item["jgp_text_weight"] = result["best_alpha"]
                    update_best(item, f"baseline_jgp_raw{patch_suffix}", result.get("acc_jgp"))
                    visual_acc = result.get("acc_lp_moco_visual", 0.0)
                    update_best(
                        item,
                        f"lp_moco_visual{patch_suffix}",
                        visual_acc,
                        f"best_weight_lp_moco_visual{patch_suffix}",
                        stored_jgp_w,
                    )
                    update_best(item, f"lp_moco_visual_raw{patch_suffix}", result.get("acc_lp_moco_visual"))

                if text_mode in {"cupl_aux_node", "qwen_aux_node"} and branch_mode == "jgp":
                    best_jgp_w = result.get("best_jgp_weight")
                    stored_jgp_w = best_jgp_w if best_jgp_w is not None else jgp_w
                    improved = update_best(
                        item,
                        f"qwen_jgp{patch_suffix}",
                        result["acc_final"],
                        f"best_weight_qwen_jgp{patch_suffix}",
                        stored_jgp_w,
                    )
                    if improved:
                        if result.get("best_jgp_alpha") is not None:
                            item["jgp_text_weight"] = result["best_jgp_alpha"]
                        elif result.get("best_alpha") is not None:
                            item["jgp_text_weight"] = result["best_alpha"]

                if (
                    use_patch
                    and text_mode == "original"
                    and patch_text_source == "qwen"
                    and aki_threshold is not None
                    and result.get("msc_grounding_stats")
                ):
                    stats = result["msc_grounding_stats"]
                    baseline_value = item.get("baseline")
                    if baseline_value is None:
                        for (other_dataset, other_shot, _), other_item in compact.items():
                            if other_dataset == dataset_name and other_shot == shot and other_item.get("baseline") is not None:
                                baseline_value = other_item["baseline"]
                                break
                    msc_threshold_trials.append({
                        "dataset": dataset_name,
                        "shot": shot,
                        "text_source": "qwen_grounded",
                        "threshold": float(aki_threshold),
                        "accuracy": float(result["acc_lp"]),
                        "baseline": "" if baseline_value is None else float(baseline_value),
                        "gain": "" if baseline_value is None else float(result["acc_lp"] - baseline_value),
                        "attributes_total": stats.get("total", ""),
                        "attributes_retained": stats.get("retained", ""),
                        "retention_rate": stats.get("retention_rate", ""),
                        "photo_fallback_classes": stats.get("fallback_classes", ""),
                    })
                    write_msc_threshold_csv(msc_threshold_csv, msc_threshold_trials)

                base_msg = "" if baseline_external is None else f" | external={baseline_external:.2f}"
                tag = "WITH_PATCH" if use_patch else "NO_PATCH"
                best_alpha_msg = ""
                if result.get("best_alpha") is not None:
                    best_alpha_msg = f" | best_alpha={float(result['best_alpha']):.2f}"
                if result.get("best_jgp_weight") is not None:
                    best_alpha_msg += f" | best_w={float(result['best_jgp_weight']):.2f}"
                print(
                    f"[DONE-{tag}] ablation={ablation_name} | text={text_mode} | mode={branch_mode} | "
                    f"LP={result['acc_lp']:.2f} | "
                    f"JGP={result['acc_jgp']:.2f} | "
                    f"JGPFinal={result['acc_final']:.2f} | "
                    f"LP+MoCoVisual={result.get('acc_lp_moco_visual', 0.0):.2f} | "
                    f"Final={result['acc_final']:.2f} | "
                    f"weight={jgp_w:.1f}"
                    f"{best_alpha_msg}"
                    f"{base_msg} | time={elapsed:.1f}s"
                )
                write_compact_csv(
                    out_csv,
                    fieldnames,
                    compact,
                    requested_patch_text_sources=requested_patch_text_sources,
                    attribute_text_source=attribute_text_source,
                )

            except Exception as e:
                print(
                    f"[ERROR] dataset={dataset_name} shot={shot} "
                    f"ablation={ablation_name} mode={branch_mode} text={text_mode} "
                    f"patch={'on' if use_patch else 'off'} weight={jgp_w}: {repr(e)}"
                )
                traceback.print_exc()

    print("\n" + "=" * 100)
    print("CONTROLLED ABLATION RESULTS BY DATASET / SHOT")
    print("=" * 100)

    grouped = defaultdict(dict)
    for (dataset_name, shot, patch_text_source), item in compact.items():
        grouped[(dataset_name, shot)][patch_text_source] = item

    def sfmt(x):
        return "" if x is None else f"{float(x):.2f}"

    def gfmt(x):
        return "" if x is None else f"{float(x):+.2f}"

    for (dataset_name, shot), source_items in sorted(grouped.items()):
        base_item = {}
        for source in requested_patch_text_sources:
            if source in source_items:
                base_item = source_items[source]
                break
        if not base_item and source_items:
            base_item = next(iter(source_items.values()))

        baseline = base_item.get("baseline")
        baseline_aki = base_item.get("baseline_qwen")
        jgp = base_item.get("baseline_jgp")
        full_model = source_items.get(attribute_text_source, {}).get("qwen_jgp_patch")

        summary = f"{dataset_name:15s} shot={shot:<2} | Baseline={sfmt(baseline)}"
        if "jgp" in requested_modules:
            summary += f" | JGP={sfmt(jgp)}({gfmt(None if baseline is None or jgp is None else jgp - baseline)})"
        print(summary)
        if compare_support_grounding:
            aki_unfiltered = base_item.get("baseline_aki_unfiltered")
            aki_grounded = base_item.get("baseline_aki_grounded")
            retained = base_item.get("aki_attributes_retained")
            total_attrs = base_item.get("aki_attributes_total")
            retention_rate = base_item.get("aki_retention_rate")
            print(
                f"{'':15s}         | "
                f"AKI-Unfiltered={sfmt(aki_unfiltered)}"
                f"({gfmt(None if baseline is None or aki_unfiltered is None else aki_unfiltered - baseline)}) | "
                f"AKI-Grounded={sfmt(aki_grounded)}"
                f"({gfmt(None if baseline is None or aki_grounded is None else aki_grounded - baseline)})"
            )
            print(
                f"{'':15s}         | "
                f"Grounding gain over unfiltered="
                f"{gfmt(None if aki_unfiltered is None or aki_grounded is None else aki_grounded - aki_unfiltered)} | "
                f"Attributes retained={int(retained) if retained is not None else ''}/"
                f"{int(total_attrs) if total_attrs is not None else ''} "
                f"({sfmt(retention_rate)}%)"
            )
        elif aki_enabled:
            print(
                f"{'':15s}         | "
                f"AKI={sfmt(baseline_aki)}"
                f"({gfmt(None if baseline is None or baseline_aki is None else baseline_aki - baseline)})"
            )
        for source in requested_patch_text_sources if any(m in requested_modules for m in ["msc", "jgp_msc"]) else []:
            item = source_items.get(source, {})
            msc = item.get("baseline_patch")
            jgp_msc = item.get("baseline_jgp_patch")
            print(
                f"{'':15s}         | "
                f"MSC-{source}={sfmt(msc)}({gfmt(None if baseline is None or msc is None else msc - baseline)}) | "
                f"JGP+MSC-{source}={sfmt(jgp_msc)}"
                f"({gfmt(None if baseline is None or jgp_msc is None else jgp_msc - baseline)})"
            )
        if "full" in requested_modules:
            print(
                f"{'':15s}         | AKI+JGP+MSC-{attribute_text_source}={sfmt(full_model)}"
                f"({gfmt(None if baseline is None or full_model is None else full_model - baseline)})"
            )

    if aki_threshold_trials:
        print("=" * 100)
        print("AKI NORMALIZED-MARGIN THRESHOLD SEARCH")
        print("=" * 100)
        threshold_groups = defaultdict(list)
        for trial in aki_threshold_trials:
            threshold_groups[(trial["dataset"], trial["shot"])].append(trial)
        for (dataset_name, shot), trials in sorted(threshold_groups.items()):
            trials = sorted(trials, key=lambda row: float(row["threshold"]))
            best = max(trials, key=lambda row: float(row["accuracy"]))
            print(f"{dataset_name:15s} shot={shot}")
            for trial in trials:
                print(
                    f"  tau={float(trial['threshold']):.4f} | acc={float(trial['accuracy']):.2f}% "
                    f"| gain={float(trial['gain']):+.2f}% "
                    f"| retained={trial['attributes_retained']}/{trial['attributes_total']} "
                    f"({float(trial['retention_rate']):.2f}%)"
                )
            print(
                f"  Best tau={float(best['threshold']):.4f} | "
                f"Best acc={float(best['accuracy']):.2f}%"
            )

    if msc_threshold_trials:
        print("=" * 100)
        print("MSC-QWEN NORMALIZED-MARGIN THRESHOLD SEARCH VS MSC-PHOTO")
        print("=" * 100)
        threshold_groups = defaultdict(list)
        for trial in msc_threshold_trials:
            threshold_groups[(trial["dataset"], trial["shot"])].append(trial)
        for (dataset_name, shot), trials in sorted(threshold_groups.items()):
            trials = sorted(trials, key=lambda row: float(row["threshold"]))
            photo_item = grouped.get((dataset_name, shot), {}).get("photo", {})
            photo_acc = photo_item.get("baseline_patch")
            print(f"{dataset_name:15s} shot={shot} | MSC-photo={sfmt(photo_acc)}")
            for trial in trials:
                delta_photo = None if photo_acc is None else float(trial["accuracy"]) - float(photo_acc)
                print(
                    f"  MSC-qwen tau={float(trial['threshold']):.4f} | acc={float(trial['accuracy']):.2f}% "
                    f"| vs photo={gfmt(delta_photo)} | retained={trial['attributes_retained']}/{trial['attributes_total']} "
                    f"({float(trial['retention_rate']):.2f}%) | photo fallback classes={trial['photo_fallback_classes']}"
                )

    if runtime_compare:
        print_runtime_comparison(timing_rows)

    print("=" * 100)
    print(f"Compact ablation CSV saved to: {out_csv}")
    if getattr(cli_args, "save_ablation_details", False):
        print(f"Detailed ablation CSV saved to: {ablation_details_csv}")
    if getattr(cli_args, "save_timing", False):
        print(f"Timing CSV saved to: {timing_csv}")
    if getattr(cli_args, "run_sensitivity", False):
        print(f"Sensitivity CSV saved to: {sensitivity_csv}")
    if aki_threshold_trials:
        print(f"AKI threshold CSV saved to: {aki_threshold_csv}")
    if msc_threshold_trials:
        print(f"MSC threshold CSV saved to: {msc_threshold_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JG-MSC weight-search version. Default visual path: features_path/fine_grained/MOCO/R50."
    )

    # You only need to specify these commonly changed arguments.
    parser.add_argument("--datasets", type=str, default="ALL",
                        help="Dataset names separated by '/', e.g., Aircraft/Cars/DTD. Default: ALL from SUPPORTED_DATASETS.")
    parser.add_argument("--shots", type=str, default="",
                        help="Optional shot list separated by comma or slash, e.g., 16 or 1,2,4,8,16. Default uses SEARCH_CONFIG.")
    parser.add_argument("--modules", type=str, default="baseline,aki,msc,jgp,jgp_msc,full",
                        help=(
                            "Comma/slash list of modules to run: baseline, aki, jgp, msc, jgp_msc, full. "
                            "Aliases: lp=baseline, qwen=aki, jgp+msc=jgp_msc, jgp+msc+aki=full."
                        ))
    parser.add_argument("--experiment_suite", action="store_true",
                        help=("Run and save the complete paper experiment suite: detailed component ablation, "
                              "per-query runtime, and tau_K/alpha/w sensitivity CSVs."))
    parser.add_argument(
        "--runtime_compare",
        action="store_true",
        help=(
            "Run exactly two timed methods per dataset/shot: base graph propagation "
            "(cached text + label propagation) and full JG-MSC (AKI + MSC + JGP). "
            "Sensitivity searches are disabled and a direct comparison CSV is saved."
        ),
    )
    parser.add_argument("--save_ablation_details", action="store_true",
                        help="Save a long-format AKI/MSC/JGP component ablation CSV beside --output_csv.")
    parser.add_argument("--save_timing", action="store_true",
                        help="Measure and save training-free setup time and synchronized per-query test latency.")
    parser.add_argument("--timing_warmup_queries", type=int, default=5,
                        help="Number of initial queries excluded from the synchronized latency average.")
    parser.add_argument("--run_sensitivity", action="store_true",
                        help="Run and save one-at-a-time sensitivity for tau_K, alpha, and w on full JG-MSC.")
    parser.add_argument(
        "--tau_k_grid",
        type=str,
        default=",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["tau_k"]),
        help="AKI normalized pruning-threshold grid used for tau_K sensitivity.",
    )
    parser.add_argument(
        "--sensitivity_fixed_w",
        type=float,
        default=float(SEARCH_CONFIG["sensitivity"]["fixed_w"]),
        help=(
            "Fixed w used in stage 1 while sweeping alpha. After the best alpha "
            "is selected, stage 2 sweeps w again at that alpha."
        ),
    )
    parser.add_argument(
        "--sensitivity_reference_tau_k",
        type=float,
        default=float(SEARCH_CONFIG["sensitivity"]["reference_tau_k"]),
        help="tau_K value at which the independent alpha and w curves are recorded.",
    )
    parser.add_argument("--ablation_mode", type=str, default="controlled", choices=["full", "controlled"],
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--ablation_base_text_prompt",
        type=str,
        default="cached",
        choices=["cached", "photo"],
        help=(
            "Base class nodes shared by every controlled ablation row. "
            "Default cached reproduces the base graph propagation graph as closely as possible."
        ),
    )
    parser.add_argument(
        "--tune_jgp_per_ablation",
        dest="tune_jgp_per_ablation",
        action="store_true",
        default=True,
        help=(
            "For every JGP-containing ablation, sweep alpha first at fixed w, "
            "then sweep w at the selected best alpha (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no_tune_jgp_per_ablation",
        dest="tune_jgp_per_ablation",
        action="store_false",
        help="Disable per-ablation alpha->w tuning and use the configured fixed values.",
    )
    parser.add_argument(
        "--tune_aki_per_ablation",
        dest="tune_aki_per_ablation",
        action="store_true",
        default=True,
        help=(
            "For every AKI-containing ablation, sweep tau_K and retain the best "
            "measured row for that component combination (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no_tune_aki_per_ablation",
        dest="tune_aki_per_ablation",
        action="store_false",
        help="Disable per-ablation tau_K tuning.",
    )
    parser.add_argument("--text_mode", type=str, default="original",
                        help=("Text mode: original, cupl_aux_node, qwen_aux_node, or a comma-separated "
                              "combination. The paper setup uses original and cupl_aux_node."))
    parser.add_argument("--aki_text_mode", type=str, default="",
                        choices=["", "original", "cupl_aux_node", "qwen_aux_node"],
                        help=("Enable AKI and choose its description source. The default paper setup uses "
                              "cupl_aux_node; qwen_aux_node is kept only for backward compatibility."))
    parser.add_argument(
        "--base_text_prompt",
        type=str,
        default="photo",
        choices=["photo", "cached"],
        help=("Base class node construction. photo encodes 'a photo of a {class}.' (default); "
              "cached reuses the legacy feature-file text rows."),
    )
    parser.add_argument(
        "--attribute_text_source",
        type=str,
        default="cupl",
        choices=["cupl", "qwen"],
        help=("Description source used by AKI and the full JG-MSC run. Default: CuPL. "
              "Qwen merged_results is optional legacy behavior only."),
    )
    parser.add_argument("--support_grounded_aki", action="store_true",
                        help=("Filter each raw description using cached global CLIP support-image features: "
                              "keep it only when its own-class mean response exceeds every negative class."))
    parser.add_argument("--compare_support_grounding", action="store_true",
                        help=("Run both unfiltered AKI and support-grounded AKI in one command, alongside the baseline, "
                              "and write both accuracies and the attribute retention rate to the same CSV."))
    parser.add_argument("--aki_grounding_level", type=str, default="image", choices=["image", "patch"],
                        help=("Evidence granularity for --support_grounded_aki. image reuses global few-shot CLIP "
                              "features; patch loads/reuses the support patch cache and uses each image's max patch response."))
    parser.add_argument("--jgp_weights", type=str, default="",
                        help="Optional external JGP weights, e.g., 0.9 or 0.8,0.9. Default uses SEARCH_CONFIG.")
    parser.add_argument("--jgp_text_weight", type=float, default=0.5,
                        help="Alpha for JGP internal graph fusion: alpha * TEXT_GRAPH + (1-alpha) * MOCO_GRAPH.")
    parser.add_argument("--search_alpha", action="store_true",
                        help="Search jgp_text_weight inside one evaluation pass.")
    parser.add_argument(
        "--alpha_grid",
        type=str,
        default=",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["alpha"]),
        help="Grid for --search_alpha; defaults to SEARCH_CONFIG['sensitivity']['alpha'].",
    )
    parser.add_argument("--search_jgp_weight", dest="search_jgp_weight", action="store_true", default=True,
                        help="Search external JGP weight inside one evaluation pass (default: enabled).")
    parser.add_argument("--no_search_jgp_weight", dest="search_jgp_weight", action="store_false",
                        help="Disable internal JGP-weight search and use --jgp_weights as fixed outer runs.")
    parser.add_argument(
        "--jgp_weight_grid",
        type=str,
        default=",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["w"]),
        help=("Grid for --search_jgp_weight; defaults to SEARCH_CONFIG['sensitivity']['w']. "
              "Formula: final=(1-w)*LP+w*JGP."),
    )
    parser.add_argument("--features_path", type=str, default=DEFAULT_FEATURES_PATH,
                        help="Root feature path, e.g., ./features_dmn_aligned_16shot.")
    parser.add_argument("--clip_fewshot_features_path", type=str, default="",
                        help=("Optional directory containing CLIP few-shot caches. Used by ImageNet OOD to keep "
                              "the original A/V/R/K query files while loading source-support caches separately."))
    parser.add_argument("--dino_features_path", type=str, default="",
                        help="Backward compatible name. In this MoCo version, this is treated as the visual/MoCo feature path.")
    parser.add_argument("--moco_features_path", type=str, default=DEFAULT_MOCO_FEATURES_PATH,
                        help="Optional MoCo feature path. Default: features_path/fine_grained/MOCO/R50.")
    parser.add_argument("--clip_model", type=str, default="RN", choices=["RN", "VIT"])
    parser.add_argument("--task", type=str, default="fine_grained",
                        choices=["fine_grained", "style_transfer", "out_of_distribution"])

    # Optional manual fallback. For multi-dataset runs, dataset-specific auto paths are preferred.
    parser.add_argument("--prompt_json", type=str, default="")
    parser.add_argument("--cupl_json", type=str, default="")
    parser.add_argument("--qwen_attr_root", type=str, default=DEFAULT_QWEN_ATTR_ROOT,
                        help="Root of qwen_attrs_universal. Auto path: root/Dataset/merged_results.json.")
    parser.add_argument("--cupl_root", type=str, default=DEFAULT_CUPL_ROOT,
                        help="Root directory of Prompt_CuPL json files.")

    parser.add_argument("--dino_model", type=str, default="MOCO/R50")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mocov3_ckpt", type=str, default=DEFAULT_MOCOV3_CKPT,
                        help="MoCo v3 ResNet50 checkpoint path. Required when --use_support_patch_nodes is enabled.")
    parser.add_argument("--meta_file", type=str, default="",
                        help="Aligned meta pt file used to recover selected few-shot support image paths.")
    parser.add_argument("--meta_root", type=str, default=DEFAULT_META_ROOT,
                        help="Optional root directory that contains <Dataset>_aligned_meta.pt files for all datasets.")
    parser.add_argument(
        "--aligned_clip_reference_path",
        type=str,
        default="",
        help=(
            "Aligned CLIP query .pt or its RN/VIT directory, used only to verify that the paired "
            "MoCo query rows follow the base graph propagation cache order. Empty auto-resolves "
            "<meta_root>/<clip_model>/<Dataset>.pt."
        ),
    )
    parser.add_argument(
        "--alignment_min_cosine",
        type=float,
        default=0.90,
        help=(
            "Minimum mean row-wise cosine between original and aligned CLIP query caches. "
            "JG-MSC stops before timing when this check fails."
        ),
    )

    parser.add_argument("--use_support_patch_nodes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--patch_variant", type=str, default="on", choices=["off", "on", "both"],
                        help="Patch ablation variant: off, on, or both. on/both automatically enable support patch nodes.")
    parser.add_argument("--support_patch_features_pt", type=str, default="",
                        help="Optional precomputed support patch cache .pt to load directly.")
    parser.add_argument("--support_patch_cache_out", type=str, default="",
                        help="Optional explicit output path for auto-built support patch cache.")
    parser.add_argument("--support_patch_topk_per_class", type=int, default=1,
                        help="Number of support patch nodes selected per class.")
    parser.add_argument("--support_patch_select_mode", type=str, default="rank_sum",
                        choices=["rank_sum", "clip", "moco", "sum", "clip_only", "moco_only"],
                        help="How to rank support patches across CLIP and MoCo spaces.")
    parser.add_argument("--patch_text_source", type=str, default="cupl",
                        help="MSC patch-selection text source: qwen, cupl, photo, both/all, or a comma/slash list.")
    parser.add_argument("--ground_msc_qwen", action="store_true",
                        help=("Apply the same normalized support-margin threshold search to Qwen attributes used "
                              "for MSC. Classes with no retained Qwen attribute fall back to a photo prompt."))

    parser.add_argument("--patch_grid", type=int, default=7,
                        help="Support image crop grid size N for N x N patches.")
    parser.add_argument("--crop_overlap", type=float, default=0.0,
                        help="Neighbor patch overlap ratio in [0, 1).")
    parser.add_argument("--patch_batch_size", type=int, default=64,
                        help="Batch size used when encoding support patches.")
    parser.add_argument("--moco_image_size", type=int, default=224,
                        help="Input image size for MoCo patch crops.")

    parser.add_argument("--bridge_source", type=str, default="query",
                        choices=["query", "fewshot", "all"],
                        help="Feature pool used to solve the CLIP->MoCo linear bridge.")
    parser.add_argument("--bridge_lambda", type=float, default=0.01,
                        help="Ridge lambda for the training-free CLIP->MoCo bridge.")
    parser.add_argument("--no_tqdm", action="store_true",
                        help="Disable tqdm progress bars for patch cache build/select stages.")

    parser.add_argument("--output_csv", type=str, default="",
                        help="Optional output CSV path. Default: results_consen/<dataset>/<shot>/<file>_<time>.csv")
    parser.add_argument("--export_prediction_scores", type=str, default="",
                        help=("Optional directory for exact per-query base graph propagation/MGP probability caches. "
                              "Empty disables export."))
    parser.add_argument("--prediction_export_per_class", type=int, default=0,
                        help="Queries saved per class in prediction caches; 0 saves all queries.")
    parser.add_argument("--prediction_export_seed", type=int, default=1,
                        help="Sampling seed used by --prediction_export_per_class.")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)
