#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal aligned CLIP + DINO/MoCo feature extractor for ECALP with robust local path auto-detection.
Supports the 11 few-shot datasets plus ImageNet-A/V2/R/Sketch cross-domain evaluation.
Output format is compatible with ECALP:
  output_root/fine_grained/RN/Dataset.pt
  output_root/fine_grained/RN/few_shots/Dataset.pt
  output_root/fine_grained/<VISUAL_MODEL_NAME>/Dataset.pt
  output_root/fine_grained/<VISUAL_MODEL_NAME>/few_shots/Dataset.pt
"""
import os, json, argparse, random, glob
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import clip


def config_first_existing(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0] if paths else ""

# ==========================================================
# User Config: only modify this part
# ==========================================================
# Choose the datasets you want to extract.
# Example:
# RUN_DATASETS = ["Flowers102"]
# RUN_DATASETS = ["Aircraft", "Cars", "Flowers102"]
# RUN_DATASETS = ["UCF101", "ImageNet"]
RUN_DATASETS = [
    # "Aircraft",
    # "Cars",
    "Flowers102",
    "EuroSAT",
    # "Pets",
    # "Food101",
    # "DTD",
    # "Caltech101",
    # "SUN397",
    # "UCF101",
    # "ImageNet",
]

# Global extraction settings.
OUTPUT_ROOT = "./features_mocov3_aligned_16shot"
TASK = "fine_grained"
CLIP_MODEL = "RN"                   # "RN" or "VIT"
VISUAL_ENCODER = "mocov3_r50"       # choose by command line: mocov3_r50 / mocov3_vit / dino
VISUAL_MODEL_NAME = ""             # empty means auto: MOCO/R50, MOCO/VIT, or DINO
VISUAL_IMAGE_SIZE = 224

MOCO_CKPT = config_first_existing(
    "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/r-50-1000ep.pth.tar",
    "/home/chy/chy/AMU-Tuning/AMU-Tuning/r-50-1000ep.pth.tar",
)
MOCO_VIT_CKPT = "/home/chy/chy/MLLM/ECALP-master/vit-b-300ep.pth.tar"

DINO_CKPT = ""
DINO_SIZE = "b"                     # "s", "b", or "l"

NUM_SHOTS = 16
SEED = 1
BATCH_SIZE = 32
TEXT_BATCH_SIZE = 64
CPU = False
CONTINUE_ON_ERROR = True

# Base paths.
BASE_DATA_ROOT = config_first_existing(
    "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/datasets",
    "/home/chy/chy/AMU-Tuning/AMU-Tuning/datasets",
)
CUPL_ROOT = config_first_existing(
    "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/Prompt_CuPL",
    "/home/chy/chy/AMU-Tuning/AMU-Tuning/Prompt_CuPL",
)
OLD_FEATURES_ROOT = config_first_existing(
    "/home/zh/chy/chy/MLLM/ECALP-master/features",
    "/home/chy/chy/MLLM/ECALP-master/features",
)


# Dataset-specific CuPL files.
# Usually you do not need to modify this part unless your file names differ.
DATASET_CUPL_JSON = {
    "Aircraft": "fgvc.json",
    "Cars": "stanford_cars.json",
    "Flowers102": "oxford_flowers.json",
    "Pets": "oxford_pets.json",
    "Food101": "food101.json",
    "DTD": "dtd.json",
    "Caltech101": "Caltech101.json",
    "EuroSAT": "eurosat.json",
    "SUN397": "sun397.json",
    "UCF101": "ucf101.json",
    "ImageNet": "imagenet.json",
    "ImageNet-A": "imagenet.json",
    "ImageNet-V2": "imagenet.json",
    "ImageNet-R": "imagenet.json",
    "ImageNet-Sketch": "imagenet.json",
}

IMAGENET_OOD_DATASETS = ["ImageNet-A", "ImageNet-V2", "ImageNet-R", "ImageNet-Sketch"]

# Same dataset argument style as your Joint Dual Graph search script.
SUPPORTED_DATASETS = [
    "ImageNet", "Caltech101", "Flowers102", "Aircraft", "Cars",
    "DTD", "SUN397", "UCF101", "Food101", "Pets", "EuroSAT",
    *IMAGENET_OOD_DATASETS,
]
DATASET_GROUPS = {
    "ALL": [
        "ImageNet", "Caltech101", "Flowers102", "Aircraft", "Cars",
        "DTD", "SUN397", "UCF101", "Food101", "Pets", "EuroSAT",
    ],
    "IMAGENET_OOD": IMAGENET_OOD_DATASETS,
    "ALL_EXCEPT_CARS_IMAGENET": [
        "Caltech101", "Flowers102", "Aircraft", "DTD",
        "SUN397", "UCF101", "Food101", "Pets", "EuroSAT",
    ],
}

# Same CuPL candidate style as your Joint Dual Graph search script.
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
    raw = str(datasets_arg).strip()
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


def first_existing(*paths):
    """Return the first existing path. If none exists, return the first candidate."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else ""


def auto_split_json(root, *names):
    """Return the first split json that exists.

    It first checks the dataset root, then checks BASE_DATA_ROOT, and finally
    does a small recursive search. This avoids failures when files such as
    split_zhou_StanfordCars.json are placed one level higher/lower than the
    image folder.
    """
    candidates = []
    for name in names:
        candidates.extend([
            os.path.join(root, name),
            os.path.join(BASE_DATA_ROOT, name),
        ])

    for p in candidates:
        if p and os.path.exists(p):
            return p

    for name in names:
        for base in [root, BASE_DATA_ROOT]:
            if base and os.path.exists(base):
                hits = glob.glob(os.path.join(base, "**", name), recursive=True)
                if hits:
                    return hits[0]

    return os.path.join(root, names[0]) if names else ""


# Robust local dataset paths.
_CALTECH_ROOT = first_existing(f"{BASE_DATA_ROOT}/caltech-101", f"{BASE_DATA_ROOT}/Caltech101", f"{BASE_DATA_ROOT}/caltech101")
_FLOWERS_ROOT = first_existing(f"{BASE_DATA_ROOT}/102flowers", f"{BASE_DATA_ROOT}/Flowers102", f"{BASE_DATA_ROOT}/oxford_flowers")
_AIRCRAFT_ROOT = first_existing(f"{BASE_DATA_ROOT}/fgvc-aircraft-2013b/data", f"{BASE_DATA_ROOT}/fgvc-aircraft-2013b", f"{BASE_DATA_ROOT}/Aircraft")
_CARS_ROOT = first_existing(f"{BASE_DATA_ROOT}/stanford_cars", f"{BASE_DATA_ROOT}/StanfordCars", f"{BASE_DATA_ROOT}/cars", f"{BASE_DATA_ROOT}/StanfordCars-archive")
_DTD_ROOT = first_existing(f"{BASE_DATA_ROOT}/dtd", f"{BASE_DATA_ROOT}/DTD")
_SUN_ROOT = first_existing(f"{BASE_DATA_ROOT}/sun397", f"{BASE_DATA_ROOT}/SUN397", f"{BASE_DATA_ROOT}/Sun397")
_UCF_ROOT = first_existing(f"{BASE_DATA_ROOT}/ucf101", f"{BASE_DATA_ROOT}/UCF101", f"{BASE_DATA_ROOT}/UCF-101")
_FOOD_ROOT = first_existing(f"{BASE_DATA_ROOT}/food-101", f"{BASE_DATA_ROOT}/Food101", f"{BASE_DATA_ROOT}/food101")
_PETS_ROOT = first_existing(f"{BASE_DATA_ROOT}/oxford_pets", f"{BASE_DATA_ROOT}/OxfordPets", f"{BASE_DATA_ROOT}/pets", f"{BASE_DATA_ROOT}/oxford-iiit-pet", f"{BASE_DATA_ROOT}/Oxford-IIIT-Pet", f"{BASE_DATA_ROOT}/OxfordIIITPet")
_EUROSAT_ROOT = first_existing(f"{BASE_DATA_ROOT}/eurosat", f"{BASE_DATA_ROOT}/EuroSAT", f"{BASE_DATA_ROOT}/eurosat/2750", f"{BASE_DATA_ROOT}/EuroSAT/2750")
_IMAGENET_ROOT = first_existing(
    f"{BASE_DATA_ROOT}/ImageNet2012",
    f"{BASE_DATA_ROOT}/imagenet2012",
    f"{BASE_DATA_ROOT}/imagenet",
    f"{BASE_DATA_ROOT}/ImageNet",
    f"{BASE_DATA_ROOT}/ILSVRC2012",
)
_IMAGENET_A_ROOT = first_existing(
    f"{BASE_DATA_ROOT}/ImageNet-A", f"{BASE_DATA_ROOT}/imagenet-a",
    f"{BASE_DATA_ROOT}/imagenet_a",
)
_IMAGENET_V2_ROOT = first_existing(
    f"{BASE_DATA_ROOT}/imagenetv2/imagenetv2-matched-frequency-format-val",
    f"{BASE_DATA_ROOT}/imagenetv2",
    f"{BASE_DATA_ROOT}/ImageNet-V2", f"{BASE_DATA_ROOT}/imagenet-v2",
    f"{BASE_DATA_ROOT}/imagenetv2-matched-frequency-format-val",
    f"{BASE_DATA_ROOT}/ImageNetV2",
)
_IMAGENET_R_ROOT = first_existing(
    f"{BASE_DATA_ROOT}/ImageNet-R", f"{BASE_DATA_ROOT}/imagenet-r",
    f"{BASE_DATA_ROOT}/imagenet_r",
)
_IMAGENET_SKETCH_ROOT = first_existing(
    f"{BASE_DATA_ROOT}/ImageNet-Sketch", f"{BASE_DATA_ROOT}/imagenet-sketch",
    f"{BASE_DATA_ROOT}/ImageNetSketch", f"{BASE_DATA_ROOT}/sketch",
)

DATASET_LOCAL_PATHS = {
    "ImageNet": {
        "root": _IMAGENET_ROOT,
        "image_dir": first_existing(os.path.join(_IMAGENET_ROOT, "images"), _IMAGENET_ROOT),
        "split_json": "",
        "classnames_txt": os.path.join(_IMAGENET_ROOT, "classnames.txt"),
        "old_feature_name": "I.pt",
    },
    "ImageNet-A": {
        "root": _IMAGENET_A_ROOT,
        "image_dir": _IMAGENET_A_ROOT,
        "split_json": "",
        "classnames_txt": os.path.join(_IMAGENET_ROOT, "classnames.txt"),
        "old_feature_name": "A.pt",
        "source_dataset": "ImageNet",
    },
    "ImageNet-V2": {
        "root": _IMAGENET_V2_ROOT,
        "image_dir": _IMAGENET_V2_ROOT,
        "split_json": "",
        "classnames_txt": os.path.join(_IMAGENET_ROOT, "classnames.txt"),
        "old_feature_name": "V.pt",
        "source_dataset": "ImageNet",
    },
    "ImageNet-R": {
        "root": _IMAGENET_R_ROOT,
        "image_dir": _IMAGENET_R_ROOT,
        "split_json": "",
        "classnames_txt": os.path.join(_IMAGENET_ROOT, "classnames.txt"),
        "old_feature_name": "R.pt",
        "source_dataset": "ImageNet",
    },
    "ImageNet-Sketch": {
        "root": _IMAGENET_SKETCH_ROOT,
        "image_dir": _IMAGENET_SKETCH_ROOT,
        "split_json": "",
        "classnames_txt": os.path.join(_IMAGENET_ROOT, "classnames.txt"),
        "old_feature_name": "K.pt",
        "source_dataset": "ImageNet",
    },
    "Caltech101": {
        "root": _CALTECH_ROOT,
        "image_dir": _CALTECH_ROOT,
        "split_json": auto_split_json(_CALTECH_ROOT, "split_zhou_Caltech101.json"),
    },
    "Flowers102": {
        "root": _FLOWERS_ROOT,
        "image_dir": first_existing(os.path.join(_FLOWERS_ROOT, "jpg"), _FLOWERS_ROOT),
        "split_json": auto_split_json(_FLOWERS_ROOT, "split_zhou_OxfordFlowers.json"),
    },
    "Aircraft": {
        "root": _AIRCRAFT_ROOT,
        "image_dir": first_existing(os.path.join(_AIRCRAFT_ROOT, "images"), os.path.join(_AIRCRAFT_ROOT, "data/images"), _AIRCRAFT_ROOT),
        "split_json": "",
    },
    "Cars": {
        "root": _CARS_ROOT,
        "image_dir": _CARS_ROOT,
        "split_json": auto_split_json(_CARS_ROOT, "split_zhou_StanfordCars.json"),
    },
    "DTD": {
        "root": _DTD_ROOT,
        "image_dir": first_existing(os.path.join(_DTD_ROOT, "images"), _DTD_ROOT),
        "split_json": auto_split_json(_DTD_ROOT, "split_zhou_DescribableTextures.json"),
    },
    "SUN397": {
        "root": _SUN_ROOT,
        "image_dir": _SUN_ROOT,
        "split_json": auto_split_json(_SUN_ROOT, "split_zhou_SUN397.json"),
    },
    "UCF101": {
        "root": _UCF_ROOT,
        "image_dir": _UCF_ROOT,
        "split_json": auto_split_json(_UCF_ROOT, "split_zhou_UCF101.json"),
    },
    "Food101": {
        "root": _FOOD_ROOT,
        "image_dir": first_existing(os.path.join(_FOOD_ROOT, "images"), _FOOD_ROOT),
        "split_json": auto_split_json(_FOOD_ROOT, "split_zhou_Food101.json"),
    },
    "Pets": {
        "root": _PETS_ROOT,
        "image_dir": _PETS_ROOT,
        "split_json": auto_split_json(_PETS_ROOT, "split_zhou_OxfordPets.json"),
    },
    "EuroSAT": {
        "root": _EUROSAT_ROOT,
        "image_dir": _EUROSAT_ROOT,
        "split_json": auto_split_json(_EUROSAT_ROOT, "split_zhou_EuroSAT.json"),
    },
}


def mkdir(p): os.makedirs(p, exist_ok=True)

def norm(x):
    return " ".join(str(x).lower().strip().replace("_", " ").replace("-", " ").split())

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pil(path): return Image.open(path).convert("RGB")


_BASENAME_CACHE = {}

def find_by_basename(search_roots, filename):
    """Find an image by filename under possible dataset roots.

    This is a fallback for Zhou split files whose relative paths do not match
    the user's local folder layout exactly. Results are cached by root list and
    filename.
    """
    filename = os.path.basename(str(filename))
    roots = []
    for r in search_roots:
        if r and os.path.exists(r) and r not in roots:
            roots.append(r)
    key = (tuple(roots), filename)
    if key in _BASENAME_CACHE:
        return _BASENAME_CACHE[key]

    for r in roots:
        hits = glob.glob(os.path.join(r, "**", filename), recursive=True)
        if hits:
            _BASENAME_CACHE[key] = hits[0]
            return hits[0]

    _BASENAME_CACHE[key] = ""
    return ""

def ensure_2d(x, name):
    x = x.detach().float().cpu()
    if x.dim() == 3 and x.shape[1] == 1: x = x.squeeze(1)
    if x.dim() != 2: raise RuntimeError(f"{name} must be [N,D], got {tuple(x.shape)}")
    return F.normalize(x, dim=-1)

def ensure_3d(x, name):
    x = x.detach().float().cpu()
    if x.dim() == 4 and x.shape[2] == 1: x = x.squeeze(2)
    if x.dim() != 3: raise RuntimeError(f"{name} must be [C,shot,D], got {tuple(x.shape)}")
    return F.normalize(x, dim=-1)

def load_cupl(cupl_json):
    if not os.path.exists(cupl_json): raise FileNotFoundError(cupl_json)
    data = json.load(open(cupl_json, "r", encoding="utf-8"))
    classes = list(data.keys())
    mp = {}
    for i, c in enumerate(classes):
        for k in {norm(c), norm(c.replace("_", " ")), norm(c.replace(" ", "_"))}:
            mp.setdefault(k, i)
    return classes, mp, data


def load_imagenet_classnames(classnames_txt):
    """Load ImageNet class names and synset -> official index mapping.

    Expected line examples:
        n01440764 tench, Tinca tinca
        n01443537 goldfish, Carassius auratus
    """
    if not classnames_txt or not os.path.exists(classnames_txt):
        raise FileNotFoundError(f"ImageNet classnames_txt not found: {classnames_txt}")

    class_names, mapping, synset_to_idx = [], {}, {}
    with open(classnames_txt, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for idx, line in enumerate(lines):
        parts = line.replace(":", " ").split()
        synset = None
        for part in parts:
            token = part.strip().strip(",")
            if token.startswith("n") and len(token) == 9 and token[1:].isdigit():
                synset = token
                break
        if synset is None:
            synset = f"class_{idx:04d}"
            name = line
        else:
            name = line.replace(synset, "", 1).strip(" ,:\t")
        if not name:
            name = synset

        class_names.append(name)
        synset_to_idx[synset] = idx
        candidates = {
            norm(name),
            norm(name.split(",")[0]),
            norm(name.replace("_", " ")),
            norm(synset),
            synset,
        }
        for c in candidates:
            mapping.setdefault(c, idx)

    print("=" * 80)
    print("ImageNet classnames loaded")
    print(f"Classnames txt : {classnames_txt}")
    print(f"Classes loaded : {len(class_names)}")
    print(f"Synsets loaded : {len(synset_to_idx)}")
    print("=" * 80)
    return class_names, mapping, synset_to_idx


def get_old_feature_reference_path(args):
    cfg = DATASET_LOCAL_PATHS.get(args.dataset, {})
    old_feature_name = cfg.get("old_feature_name", f"{args.dataset}.pt")
    if not old_feature_name.endswith(".pt"):
        old_feature_name += ".pt"
    return os.path.join(args.old_features_root, args.task, args.clip_model, old_feature_name)


def preflight_old_feature_check(args):
    if not getattr(args, "align_to_old_order", False):
        return None
    ref_path = get_old_feature_reference_path(args)
    print("=" * 80)
    print("[Preflight] Old feature reference check")
    print(f"Dataset        : {args.dataset}")
    print(f"Reference path : {ref_path}")
    print(f"Exists         : {os.path.exists(ref_path)}")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Old ECALP feature reference was not found. Expected: {ref_path}\n"
            "ImageNet uses I.pt; ImageNet OOD uses A.pt, V.pt, R.pt and K.pt "
            "under features/style_transfer/RN."
        )
    old = safe_torch_load(ref_path, map_location="cpu")
    keys = list(old.keys()) if isinstance(old, dict) else []
    print(f"Keys           : {keys}")
    if isinstance(old, dict) and "image_features" in old:
        print(f"Image shape    : {tuple(old['image_features'].shape)}")
    if isinstance(old, dict) and "labels" in old:
        labels = old["labels"].long().cpu()
        print(f"Labels shape   : {tuple(labels.shape)}")
        print(f"Labels min/max : {int(labels.min())}/{int(labels.max())}")
        print(f"Classes in old : {len(torch.unique(labels))}")
        print(f"First 20 labels: {labels[:20].tolist()}")
    print("=" * 80)
    return ref_path


def maximum_weight_assignment(similarity, context):
    """Return row/column indices for the maximum-sum one-to-one assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise ImportError(
            "scipy is required for exact old-feature alignment. "
            "Install it in the current environment with: pip install scipy"
        ) from error
    matrix = similarity.detach().float().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-matrix)
    if len(row_ind) != min(matrix.shape):
        raise RuntimeError(
            f"[{context}] Incomplete Hungarian assignment: "
            f"matched={len(row_ind)}, shape={matrix.shape}"
        )
    return torch.as_tensor(row_ind).long(), torch.as_tensor(col_ind).long()


def derive_ood_local_to_global_map(args, old_cache, old_unique, new_unique):
    """Identify A/R local classes from original ECALP text features, not id sorting."""
    if "text_features" not in old_cache:
        raise RuntimeError("Old OOD cache has no text_features for class-space alignment.")
    source_path = os.path.join(args.old_features_root, "fine_grained", args.clip_model, "I.pt")
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Original ImageNet source feature is required for OOD class mapping: {source_path}"
        )
    source_cache = safe_torch_load(source_path, map_location="cpu")
    if "text_features" not in source_cache:
        raise RuntimeError(f"Original ImageNet cache has no text_features: {source_path}")

    old_text_all = F.normalize(torch.as_tensor(old_cache["text_features"]).float(), dim=-1)
    source_text_all = F.normalize(torch.as_tensor(source_cache["text_features"]).float(), dim=-1)
    if int(old_unique.max()) >= old_text_all.shape[0] or int(new_unique.max()) >= source_text_all.shape[0]:
        raise RuntimeError(
            "Text-feature rows do not cover the old/new class ids: "
            f"old_text={old_text_all.shape[0]}, source_text={source_text_all.shape[0]}"
        )
    old_text = old_text_all.index_select(0, old_unique)
    candidate_text = source_text_all.index_select(0, new_unique)
    similarity = old_text @ candidate_text.t()
    rows, cols = maximum_weight_assignment(similarity, "OOD class mapping")
    mapping = {
        int(old_unique[row]): int(new_unique[col])
        for row, col in zip(rows.tolist(), cols.tolist())
    }
    scores = similarity[rows, cols]
    if scores.mean().item() < 0.95:
        raise RuntimeError(
            "[OOD class mapping] Original OOD/ImageNet text features do not align reliably: "
            f"mean cosine={scores.mean().item():.6f}"
        )
    print("=" * 80)
    print("[Order Alignment] OOD local classes matched through original text features")
    print(f"OOD text cache    : {get_old_feature_reference_path(args)}")
    print(f"ImageNet text     : {source_path}")
    print(f"Classes matched   : {len(mapping)}")
    print(f"Mean text cosine  : {scores.mean().item():.6f}")
    print(f"Min text cosine   : {scores.min().item():.6f}")
    print(f"First mappings    : {list(mapping.items())[:10]}")
    print("=" * 80)
    return mapping


def derive_imagenet_v2_local_to_global_map(test_paths, old_unique, new_unique):
    """Reproduce DMN's ImageFolder ordering for ImageNet-V2 numeric folders.

    DMN loads ImageNet-V2 with torchvision.datasets.ImageFolder and then orders
    its text classes with ``imagenet_v_mask``. ImageFolder sorts directory names
    lexicographically, so the local class order is 0, 1, 10, 100, ... rather
    than the numeric ImageNet order 0, 1, 2, 3, .... The newly extracted cache
    deliberately keeps official ImageNet-1K ids; this mapping converts the old
    DMN local row ids to those official ids before image-level alignment.
    """
    folder_names = sorted({
        os.path.basename(os.path.dirname(str(path)))
        for path in test_paths
    })
    non_numeric = [name for name in folder_names if not name.isdigit()]
    if non_numeric:
        raise RuntimeError(
            "[ImageNet-V2 class mapping] Expected numeric class folders, got: "
            f"{non_numeric[:10]}"
        )

    global_ids = [int(name) for name in folder_names]
    if len(global_ids) != old_unique.numel():
        raise RuntimeError(
            "[ImageNet-V2 class mapping] Class-folder count does not match the "
            f"old cache: folders={len(global_ids)}, old={old_unique.numel()}"
        )
    if set(global_ids) != set(int(value) for value in new_unique.tolist()):
        raise RuntimeError(
            "[ImageNet-V2 class mapping] Numeric folder ids do not match the "
            "new ImageNet-1K labels."
        )

    mapping = {
        int(old_id): int(global_id)
        for old_id, global_id in zip(old_unique.tolist(), global_ids)
    }
    print("=" * 80)
    print("[Order Alignment] ImageNet-V2 DMN/ImageFolder class mapping")
    print("Old local order : lexicographic numeric-folder order")
    print("New label order : official ImageNet-1K numeric ids")
    print(f"Classes mapped  : {len(mapping)}")
    print(f"First mappings  : {list(mapping.items())[:15]}")
    print("=" * 80)
    return mapping


def align_test_features_to_old_order(args, clip_test_features, visual_test_features, test_labels, test_paths):
    """Reorder new CLIP and visual(MoCo/DINO) test features to old ECALP/DMN order.

    Matching anchor is the old CLIP image feature file. The same computed order is
    applied to the visual branch, labels and paths, so CLIP and MoCo stay aligned.
    """
    if not getattr(args, "align_to_old_order", False):
        return clip_test_features, visual_test_features, test_labels, test_paths, None

    ref_path = get_old_feature_reference_path(args)
    if not os.path.exists(ref_path):
        print("=" * 80)
        print("[Order Alignment] Reference old feature not found, skip alignment.")
        print(f"Reference path: {ref_path}")
        print("=" * 80)
        return clip_test_features, visual_test_features, test_labels, test_paths, None

    old = safe_torch_load(ref_path, map_location="cpu")
    if "image_features" not in old or "labels" not in old:
        print("=" * 80)
        print("[Order Alignment] Reference feature lacks image_features/labels, skip alignment.")
        print(f"Reference path: {ref_path}")
        print("=" * 80)
        return clip_test_features, visual_test_features, test_labels, test_paths, None

    old_feat = F.normalize(old["image_features"].float(), dim=-1)
    old_labels = old["labels"].long().cpu()
    new_feat = F.normalize(clip_test_features.float(), dim=-1)
    new_labels = torch.tensor(test_labels).long().cpu()

    # ImageNet-A/R old ECALP caches may use a local 0..199 label space,
    # whereas the new cross-domain extractor keeps official ImageNet-1K ids.
    # When both files contain the same number of classes but different id sets,
    # map old sorted class ids to new sorted global ids before row matching.
    new_labels_for_match = new_labels.clone()
    old_unique = torch.unique(old_labels, sorted=True)
    new_unique = torch.unique(new_labels, sorted=True)
    label_id_remap = None
    # ImageNet-V2 is a special case: both old and new labels span 0..999, but
    # their meanings differ. The old DMN cache follows ImageFolder's
    # lexicographic numeric-folder order and the new extractor uses official
    # numeric ImageNet ids. Therefore this remapping is required even though
    # the two unique-id tensors look identical.
    force_v2_remap = args.dataset == "ImageNet-V2"
    if force_v2_remap or not torch.equal(old_unique, new_unique):
        if old_unique.numel() != new_unique.numel():
            raise RuntimeError(
                "[Order Alignment] Old/new class counts differ and labels cannot be remapped: "
                f"old={old_unique.numel()}, new={new_unique.numel()}"
            )
        if force_v2_remap:
            label_id_remap = derive_imagenet_v2_local_to_global_map(
                test_paths, old_unique, new_unique
            )
        elif args.dataset in IMAGENET_OOD_DATASETS:
            label_id_remap = derive_ood_local_to_global_map(
                args, old, old_unique, new_unique
            )
        else:
            label_id_remap = {
                int(old_id): int(new_id)
                for old_id, new_id in zip(old_unique.tolist(), new_unique.tolist())
            }
        global_to_old = {global_id: old_id for old_id, global_id in label_id_remap.items()}
        missing_global_ids = sorted(set(new_labels.tolist()) - set(global_to_old.keys()))
        if missing_global_ids:
            raise RuntimeError(
                "[Order Alignment] New global labels cannot be converted to the old local space: "
                f"{missing_global_ids[:20]}"
            )
        new_labels_for_match = torch.tensor(
            [global_to_old[int(label)] for label in new_labels.tolist()],
            dtype=torch.long,
        )
        print("=" * 80)
        print("[Order Alignment] Remap old local class ids to ImageNet global ids")
        print(f"Classes        : {old_unique.numel()}")
        print(f"Old id range   : {int(old_unique.min())}..{int(old_unique.max())}")
        print(f"New id range   : {int(new_unique.min())}..{int(new_unique.max())}")
        print(f"First mappings : {list(label_id_remap.items())[:10]}")
        print("=" * 80)

    if old_feat.shape != new_feat.shape or old_labels.numel() != new_labels.numel():
        print("=" * 80)
        print("[Order Alignment] Shape mismatch, skip alignment.")
        print(f"Reference path : {ref_path}")
        print(f"old image shape: {tuple(old_feat.shape)}")
        print(f"new image shape: {tuple(new_feat.shape)}")
        print(f"old labels     : {tuple(old_labels.shape)}")
        print(f"new labels     : {tuple(new_labels.shape)}")
        print("=" * 80)
        return clip_test_features, visual_test_features, test_labels, test_paths, None

    print("=" * 80)
    print("[Order Alignment] Aligning CLIP and visual branch to original ECALP/DMN order")
    print(f"Dataset        : {args.dataset}")
    print(f"Reference path : {ref_path}")
    print(f"Samples        : {old_labels.numel()}")
    print(f"Classes        : {len(torch.unique(old_labels))}")
    print("=" * 80)

    order = torch.full((old_labels.numel(),), -1, dtype=torch.long)
    all_scores = []
    for c in torch.unique(old_labels).tolist():
        old_idx = torch.where(old_labels == int(c))[0]
        new_idx = torch.where(new_labels_for_match == int(c))[0]
        if old_idx.numel() != new_idx.numel():
            raise RuntimeError(
                f"[Order Alignment] Class {c} count mismatch: "
                f"old={old_idx.numel()}, new={new_idx.numel()}"
            )
        old_c = old_feat[old_idx]
        new_c = new_feat[new_idx]
        sim = old_c @ new_c.t()
        rows, cols = maximum_weight_assignment(sim, f"Image class {c}")
        for row, col in zip(rows.tolist(), cols.tolist()):
            order[int(old_idx[row].item())] = int(new_idx[col].item())
            all_scores.append(float(sim[row, col].item()))

    unique_order = torch.unique(order)
    if unique_order.numel() != order.numel():
        raise RuntimeError(
            "[Order Alignment] Computed order is not a one-to-one permutation: "
            f"unique={unique_order.numel()}, total={order.numel()}"
        )
    if int(order.min()) < 0 or int(order.max()) >= new_labels.numel():
        raise RuntimeError(
            "[Order Alignment] Computed order contains an out-of-range row index: "
            f"min={int(order.min())}, max={int(order.max())}, N={new_labels.numel()}"
        )

    aligned_clip = clip_test_features[order]
    aligned_visual = visual_test_features[order]
    aligned_global_labels = new_labels[order].tolist()
    aligned_labels = new_labels_for_match[order].tolist()
    aligned_paths = [test_paths[int(i)] for i in order.tolist()]

    label_equal = torch.equal(torch.tensor(aligned_labels).long(), old_labels.long())
    if not label_equal:
        raise RuntimeError(
            "[Order Alignment] Labels do not match the old ECALP order after alignment."
        )
    aligned_img = F.normalize(aligned_clip.float(), dim=-1)
    img_cos = (old_feat * aligned_img).sum(dim=1)
    if not torch.isfinite(img_cos).all():
        raise RuntimeError("[Order Alignment] Non-finite CLIP cosine found after alignment.")
    if img_cos.mean().item() < 0.90:
        raise RuntimeError(
            "[Order Alignment] Mean matched CLIP cosine is below 0.90, so image-level "
            f"alignment is not reliable: {img_cos.mean().item():.6f}"
        )

    print("=" * 80)
    print("[Order Alignment] Done")
    print(f"Label equal after alignment : {label_equal}")
    print(f"Mean matched cosine         : {sum(all_scores) / len(all_scores):.6f}")
    print(f"Image mean cosine to old    : {img_cos.mean().item():.6f}")
    print(f"Image min cosine to old     : {img_cos.min().item():.6f}")
    print(f"Image p05 cosine to old     : {torch.quantile(img_cos, 0.05).item():.6f}")
    print("Assignment solver           : Hungarian (maximum total cosine)")
    print(f"First 20 old labels (saved) : {old_labels[:20].tolist()}")
    if label_id_remap is not None:
        print(f"New labels in old space     : {new_labels_for_match[order][:20].tolist()}")
        print(f"Matched ImageNet global ids : {aligned_global_labels[:20]} (metadata only)")
    print(f"First 20 aligned labels     : {aligned_labels[:20]}")
    print("=" * 80)

    align_info = {
        "aligned_to_old_order": True,
        "old_feature_path": ref_path,
        "order_new_index_for_each_old_position": order,
        "label_equal_after_alignment": label_equal,
        "mean_matched_cosine": sum(all_scores) / len(all_scores),
        "image_mean_cosine_to_old": img_cos.mean().item(),
        "image_min_cosine_to_old": img_cos.min().item(),
        "old_to_global_label_id_remap": label_id_remap,
        "aligned_global_labels": torch.tensor(aligned_global_labels, dtype=torch.long),
        "assignment_solver": "hungarian_maximum_total_cosine",
        "image_p05_cosine_to_old": torch.quantile(img_cos, 0.05).item(),
        "order_is_bijective": True,
        "alignment_verified": True,
    }
    return aligned_clip, aligned_visual, aligned_labels, aligned_paths, align_info


def build_imagenet_synset_map(classes):
    """Build synset/name mapping for ImageNet.

    Supports CuPL keys such as:
        n01440764
        tench
        tench, Tinca tinca
    and ImageNet folder names such as n01440764.
    """
    mp = {}
    for i, c in enumerate(classes):
        c_str = str(c).strip()
        if c_str:
            mp.setdefault(norm(c_str), i)
            mp.setdefault(c_str, i)

        # If key is like "n01440764 tench" or "n01440764, tench"
        first = c_str.replace(",", " ").split()[0] if c_str else ""
        if first.startswith("n") and len(first) == 9:
            mp.setdefault(first, i)

        # If key is "tench, Tinca tinca", map each alias.
        for part in c_str.split(","):
            part = part.strip()
            if part:
                mp.setdefault(norm(part), i)

    return mp


def imagenet_label_to_idx(label, mp):
    label = str(label).strip()
    # ImageNet-V2 commonly stores classes as numeric folders 0..999.
    if label.isdigit():
        idx = int(label)
        if 0 <= idx < 1000:
            return idx
    if label in mp:
        return mp[label]
    nlabel = norm(label)
    if nlabel in mp:
        return mp[nlabel]
    raise KeyError(f"Cannot map ImageNet label/folder to class index: {label}")

def label_to_idx(label, mp):
    k = norm(label)
    if k in mp: return mp[k]
    raise KeyError(f"Cannot map label to CuPL class: {label}")

def resolve_path(root, image_dir, rel):
    """Resolve image path robustly across common dataset layouts.

    This version supports Flowers102-style layouts such as:
        102flowers/jpg/image_00001.jpg
    while also keeping the old resolution logic.
    """
    rel = str(rel).replace("\\", "/")
    cand = []

    if os.path.isabs(rel):
        cand.append(rel)

    # Try the raw relative path under common roots.
    prefixes = [
        root,
        image_dir,
        os.path.join(root, "images"),
        os.path.join(root, "Images"),
        os.path.join(root, "JPEGImages"),
        os.path.join(root, "jpg"),
        os.path.join(root, "JPG"),
        os.path.join(root, "SUN397"),
        os.path.join(root, "SUN397", "images"),
        os.path.join(root, "Places365"),
        os.path.join(image_dir, "images"),
        os.path.join(image_dir, "Images"),
        os.path.join(image_dir, "JPEGImages"),
        os.path.join(image_dir, "jpg"),
        os.path.join(image_dir, "JPG"),
    ]

    for pre in prefixes:
        cand.append(os.path.join(pre, rel))

    base, ext = os.path.splitext(rel)
    if not ext:
        for e in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]:
            for pre in prefixes:
                cand.append(os.path.join(pre, rel + e))

    # Remove duplicates while preserving order.
    seen = set()
    uniq = []
    for p in cand:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    for p in uniq:
        if os.path.exists(p):
            return p

    # Final fallback: search by basename recursively. This fixes cases where
    # split json contains paths like "images/xxx.jpg" but local data is stored
    # as "JPEGImages/class/xxx.jpg" or similar.
    hit = find_by_basename([root, image_dir, BASE_DATA_ROOT], os.path.basename(rel))
    if hit:
        return hit

    return uniq[0] if uniq else rel


def read_aircraft(root, image_dir, split, mp):
    fp = os.path.join(root, f"images_variant_{split}.txt")
    if not os.path.exists(fp): raise FileNotFoundError(fp)
    items = []
    for line in open(fp, "r", encoding="utf-8"):
        line = line.strip()
        if not line: continue
        parts = line.split(" ")
        image_id, label = parts[0], " ".join(parts[1:]).strip()
        path = os.path.join(image_dir, image_id + ".jpg")
        if not os.path.exists(path): raise FileNotFoundError(path)
        items.append({"path": path, "label": label_to_idx(label, mp), "label_name": label, "image_id": image_id})
    return items


def read_imagenet(root, image_dir, split, mp):
    """Read ImageNet from common folder layouts.

    Supported layouts:
      1) root/train/n01440764/*.JPEG and root/val/n01440764/*.JPEG
      2) image_dir/train/n01440764/*.JPEG and image_dir/val/n01440764/*.JPEG
      3) root/ILSVRC2012_img_train/n01440764/*.JPEG
      4) root/ILSVRC2012_img_val/n01440764/*.JPEG, if val images are already organized by class

    This function expects class folders. If your ImageNet val set is flat with
    labels in a separate text file, organize val into class subfolders first or
    pass an ImageFolder-style image_root.
    """
    split_alias = {
        "test": ["val", "validation", "test"],
        "val": ["val", "validation"],
        "train": ["train", "ILSVRC2012_img_train"],
        "trainval": ["train", "ILSVRC2012_img_train"],
    }

    names = split_alias.get(split, [split])
    candidate_bases = []

    for base_root in [image_dir, root]:
        for name in names:
            candidate_bases.append(os.path.join(base_root, name))
            candidate_bases.append(os.path.join(base_root, "images", name))
        if split in ["val", "test"]:
            candidate_bases.append(os.path.join(base_root, "ILSVRC2012_img_val"))
            candidate_bases.append(os.path.join(base_root, "images", "ILSVRC2012_img_val"))
        if split in ["train", "trainval"]:
            candidate_bases.append(os.path.join(base_root, "ILSVRC2012_img_train"))
            candidate_bases.append(os.path.join(base_root, "images", "ILSVRC2012_img_train"))
        # ImageNet-V2 archives are often extracted with one extra directory
        # between the dataset root and the numeric 0..999 class folders.
        candidate_bases.append(os.path.join(base_root, "imagenetv2-matched-frequency-format-val"))
        candidate_bases.append(os.path.join(base_root, "matched-frequency-format-val"))
        # OOD datasets are usually already rooted at their class folders.
        candidate_bases.append(base_root)

    base = None
    for p in candidate_bases:
        if os.path.exists(p):
            base = p
            break

    if base is None:
        raise FileNotFoundError(
            "ImageNet split folder not found. Tried:\\n  "
            + "\\n  ".join(candidate_bases)
            + "\\nPlease set --dataset_root / --image_root correctly."
        )

    items = []
    exts = ["*.JPEG", "*.jpeg", "*.JPG", "*.jpg", "*.png", "*.PNG"]

    class_dirs = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
    if not class_dirs:
        raise RuntimeError(
            f"ImageNet folder has no class subdirectories: {base}\\n"
            "Expected layout like val/n01440764/*.JPEG or train/n01440764/*.JPEG."
        )

    # In many local ImageNet copies, folder names are WordNet synsets such as
    # n01440764, while CuPL imagenet.json may use only English names such as
    # "tench". In that case direct name matching fails and every folder would
    # be skipped. If the folder set looks like a full ImageNet-1K synset folder
    # layout, fall back to the standard ImageFolder class index: sorted folder
    # order. This is the same convention used by torchvision ImageFolder.
    C = max([int(v) for v in mp.values()]) + 1 if mp else len(class_dirs)
    synset_like_dirs = [d for d in class_dirs if d.startswith("n") and len(d) == 9 and d[1:].isdigit()]
    use_sorted_synset_fallback = (len(synset_like_dirs) == len(class_dirs) and len(class_dirs) == C)
    if use_sorted_synset_fallback:
        print("[ImageNet] CuPL keys do not have to be synsets; using sorted synset folder order as class index fallback.")

    skipped_folders = []

    for cls_folder in class_dirs:
        try:
            y = imagenet_label_to_idx(cls_folder, mp)
        except Exception:
            if use_sorted_synset_fallback:
                y = class_dirs.index(cls_folder)
            else:
                skipped_folders.append(cls_folder)
                continue

        folder = os.path.join(base, cls_folder)
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(folder, e)))

        for p in sorted(paths):
            items.append({
                "path": p,
                "label": y,
                "label_name": cls_folder,
                "image_id": os.path.splitext(os.path.basename(p))[0],
            })

    if skipped_folders:
        print(f"[ImageNet Warning] Skipped folders not mapped to CuPL classes: {len(skipped_folders)} | examples: {skipped_folders[:5]}")

    if not items:
        example_dirs = class_dirs[:10]
        raise RuntimeError(
            f"No ImageNet images loaded from {base}. "
            f"Found class folders like {example_dirs}. "
            "Your CuPL class keys probably do not match WordNet synset folder names. "
            "This fixed version can handle full ImageNet-1K synset folders by sorted order; "
            "if it still fails, check whether val/train has 1000 class subdirectories."
        )

    return items


def read_split_json(split_json, root, image_dir, split, mp):
    data = json.load(open(split_json, "r", encoding="utf-8"))
    if split == "trainval" and "train" in data and "val" in data:
        raw = data["train"] + data["val"]
    else:
        if split not in data: raise KeyError(f"Split {split} not in {split_json}; keys={list(data.keys())}")
        raw = data[split]
    items, missing = [], []
    for x in raw:
        if isinstance(x, dict):
            rel = x.get("impath", x.get("image", x.get("path", "")))
            label_name = x.get("classname", x.get("class_name", x.get("label_name", "")))
            label_int = x.get("label", None)
        else:
            rel = x[0]
            label_int = x[1] if len(x) > 1 else None
            label_name = x[2] if len(x) > 2 else ""
        path = resolve_path(root, image_dir, rel)
        if not os.path.exists(path):
            missing.append(path); continue
        if label_name:
            try: y = label_to_idx(label_name, mp)
            except Exception:
                if label_int is None: raise
                y = int(label_int)
        else:
            y = int(label_int)
        items.append({"path": path, "label": y, "label_name": str(label_name) if label_name else str(y), "image_id": os.path.splitext(os.path.basename(path))[0]})
    if missing: print(f"[Warning] Missing images skipped: {len(missing)} | examples: {missing[:3]}")
    if not items: raise RuntimeError(f"No images loaded from {split_json}")
    return items


def read_imagefolder(root, image_dir, split, mp):
    base = os.path.join(image_dir, split)
    if not os.path.exists(base):
        base = image_dir
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"ImageFolder path not found: {base}\n"
            f"root={root}\nimage_dir={image_dir}\n"
            f"Please set --dataset_root / --image_root manually for this dataset."
        )
    items = []
    for cls in sorted(os.listdir(base)):
        folder = os.path.join(base, cls)
        if not os.path.isdir(folder): continue
        try: y = label_to_idx(cls, mp)
        except Exception: continue
        paths = []
        for e in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]: paths += glob.glob(os.path.join(folder, e))
        for p in sorted(paths):
            items.append({"path": p, "label": y, "label_name": cls, "image_id": os.path.splitext(os.path.basename(p))[0]})
    # Some datasets are not organized as class subfolders in the local copy.
    # If split json is missing and ImageFolder fails, give a clearer error.
    if not items:
        raise RuntimeError(
            f"No images loaded from ImageFolder: {base}. "
            f"Usually this means the split_zhou json was not found or class "
            f"folder names do not match CuPL class names."
        )
    return items


def load_items(args, split, mp):
    cfg = DATASET_LOCAL_PATHS[args.dataset]
    root = args.dataset_root or cfg["root"]
    image_dir = args.image_root or cfg["image_dir"]
    split_json = args.split_json or cfg.get("split_json", "")
    print("="*80)
    print(f"Dataset      : {args.dataset}")
    print(f"Dataset root : {root}")
    print(f"Image dir    : {image_dir}")
    print(f"Split json   : {split_json}")
    print(f"Split        : {split}")
    print("="*80)
    if not os.path.exists(root):
        print(f"[Warning] Resolved dataset root does not exist: {root}")
        print("          You can override it with --dataset_root and --image_root.")
    if not os.path.exists(image_dir):
        print(f"[Warning] Resolved image dir does not exist: {image_dir}")
        print("          You can override it with --image_root.")
    if args.dataset == "Aircraft":
        items = read_aircraft(root, image_dir, split, mp)
    elif args.dataset == "ImageNet" or args.dataset in IMAGENET_OOD_DATASETS:
        # ImageNet uses synset folder names, e.g., n01440764.
        # Build an expanded mapping from CuPL classes to synset/name aliases.
        classes_tmp = list(json.load(open(args.cupl_json, "r", encoding="utf-8")).keys()) if os.path.exists(args.cupl_json) else []
        imagenet_mp = build_imagenet_synset_map(classes_tmp)
        imagenet_mp.update(mp)
        items = read_imagenet(root, image_dir, split, imagenet_mp)
    elif split_json and os.path.exists(split_json):
        items = read_split_json(split_json, root, image_dir, split, mp)
    else:
        items = read_imagefolder(root, image_dir, split, mp)
    if args.max_images > 0: items = items[:args.max_images]
    print(f"Loaded images : {len(items)}")
    print(f"Loaded classes: {len(set(i['label'] for i in items))}")
    print(f"First item    : {os.path.basename(items[0]['path'])} | {items[0]['label_name']} -> {items[0]['label']}")
    return items


def collapse(out, name):
    if isinstance(out, dict):
        for k in ["image_features", "x_norm_clstoken", "cls_token", "features", "last_hidden_state"]:
            if k in out: out = out[k]; break
    if isinstance(out, (list, tuple)): out = out[0]
    if not torch.is_tensor(out): raise RuntimeError(f"{name} output is not tensor")
    if out.dim() == 3: out = out[:,0,:]
    if out.dim() == 4: out = out.mean(dim=(2,3))
    if out.dim() != 2: raise RuntimeError(f"{name} must be [B,D], got {tuple(out.shape)}")
    return out


def load_dino(size, ckpt, device):
    from clip.dinoV2 import dinov2_vits14, dinov2_vitb14, dinov2_vitl14
    m = {"s": dinov2_vits14, "b": dinov2_vitb14, "l": dinov2_vitl14}[size](pretrained=False)
    sd = safe_torch_load(ckpt, "cpu")
    if isinstance(sd, dict): sd = sd.get("state_dict", sd.get("model", sd))
    ns = {}
    for k,v in sd.items():
        nk = k
        for pre in ["module.", "backbone.", "teacher."]:
            if nk.startswith(pre): nk = nk[len(pre):]
        ns[nk] = v
    missing, unexpected = m.load_state_dict(ns, strict=False)
    print("="*80); print("DINOv2 checkpoint loaded"); print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}"); print("="*80)
    return m.to(device).eval()


def dino_transform(size=224):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        interp = InterpolationMode.BICUBIC
    except Exception:
        interp = Image.BICUBIC
    return Compose([Resize(size, interpolation=interp), CenterCrop(size), ToTensor(), Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])


def load_mocov3_resnet50(ckpt, device):
    """Load official MoCo v3 ResNet50 checkpoint.

    Expected checkpoint:
        r-50-1000ep.pth.tar

    MoCo checkpoints often store keys as:
        state_dict["module.base_encoder.xxx"]
    or similar. This loader strips common prefixes and loads only the
    ResNet backbone weights. The final fc layer is replaced by Identity,
    so the output feature dimension is 2048.
    """
    import torchvision.models as models
    import torch.nn as nn

    model = models.resnet50(pretrained=False)
    model.fc = nn.Identity()

    ckpt_obj = safe_torch_load(ckpt, "cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj

    new_state = {}
    for k, v in state.items():
        nk = k
        for pre in [
            "module.base_encoder.",
            "module.encoder.",
            "module.backbone.",
            "module.",
            "base_encoder.",
            "encoder.",
            "backbone.",
        ]:
            if nk.startswith(pre):
                nk = nk[len(pre):]

        # Skip projection/prediction heads and classifier weights.
        if nk.startswith("fc.") or nk.startswith("head.") or nk.startswith("mlp."):
            continue
        if "predictor" in nk or "projector" in nk:
            continue

        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("=" * 80)
    print("MoCo v3 ResNet50 checkpoint loaded")
    print(f"Checkpoint : {ckpt}")
    print(f"Missing    : {len(missing)}")
    print(f"Unexpected : {len(unexpected)}")
    print("=" * 80)
    return model.to(device).eval()



def load_mocov3_vitb16(ckpt, device):
    """Load official MoCo v3 ViT-Base/16 checkpoint.

    Expected checkpoint:
        vit-b-300ep.pth.tar

    This uses timm ViT-B/16 and strips common MoCo prefixes such as
    module.base_encoder.*. The output feature dimension is 768.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for MoCo v3 ViT-B/16. Install it with: pip install timm"
        ) from e

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)

    ckpt_obj = safe_torch_load(ckpt, "cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj

    new_state = {}
    for k, v in state.items():
        nk = k
        for pre in [
            "module.base_encoder.",
            "module.encoder.",
            "module.backbone.",
            "module.",
            "base_encoder.",
            "encoder.",
            "backbone.",
        ]:
            if nk.startswith(pre):
                nk = nk[len(pre):]

        # Skip MoCo heads / linear classifiers.
        if nk.startswith("head.") or nk.startswith("fc.") or nk.startswith("mlp_head."):
            continue
        if "predictor" in nk or "projector" in nk:
            continue

        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("=" * 80)
    print("MoCo v3 ViT-B/16 checkpoint loaded")
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


def load_visual_encoder(args, device):
    """Load DINOv2, MoCo v3 ResNet50, or MoCo v3 ViT-B/16 visual encoder.

    Command-line behavior:
      --visual_encoder mocov3_r50  -> ckpt=args.mocov3_ckpt,     save folder=MOCO/R50
      --visual_encoder mocov3_vit  -> ckpt=args.mocov3_vit_ckpt, save folder=MOCO/VIT
      --visual_encoder dino        -> ckpt=args.dino_ckpt,       save folder=DINO
    """
    if args.visual_encoder == "dino":
        if not args.dino_ckpt:
            raise ValueError("--dino_ckpt is required when --visual_encoder dino")
        model = load_dino(args.dino_size, args.dino_ckpt, device)
        pre = dino_transform(args.visual_image_size)
        name = args.visual_model_name or "DINO"
        return model, pre, name

    if args.visual_encoder in ["mocov3", "mocov3_r50", "r50"]:
        if not args.mocov3_ckpt:
            raise ValueError("--mocov3_ckpt is required when --visual_encoder mocov3_r50")
        model = load_mocov3_resnet50(args.mocov3_ckpt, device)
        pre = moco_transform(args.visual_image_size)
        name = args.visual_model_name or "MOCO/R50"
        return model, pre, name

    if args.visual_encoder in ["mocov3_vit", "mocov3_vitb", "vit"]:
        if not args.mocov3_vit_ckpt:
            raise ValueError("--mocov3_vit_ckpt is required when --visual_encoder mocov3_vit")
        model = load_mocov3_vitb16(args.mocov3_vit_ckpt, device)
        pre = moco_transform(args.visual_image_size)
        name = args.visual_model_name or "MOCO/VIT"
        return model, pre, name

    raise ValueError(f"Unknown visual_encoder: {args.visual_encoder}")


def encode_dual(paths, clip_model, clip_pre, visual_model, visual_pre, device, bs, desc):
    cf_all, vf_all = [], []
    for s in tqdm(range(0, len(paths), bs), desc=desc):
        ims = [load_pil(p) for p in paths[s:s+bs]]
        cb = torch.stack([clip_pre(im) for im in ims]).to(device)
        vb = torch.stack([visual_pre(im) for im in ims]).to(device)
        with torch.no_grad():
            cf = F.normalize(collapse(clip_model.encode_image(cb), "CLIP").float(), dim=-1)
            vf = F.normalize(collapse(visual_model(vb), "VISUAL").float(), dim=-1)
        cf_all.append(cf.cpu()); vf_all.append(vf.cpu())
    return ensure_2d(torch.cat(cf_all), desc+" CLIP"), ensure_2d(torch.cat(vf_all), desc+" VISUAL")


def _normalize_prompt_value(v):
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [str(p).strip() for p in v if str(p).strip()]
    if isinstance(v, dict):
        for key in ["prompts", "prompt", "descriptions", "description", "captions", "sentences", "texts"]:
            if key in v:
                return _normalize_prompt_value(v[key])
    return []


def get_prompts(cupl, cls):
    # For ImageNet, classnames.txt often gives "tench, Tinca tinca" while CuPL uses "tench".
    candidates = [cls, cls.replace("_", " "), cls.replace(" ", "_")]
    if "," in str(cls):
        first = str(cls).split(",")[0].strip()
        candidates.extend([first, first.replace("_", " "), first.replace(" ", "_")])
    for k in candidates:
        if k in cupl:
            ps = _normalize_prompt_value(cupl[k])
            if ps:
                return ps
    norm_candidates = {norm(x) for x in candidates}
    for k, v in cupl.items():
        if norm(k) in norm_candidates:
            ps = _normalize_prompt_value(v)
            if ps:
                return ps
    return [f"a photo of a {str(cls).replace('_', ' ')}."]


def text_features(clip_model, classes, cupl, device, bs):
    outs = []
    for c in tqdm(classes, desc="Encode CuPL text"):
        prompts = get_prompts(cupl, c)
        fs = []
        for s in range(0, len(prompts), bs):
            tok = clip.tokenize(prompts[s:s+bs], truncate=True).to(device)
            with torch.no_grad(): fs.append(F.normalize(clip_model.encode_text(tok).float(), dim=-1))
        outs.append(F.normalize(torch.cat(fs).mean(0), dim=-1).cpu())
    return F.normalize(torch.stack(outs), dim=-1)


def select_few(labels, C, shots, seed):
    rng = random.Random(seed); pc = [[] for _ in range(C)]
    for i,y in enumerate(labels):
        if 0 <= int(y) < C: pc[int(y)].append(i)
    sel = []
    for c in range(C):
        inds = pc[c][:]
        if len(inds) < shots: raise RuntimeError(f"Class {c} has only {len(inds)} samples, shots={shots}")
        rng.shuffle(inds); sel.append(inds[:shots])
    return sel


def gather(feats, sel): return ensure_3d(torch.stack([feats[i] for i in sel]), "few-shot image_features")

def onehot(sel, labels, C):
    out = torch.zeros(len(sel), len(sel[0]), C)
    for c, inds in enumerate(sel):
        for j, idx in enumerate(inds): out[c,j,int(labels[idx])] = 1.0
    return out


def _source_feature_paths(args, visual_model_name):
    root = getattr(args, "source_features_root", "") or args.output_root
    task = getattr(args, "source_features_task", "fine_grained")
    old_clip_root = os.path.join(args.old_features_root, task, args.clip_model)
    generated_clip_root = os.path.join(root, task, args.clip_model)
    visual_root = os.path.join(root, task, visual_model_name)
    clip_few_candidates = [
        os.path.join(old_clip_root, "few_shots", "I.pt"),
        os.path.join(generated_clip_root, "few_shots", "ImageNet.pt"),
    ]
    return {
        # Released ECALP caches use I.pt.  A cache created by this extractor
        # uses the canonical dataset name ImageNet.pt.  Prefer the released
        # cache when present, but allow a freshly generated ViT source cache
        # so ImageNet and its OOD variants can be extracted in one command.
        "clip_few": first_existing(*clip_few_candidates),
        "clip_few_candidates": clip_few_candidates,
        "visual_test": os.path.join(visual_root, "ImageNet.pt"),
        "visual_few": os.path.join(visual_root, "few_shots", "ImageNet.pt"),
        "meta": os.path.join(root, task, "ImageNet_aligned_meta.pt"),
    }


def require_imagenet_source_caches(source_paths):
    missing = []
    if not os.path.exists(source_paths["clip_few"]):
        alternatives = source_paths.get("clip_few_candidates", [source_paths["clip_few"]])
        missing.append("CLIP support (tried: " + ", ".join(alternatives) + ")")
    for label, key in [("visual support", "visual_few"), ("metadata", "meta")]:
        if not os.path.exists(source_paths[key]):
            missing.append(f"{label}: {source_paths[key]}")
    if missing:
        raise FileNotFoundError(
            "ImageNet source cache is required before OOD extraction. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun this extractor with ImageNet before the OOD datasets, using the same "
              "--features_path, --clip_model, --visual_encoder, --num_shots, and --seed."
        )


def ood_cache_filename(dataset_name):
    name = DATASET_LOCAL_PATHS.get(dataset_name, {}).get("old_feature_name", f"{dataset_name}.pt")
    return name if str(name).endswith(".pt") else f"{name}.pt"


def subset_imagenet_source_support(cache, global_class_ids, num_local_classes, cache_name):
    features = torch.as_tensor(cache["image_features"])
    if features.dim() != 3:
        raise RuntimeError(f"{cache_name} image_features must be [C,shot,D], got {tuple(features.shape)}")
    global_ids = torch.as_tensor(global_class_ids, dtype=torch.long)
    if global_ids.numel() != num_local_classes:
        raise RuntimeError(
            f"{cache_name} class map has {global_ids.numel()} rows, expected {num_local_classes}"
        )
    if int(global_ids.min()) < 0 or int(global_ids.max()) >= features.shape[0]:
        raise RuntimeError(
            f"{cache_name} cannot select ImageNet global class ids "
            f"{int(global_ids.min())}..{int(global_ids.max())} from {features.shape[0]} rows"
        )
    selected = features.index_select(0, global_ids)
    shots = selected.shape[1]
    labels = torch.eye(num_local_classes, dtype=torch.float32)[:, None, :].expand(
        num_local_classes, shots, num_local_classes
    ).clone()
    output = dict(cache)
    output["image_features"] = selected
    output["labels"] = labels
    output["source_global_class_ids"] = global_ids
    output["local_label_space"] = True
    return output


def subset_source_meta(source_meta, global_class_ids):
    few_paths = source_meta.get("few_paths", [])
    selected_indices = source_meta.get("selected_indices", [])
    if not few_paths:
        raise RuntimeError("ImageNet source metadata is missing few_paths")
    if not selected_indices:
        raise RuntimeError("ImageNet source metadata is missing selected_indices")
    if max(global_class_ids) >= len(selected_indices):
        raise RuntimeError(
            f"ImageNet source metadata selected_indices has {len(selected_indices)} classes, "
            f"but class id {max(global_class_ids)} is required"
        )

    # selected_indices address the flat ImageNet few_paths pool, so the pool
    # must remain complete even though only OOD-present classes are retained.
    return {
        "few_paths": list(few_paths),
        "selected_indices": [selected_indices[index] for index in global_class_ids],
    }


def validate_imagenet_ood_outputs(args, visual_model_name):
    """Fail fast unless MoCo exactly follows the original A/V/R/K cache order."""
    task_root = os.path.join(args.output_root, args.task)
    filename = ood_cache_filename(args.dataset)
    old_clip_path = get_old_feature_reference_path(args)
    clip_few_path = os.path.join(task_root, args.clip_model, "few_shots", filename)
    visual_path = os.path.join(task_root, visual_model_name, filename)
    visual_few_path = os.path.join(task_root, visual_model_name, "few_shots", filename)
    meta_path = os.path.join(task_root, f"{args.dataset}_aligned_meta.pt")
    for path in [old_clip_path, clip_few_path, visual_path, visual_few_path, meta_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected aligned OOD cache was not saved: {path}")

    clip_data = safe_torch_load(old_clip_path, map_location="cpu")
    clip_few = safe_torch_load(clip_few_path, map_location="cpu")
    visual_data = safe_torch_load(visual_path, map_location="cpu")
    visual_few = safe_torch_load(visual_few_path, map_location="cpu")
    meta = safe_torch_load(meta_path, map_location="cpu")
    clip_features = clip_data["image_features"]
    visual_features = visual_data["image_features"]
    clip_labels = torch.as_tensor(clip_data["labels"]).long()
    visual_labels = torch.as_tensor(visual_data["labels"]).long()
    if clip_features.shape[0] != visual_features.shape[0]:
        raise RuntimeError(
            "CLIP/MoCo OOD row counts differ after alignment: "
            f"CLIP={clip_features.shape[0]}, visual={visual_features.shape[0]}"
        )
    if not torch.equal(clip_labels, visual_labels):
        raise RuntimeError("CLIP/MoCo OOD labels are not in the same row order.")

    visual_align = visual_data.get("align_info") or {}
    required_true = [
        "aligned_to_old_order",
        "alignment_verified",
        "label_equal_after_alignment",
        "order_is_bijective",
    ]
    missing_flags = [key for key in required_true if not bool(visual_align.get(key, False))]
    if missing_flags:
        raise RuntimeError(f"OOD order alignment was not verified: {missing_flags}")

    visual_order_value = visual_align.get("order_new_index_for_each_old_position")
    if visual_order_value is None:
        raise RuntimeError("MoCo alignment permutation was not saved.")
    visual_order = torch.as_tensor(visual_order_value).long()
    num_rows = int(clip_features.shape[0])
    if visual_order.numel() != num_rows or torch.unique(visual_order).numel() != num_rows:
        raise RuntimeError("Saved OOD alignment order is not a complete one-to-one permutation.")

    protocol = visual_data.get("cross_domain_protocol") or {}
    if protocol.get("source_dataset") != "ImageNet" or protocol.get("target_dataset") != args.dataset:
        raise RuntimeError(f"Invalid OOD source/target protocol in {visual_path}")
    if not meta.get("few_paths") or not meta.get("selected_indices"):
        raise RuntimeError("ImageNet source support paths/indices are missing from OOD metadata.")
    if clip_few["image_features"].shape[:2] != visual_few["image_features"].shape[:2]:
        raise RuntimeError("CLIP/MoCo source-support class/shot layouts differ.")
    if not torch.equal(torch.as_tensor(clip_few["labels"]), torch.as_tensor(visual_few["labels"])):
        raise RuntimeError("CLIP/MoCo source-support labels differ.")

    print("=" * 80)
    print("Strict OOD alignment validation passed")
    print(f"Old CLIP used : {old_clip_path}")
    print(f"MoCo cache    : {visual_path}")
    print(f"Rows          : {num_rows}")
    print(f"Labels equal  : {torch.equal(clip_labels, visual_labels)}")
    print(f"One-to-one    : {visual_align.get('order_is_bijective')}")
    print(f"Mean CLIP cos : {float(visual_align.get('image_mean_cosine_to_old', float('nan'))):.6f}")
    print(f"Min CLIP cos  : {float(visual_align.get('image_min_cosine_to_old', float('nan'))):.6f}")
    print("Original CLIP cache is preserved and MoCo follows its row order: True")
    print("=" * 80)


def run_imagenet_ood_dataset(args):
    """Align a new MoCo query cache to the original ECALP A/V/R/K cache."""
    preflight_old_feature_check(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    classes, mp, _ = load_cupl(args.cupl_json)
    classnames_txt = DATASET_LOCAL_PATHS[args.dataset]["classnames_txt"]
    classes, imagenet_mp, _ = load_imagenet_classnames(classnames_txt)
    mp.update(imagenet_mp)
    C = len(classes)

    target = load_items(args, args.test_split, mp)
    target_paths = [x["path"] for x in target]
    target_y = [int(x["label"]) for x in target]
    if not target_y or min(target_y) < 0 or max(target_y) >= C:
        raise RuntimeError(f"Invalid ImageNet-1K target labels: min={min(target_y)}, max={max(target_y)}, C={C}")

    cname = "RN50" if args.clip_model == "RN" else "ViT-B/16"
    loaded = clip.load(cname, device=device)
    if len(loaded) == 3:
        clip_model, _, clip_pre = loaded
    else:
        clip_model, clip_pre = loaded
    clip_model.eval()
    visual_model, visual_pre, visual_model_name = load_visual_encoder(args, device)

    # Check the ImageNet source caches before encoding the potentially large
    # OOD target.  This prevents a missing support cache from being discovered
    # only after tens of minutes of target feature extraction.
    source_paths = _source_feature_paths(args, visual_model_name)
    require_imagenet_source_caches(source_paths)

    clip_target, visual_target = encode_dual(
        target_paths, clip_model, clip_pre, visual_model, visual_pre,
        device, args.batch_size, f"Extract {args.dataset} target"
    )
    clip_target, visual_target, target_y, target_paths, align_info = align_test_features_to_old_order(
        args, clip_target, visual_target, target_y, target_paths
    )

    source_clip_few = safe_torch_load(source_paths["clip_few"], map_location="cpu")
    source_visual_few = safe_torch_load(source_paths["visual_few"], map_location="cpu")
    source_meta = safe_torch_load(source_paths["meta"], map_location="cpu") if os.path.exists(source_paths["meta"]) else {}

    old_clip_path = get_old_feature_reference_path(args)
    old_clip = safe_torch_load(old_clip_path, map_location="cpu")
    if "text_features" not in old_clip:
        raise RuntimeError(f"Old ECALP cache has no text_features: {old_clip_path}")
    old_labels = torch.as_tensor(old_clip["labels"]).long()
    old_text = torch.as_tensor(old_clip["text_features"])
    num_local_classes = int(old_text.shape[0])
    expected_local_ids = torch.arange(num_local_classes, dtype=torch.long)
    old_unique_ids = torch.unique(old_labels, sorted=True)
    if not torch.equal(old_unique_ids, expected_local_ids):
        raise RuntimeError(
            "Old ECALP OOD labels must index the old text rows directly. "
            f"Expected 0..{num_local_classes - 1}, got "
            f"{old_unique_ids[:10].tolist()}...{old_unique_ids[-10:].tolist()}"
        )

    old_to_global = align_info.get("old_to_global_label_id_remap") or {
        int(index): int(index) for index in range(num_local_classes)
    }
    global_class_ids = []
    for local_id in range(num_local_classes):
        if local_id not in old_to_global:
            raise RuntimeError(f"No ImageNet global-class mapping for old local class {local_id}")
        global_id = int(old_to_global[local_id])
        if not 0 <= global_id < C:
            raise RuntimeError(f"Mapped ImageNet class id is out of range: {local_id}->{global_id}")
        global_class_ids.append(global_id)

    output_class_names = old_clip.get("class_names") or old_clip.get("classes")
    if not isinstance(output_class_names, (list, tuple)) or len(output_class_names) != num_local_classes:
        output_class_names = [classes[index] for index in global_class_ids]
    else:
        output_class_names = list(output_class_names)

    source_clip_few_local = subset_imagenet_source_support(
        source_clip_few, global_class_ids, num_local_classes, "CLIP source support"
    )
    source_visual_few_local = subset_imagenet_source_support(
        source_visual_few, global_class_ids, num_local_classes, "MoCo source support"
    )
    source_meta_local = subset_source_meta(source_meta, global_class_ids)

    clip_dir = os.path.join(args.output_root, args.task, args.clip_model)
    visual_dir = os.path.join(args.output_root, args.task, visual_model_name)
    mkdir(clip_dir); mkdir(os.path.join(clip_dir, "few_shots"))
    mkdir(visual_dir); mkdir(os.path.join(visual_dir, "few_shots"))
    filename = ood_cache_filename(args.dataset)

    protocol = {
        "type": "imagenet_cross_domain",
        "source_dataset": "ImageNet",
        "target_dataset": args.dataset,
        "query_clip_source": old_clip_path,
        "query_label_space": "original ECALP local labels",
        "source_features_task": getattr(args, "source_features_task", "fine_grained"),
        "source_paths": source_paths,
        "target_present_class_ids": sorted(
            set(torch.as_tensor(align_info["aligned_global_labels"]).long().tolist())
        ),
        "local_to_global_class_ids": global_class_ids,
        "target_support_used": False,
    }
    visual_out = {
        "image_features": visual_target,
        "labels": old_labels.clone(),
        "class_names": output_class_names,
        "classes": output_class_names,
        "align_info": align_info,
        "cross_domain_protocol": protocol,
    }
    torch.save(source_clip_few_local, os.path.join(clip_dir, "few_shots", filename))
    torch.save(visual_out, os.path.join(visual_dir, filename))
    torch.save(source_visual_few_local, os.path.join(visual_dir, "few_shots", filename))

    meta = {
        "dataset": args.dataset,
        "source_dataset": "ImageNet",
        "class_names": output_class_names,
        "source_global_class_ids": global_class_ids,
        "test_paths": target_paths,
        "few_paths": source_meta_local["few_paths"],
        "selected_indices": source_meta_local["selected_indices"],
        "num_shots": args.num_shots,
        "visual_encoder": args.visual_encoder,
        "visual_model_name": visual_model_name,
        "align_info": align_info,
        "cross_domain_protocol": protocol,
    }
    torch.save(meta, os.path.join(args.output_root, args.task, f"{args.dataset}_aligned_meta.pt"))
    validate_imagenet_ood_outputs(args, visual_model_name)

    print("=" * 80)
    print("Saved ImageNet cross-domain aligned features")
    print(f"Source support : ImageNet ({args.num_shots}-shot)")
    print(f"Target domain  : {args.dataset}")
    print(f"Target samples : {len(old_labels)}")
    print(f"Local classes  : {num_local_classes}")
    print(f"Old CLIP target: {old_clip_path} (unchanged)")
    print(f"MoCo target    : {os.path.join(visual_dir, filename)}")
    print(f"CLIP support   : {os.path.join(clip_dir, 'few_shots', filename)}")
    print(f"MoCo support   : {os.path.join(visual_dir, 'few_shots', filename)}")
    print("=" * 80)


def run_one_dataset(args):
    if args.dataset in IMAGENET_OOD_DATASETS:
        return run_imagenet_ood_dataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Check old ECALP/DMN reference before expensive extraction when requested.
    preflight_old_feature_check(args)

    classes, mp, cupl = load_cupl(args.cupl_json)

    # ImageNet must use the official classnames.txt order, not imagenet.json key order.
    # This keeps labels, text_features rows, CLIP, and MoCo aligned with ECALP/DMN.
    if args.dataset == "ImageNet":
        cfg = DATASET_LOCAL_PATHS.get(args.dataset, {})
        classnames_txt = cfg.get("classnames_txt", "")
        imagenet_classes, imagenet_mp, synset_to_idx = load_imagenet_classnames(classnames_txt)
        classes = imagenet_classes
        mp.update(imagenet_mp)
        print("=" * 80)
        print("[ImageNet] Class source override")
        print(f"CuPL json classes      : {len(cupl)}")
        print(f"Classnames.txt classes : {len(classes)}")
        print(f"Synset mapping classes : {len(synset_to_idx)}")
        print("Text feature rows will follow classnames.txt order, not imagenet.json key order.")
        print("=" * 80)

    C = len(classes)
    test = load_items(args, args.test_split, mp)
    few = load_items(args, args.fewshot_split, mp)
    test_paths, test_y = [x["path"] for x in test], [x["label"] for x in test]
    few_paths, few_y = [x["path"] for x in few], [x["label"] for x in few]

    cname = "RN50" if args.clip_model == "RN" else "ViT-B/16"
    loaded = clip.load(cname, device=device)
    if len(loaded) == 3: clip_model, _, clip_pre = loaded
    else: clip_model, clip_pre = loaded
    clip_model.eval()
    visual_model, visual_pre, visual_model_name = load_visual_encoder(args, device)

    clip_test, visual_test = encode_dual(
        test_paths, clip_model, clip_pre, visual_model, visual_pre,
        device, args.batch_size, f"Extract {args.dataset} test"
    )
    clip_few_all, visual_few_all = encode_dual(
        few_paths, clip_model, clip_pre, visual_model, visual_pre,
        device, args.batch_size, f"Extract {args.dataset} fewshot-pool"
    )
    # If requested, reorder both CLIP and visual features to the original ECALP/DMN feature order.
    # The same order is applied to MoCo/DINO, labels and paths.
    clip_test, visual_test, test_y, test_paths, align_info = align_test_features_to_old_order(
        args, clip_test, visual_test, test_y, test_paths
    )

    # ------------------------------------------------------------------
    # Text feature alignment
    # ------------------------------------------------------------------
    # When --align_to_old_order is enabled, image_features and labels are
    # reordered to the original ECALP/DMN .pt order. In that case, the safest
    # text feature source is also the old ECALP/DMN .pt file, because some
    # datasets such as Caltech101 do not store class_names in the old file and
    # the CuPL class order may differ from the old text_features row order.
    #
    # Without this block, we would have:
    #   image_features / labels -> old ECALP order
    #   text_features           -> newly generated CuPL order
    # which can make CLIP baseline collapse even though labels are aligned.
    if getattr(args, "align_to_old_order", False):
        ref_path = get_old_feature_reference_path(args)
        old_ref = safe_torch_load(ref_path, map_location="cpu") if os.path.exists(ref_path) else None

        if isinstance(old_ref, dict) and "text_features" in old_ref:
            txt = F.normalize(old_ref["text_features"].float(), dim=-1)

            print("=" * 80)
            print("[Text Alignment] Using old ECALP/DMN text_features")
            print(f"Reference path : {ref_path}")
            print(f"Text shape     : {tuple(txt.shape)}")
            print("=" * 80)

            if txt.shape[0] != C:
                print("=" * 80)
                print("[Warning] Old text_features rows != current class count")
                print(f"Old text rows  : {txt.shape[0]}")
                print(f"Current classes: {C}")
                print("Using old text feature row count as the class count for saving labels/text.")
                print("=" * 80)
                C = txt.shape[0]
        else:
            print("=" * 80)
            print("[Text Alignment] Old text_features not found; regenerate CuPL text_features")
            print(f"Reference path : {ref_path}")
            print("=" * 80)
            txt = text_features(clip_model, classes, cupl, device, args.text_batch_size)
    else:
        txt = text_features(clip_model, classes, cupl, device, args.text_batch_size)

    sel = select_few(few_y, C, args.num_shots, args.seed)
    clip_few, visual_few = gather(clip_few_all, sel), gather(visual_few_all, sel)
    y_few = onehot(sel, few_y, C)

    clip_dir = os.path.join(args.output_root, args.task, args.clip_model)
    visual_dir = os.path.join(args.output_root, args.task, visual_model_name)
    mkdir(clip_dir); mkdir(os.path.join(clip_dir, "few_shots"))
    mkdir(visual_dir); mkdir(os.path.join(visual_dir, "few_shots"))

    torch.save({"image_features": clip_test, "text_features": txt, "labels": torch.tensor(test_y).long(), "class_names": classes, "classes": classes, "align_info": align_info}, os.path.join(clip_dir, f"{args.dataset}.pt"))
    torch.save({"image_features": clip_few, "labels": y_few}, os.path.join(clip_dir, "few_shots", f"{args.dataset}.pt"))
    torch.save({"image_features": visual_test, "labels": torch.tensor(test_y).long(), "class_names": classes, "classes": classes, "align_info": align_info}, os.path.join(visual_dir, f"{args.dataset}.pt"))
    torch.save({"image_features": visual_few, "labels": y_few}, os.path.join(visual_dir, "few_shots", f"{args.dataset}.pt"))

    meta = {
        "dataset": args.dataset,
        "class_names": classes,
        "test_paths": test_paths,
        "few_paths": few_paths,
        "selected_indices": sel,
        "num_shots": args.num_shots,
        "visual_encoder": args.visual_encoder,
        "visual_model_name": visual_model_name,
        "align_info": align_info,
    }
    torch.save(meta, os.path.join(args.output_root, args.task, f"{args.dataset}_aligned_meta.pt"))

    print("=" * 80)
    print("Saved aligned features")
    print(f"CLIP test  : {os.path.join(clip_dir, f'{args.dataset}.pt')}")
    print(f"Visual test: {os.path.join(visual_dir, f'{args.dataset}.pt')}")
    print(f"Visual name: {visual_model_name}")
    print(f"CLIP test image_features  : {tuple(clip_test.shape)}")
    print(f"CLIP text_features        : {tuple(txt.shape)}")
    print(f"CLIP few image_features   : {tuple(clip_few.shape)}")
    print(f"Visual test image_features: {tuple(visual_test.shape)}")
    print(f"Visual few image_features : {tuple(visual_few.shape)}")
    print("=" * 80)


def get_cupl_json_for_dataset(dataset_name, cupl_root=None):
    """Return CuPL json path for one dataset.

    This follows the same candidate-file style as the Joint Dual Graph
    search script: try several common names under --cupl_root and use
    the first existing file.
    """
    root = cupl_root or CUPL_ROOT
    names = DATASET_CUPL_CANDIDATES.get(dataset_name, [])
    if not names and dataset_name in DATASET_CUPL_JSON:
        names = [DATASET_CUPL_JSON[dataset_name]]

    tried = []
    for name in names:
        path = os.path.join(root, name)
        tried.append(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"CuPL json not found for {dataset_name}. Tried:\n  " + "\n  ".join(tried)
    )


def build_args_for_dataset(dataset_name, cli_args=None):
    """Build argparse-like args from RUN_DATASETS, while respecting CLI overrides.

    Important:
      In list mode, command-line arguments such as
          --visual_encoder mocov3_vit
      must still take effect. Therefore this function copies the shared
      extraction settings from cli_args when cli_args is provided.
    """
    if dataset_name not in DATASET_LOCAL_PATHS:
        raise KeyError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(DATASET_LOCAL_PATHS.keys())}"
        )

    args = argparse.Namespace()
    args.dataset = dataset_name

    # Dataset path options are dataset-specific in list mode.
    # Keep them empty so DATASET_LOCAL_PATHS is used.
    args.dataset_root = ""
    args.image_root = ""
    args.split_json = ""

    # ImageNet usually uses val as the evaluation split.
    args.test_split = "val" if dataset_name == "ImageNet" else "test"
    args.fewshot_split = "train"

    # Shared settings: prefer CLI values, fall back to global config.
    requested_task = getattr(cli_args, "task", None) if cli_args is not None else None
    args.task = requested_task or ("style_transfer" if dataset_name in IMAGENET_OOD_DATASETS else TASK)
    args.output_root = getattr(cli_args, "features_path", None) or (getattr(cli_args, "output_root", OUTPUT_ROOT) if cli_args is not None else OUTPUT_ROOT)
    args.clip_model = getattr(cli_args, "clip_model", CLIP_MODEL) if cli_args is not None else CLIP_MODEL

    args.visual_encoder = getattr(cli_args, "visual_encoder", VISUAL_ENCODER) if cli_args is not None else VISUAL_ENCODER
    args.visual_model_name = getattr(cli_args, "visual_model_name", VISUAL_MODEL_NAME) if cli_args is not None else VISUAL_MODEL_NAME
    args.visual_image_size = getattr(cli_args, "visual_image_size", VISUAL_IMAGE_SIZE) if cli_args is not None else VISUAL_IMAGE_SIZE

    args.dino_model = getattr(cli_args, "dino_model", "DINO") if cli_args is not None else "DINO"
    args.dino_ckpt = getattr(cli_args, "dino_ckpt", DINO_CKPT) if cli_args is not None else DINO_CKPT
    args.dino_size = getattr(cli_args, "dino_size", DINO_SIZE) if cli_args is not None else DINO_SIZE

    # MoCo weights are separated.
    # mocov3_r50 -> args.mocov3_ckpt
    # mocov3_vit -> args.mocov3_vit_ckpt
    args.mocov3_ckpt = getattr(cli_args, "mocov3_ckpt", MOCO_CKPT) if cli_args is not None else MOCO_CKPT
    args.mocov3_vit_ckpt = getattr(cli_args, "mocov3_vit_ckpt", MOCO_VIT_CKPT) if cli_args is not None else MOCO_VIT_CKPT

    args.cupl_json = get_cupl_json_for_dataset(dataset_name, getattr(cli_args, "cupl_root", CUPL_ROOT) if cli_args is not None else CUPL_ROOT)

    args.num_shots = getattr(cli_args, "num_shots", NUM_SHOTS) if cli_args is not None else NUM_SHOTS
    args.seed = getattr(cli_args, "seed", SEED) if cli_args is not None else SEED
    args.batch_size = getattr(cli_args, "batch_size", BATCH_SIZE) if cli_args is not None else BATCH_SIZE
    args.text_batch_size = getattr(cli_args, "text_batch_size", TEXT_BATCH_SIZE) if cli_args is not None else TEXT_BATCH_SIZE
    args.max_images = getattr(cli_args, "max_images", -1) if cli_args is not None else -1
    args.cpu = getattr(cli_args, "cpu", CPU) if cli_args is not None else CPU
    requested_alignment = getattr(cli_args, "align_to_old_order", None) if cli_args is not None else None
    args.align_to_old_order = (
        dataset_name in IMAGENET_OOD_DATASETS
        if requested_alignment is None
        else bool(requested_alignment)
    )
    args.old_features_root = getattr(cli_args, "old_features_root", OLD_FEATURES_ROOT) if cli_args is not None else OLD_FEATURES_ROOT
    args.source_features_root = getattr(cli_args, "source_features_root", "") if cli_args is not None else ""
    args.source_features_task = getattr(cli_args, "source_features_task", "fine_grained") if cli_args is not None else "fine_grained"
    return args


def run_from_dataset_list(cli_args=None):
    """Run all datasets listed in RUN_DATASETS.

    Unlike the old version, this version respects command-line arguments
    in list mode. Example:
        python extract_aligned_clip_moco_features.py --visual_encoder mocov3_vit
    will really use MoCo v3 ViT-B/16 and the ViT checkpoint.
    """
    datasets_to_run = expand_dataset_arg(getattr(cli_args, "datasets", "")) if cli_args is not None and getattr(cli_args, "datasets", "") else list(RUN_DATASETS)
    if not datasets_to_run:
        raise RuntimeError("No dataset is selected. Use --datasets Aircraft/Cars/DTD or edit RUN_DATASETS.")

    visual_encoder = getattr(cli_args, "visual_encoder", VISUAL_ENCODER) if cli_args is not None else VISUAL_ENCODER
    visual_model_name = getattr(cli_args, "visual_model_name", VISUAL_MODEL_NAME) if cli_args is not None else VISUAL_MODEL_NAME
    output_root = getattr(cli_args, "features_path", None) or (getattr(cli_args, "output_root", OUTPUT_ROOT) if cli_args is not None else OUTPUT_ROOT)
    num_shots = getattr(cli_args, "num_shots", NUM_SHOTS) if cli_args is not None else NUM_SHOTS
    moco_r50_ckpt = getattr(cli_args, "mocov3_ckpt", MOCO_CKPT) if cli_args is not None else MOCO_CKPT
    moco_vit_ckpt = getattr(cli_args, "mocov3_vit_ckpt", MOCO_VIT_CKPT) if cli_args is not None else MOCO_VIT_CKPT

    print("=" * 80)
    print("Run from simple RUN_DATASETS list")
    print("=" * 80)
    print(f"DATASETS          : {datasets_to_run}")
    print(f"Visual encoder    : {visual_encoder}")
    print(f"Visual model name : {visual_model_name}")
    print(f"Output root       : {output_root}")
    print(f"Num shots         : {num_shots}")
    if visual_encoder in ["mocov3", "mocov3_r50", "r50"]:
        print(f"MoCo R50 ckpt     : {moco_r50_ckpt}")
        print("Save visual dir   : MOCO/R50")
    elif visual_encoder in ["mocov3_vit", "mocov3_vitb", "vit"]:
        print(f"MoCo ViT ckpt     : {moco_vit_ckpt}")
        print("Save visual dir   : MOCO/VIT")
    print("=" * 80)

    failed = []
    for dataset_name in datasets_to_run:
        print("\n" + "#" * 100)
        print(f"# Running dataset: {dataset_name}")
        print("#" * 100)

        try:
            args = build_args_for_dataset(dataset_name, cli_args=cli_args)

            if args.visual_encoder == "dino" and not args.dino_ckpt:
                raise ValueError("DINO_CKPT is required when VISUAL_ENCODER='dino'")
            if args.visual_encoder in ["mocov3", "mocov3_r50", "r50"] and not args.mocov3_ckpt:
                raise ValueError("MOCO_CKPT is required when VISUAL_ENCODER='mocov3_r50'")
            if args.visual_encoder in ["mocov3_vit", "mocov3_vitb", "vit"] and not args.mocov3_vit_ckpt:
                raise ValueError("MOCO_VIT_CKPT is required when VISUAL_ENCODER='mocov3_vit'")

            run_one_dataset(args)

        except Exception as e:
            failed.append((dataset_name, repr(e)))
            print("=" * 80)
            print(f"[Error] Dataset failed: {dataset_name}")
            print(repr(e))
            print("=" * 80)

            if not CONTINUE_ON_ERROR:
                raise

    print("\n" + "=" * 80)
    print("Batch extraction finished")
    print("=" * 80)
    if failed:
        print("Failed datasets:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        if any(name in IMAGENET_OOD_DATASETS for name, _ in failed):
            details = "\n".join(f"  {name}: {err}" for name, err in failed)
            raise RuntimeError("ImageNet OOD extraction/alignment failed:\n" + details)
    else:
        print("All datasets succeeded.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Universal aligned CLIP + DINO/MoCo feature extractor with simple RUN_DATASETS list"
    )
    parser.add_argument(
        "--single_dataset",
        action="store_true",
        help="Use CLI arguments to run only one dataset. Otherwise RUN_DATASETS at the top is used.",
    )

    # Single-dataset CLI mode, kept for compatibility.
    parser.add_argument("--datasets", type=str, default="",
                        help="Same as search script: Aircraft/Cars/DTD, ALL, or ALL_EXCEPT_CARS_IMAGENET.")
    parser.add_argument("--dataset", type=str, default="Aircraft", choices=list(DATASET_LOCAL_PATHS.keys()))
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Output task directory. Default: style_transfer for ImageNet OOD, fine_grained otherwise.",
    )
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--split_json", type=str, default="")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--fewshot_split", type=str, default="train")
    parser.add_argument("--features_path", type=str, default=OUTPUT_ROOT,
                        help="Same meaning as the search script --features_path. Extracted features are saved here.")
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT,
                        help="Backward-compatible alias. --features_path has priority.")
    parser.add_argument("--clip_model", type=str, default=CLIP_MODEL, choices=["RN", "VIT"])
    parser.add_argument("--visual_encoder", type=str, default=VISUAL_ENCODER, choices=["dino", "mocov3", "mocov3_r50", "mocov3_vit", "mocov3_vitb", "r50", "vit"])
    parser.add_argument("--visual_model_name", type=str, default=VISUAL_MODEL_NAME)
    parser.add_argument("--visual_image_size", type=int, default=VISUAL_IMAGE_SIZE)

    parser.add_argument("--dino_model", type=str, default="DINO")
    parser.add_argument("--dino_ckpt", type=str, default=DINO_CKPT)
    parser.add_argument("--dino_size", type=str, default=DINO_SIZE, choices=["s", "b", "l"])

    parser.add_argument("--mocov3_ckpt", type=str, default=MOCO_CKPT)
    parser.add_argument("--mocov3_vit_ckpt", type=str, default=MOCO_VIT_CKPT)
    parser.add_argument("--cupl_root", type=str, default=CUPL_ROOT,
                        help="Same as search script --cupl_root. Dataset-specific CuPL json is auto-resolved here.")
    parser.add_argument("--cupl_json", type=str, default="")
    parser.add_argument("--num_shots", type=int, default=NUM_SHOTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--text_batch_size", type=int, default=TEXT_BATCH_SIZE)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--align_to_old_order", action=argparse.BooleanOptionalAction, default=None,
                        help="Match original ECALP/DMN row order. Default: enabled for ImageNet OOD.")
    parser.add_argument("--old_features_root", type=str, default=OLD_FEATURES_ROOT,
                        help="Root of original ECALP/DMN features used as order reference.")
    parser.add_argument("--source_features_root", type=str, default="",
                        help="Root containing ImageNet source support/text caches. Default: --features_path.")
    parser.add_argument("--source_features_task", type=str, default="fine_grained",
                        help="Task directory containing ImageNet source caches used by OOD targets.")

    args = parser.parse_args()
    args.output_root = args.features_path or args.output_root

    if args.single_dataset:
        if args.task is None:
            args.task = "style_transfer" if args.dataset in IMAGENET_OOD_DATASETS else TASK
        if args.align_to_old_order is None:
            args.align_to_old_order = args.dataset in IMAGENET_OOD_DATASETS
        if not args.cupl_json:
            args.cupl_json = get_cupl_json_for_dataset(args.dataset, args.cupl_root)
        if args.visual_encoder == "dino" and not args.dino_ckpt:
            raise ValueError("--dino_ckpt is required when --visual_encoder dino")
        if args.visual_encoder in ["mocov3", "mocov3_r50", "r50"] and not args.mocov3_ckpt:
            raise ValueError("--mocov3_ckpt is required when --visual_encoder mocov3_r50")
        if args.visual_encoder in ["mocov3_vit", "mocov3_vitb", "vit"] and not args.mocov3_vit_ckpt:
            raise ValueError("--mocov3_vit_ckpt is required when --visual_encoder mocov3_vit")
        run_one_dataset(args)
    else:
        run_from_dataset_list(args)
