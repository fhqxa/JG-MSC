# MSC: Multi-Space Consensus.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import clip

from config import *
from common import *
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
