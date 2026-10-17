# AKI: Attribute Knowledge Integration.
from collections import defaultdict

import torch
import torch.nn.functional as F
import clip

from config import *
from common import *
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
