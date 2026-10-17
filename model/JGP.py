# JGP: Joint Graph Propagation.
import os
import torch
import torch.nn.functional as F

try:
    import networkx as nx
except ImportError:
    nx = None

from config import *
from common import *
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
    if not (args.use_dino_cache or args.use_dino_clique or getattr(args, "use_joint_dual_graph", False) or getattr(args, "use_cross_graph", False)):
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
            "The MoCo query cache is not aligned to the original ECALP query cache"
            f"{detail}. CLIP={clip_path}, MoCo={moco_path}"
        )

def validate_aligned_clip_reference(args, dataset_name, original_features, original_labels):
    """Verify that the aligned visual cache follows the original ECALP row order.

    The jointly extracted aligned CLIP file is not used for prediction here.
    It acts as an image-level alignment witness: if its row-wise CLIP cosine to
    the original ECALP cache is high, the paired MoCo rows can safely be fused
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
    print("Original ECALP / aligned CLIP image-order verification")
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
            "The aligned CLIP cache is not in the same image order as the original ECALP cache "
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

def compute_joint_gate(text_mask, dino_mask, args):
    if args.joint_gate_mode == "union":
        joint_gate = torch.clamp(text_mask + dino_mask, 0.0, 1.0)
    elif args.joint_gate_mode == "intersection":
        joint_gate = text_mask * dino_mask
        if joint_gate.sum().item() < 1:
            joint_gate = torch.clamp(text_mask + dino_mask, 0.0, 1.0)
    elif args.joint_gate_mode == "soft":
        joint_gate = float(args.joint_text_gate) * text_mask + float(args.joint_dino_gate) * dino_mask
        joint_gate = joint_gate / joint_gate.max().clamp(min=1e-6)
    else:
        raise ValueError(f"Unknown joint_gate_mode: {args.joint_gate_mode}")
    return joint_gate

def compute_jgp_score(lp_score, text_clique_score, dino_clique_score, text_mask, dino_mask, args):
    """Joint semantic-visual graph guidance.

    The Qwen/text clique graph and the DINO cache clique graph jointly build a
    community gate. This branch is different from late score fusion: the two
    graphs first guide the same prediction distribution through a shared class
    gate, then the guided distribution is fused with other branches.
    """
    lp_prob = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
    text_prob = torch.softmax(float(args.text_clique_scale) * text_clique_score.float(), dim=1)
    dino_prob = torch.softmax(float(args.dino_clique_scale) * dino_clique_score.float(), dim=1)

    joint_gate = compute_joint_gate(text_mask, dino_mask, args)

    text_graph_w = max(0.0, min(1.0, float(getattr(args, "joint_graph_text_weight", 0.5))))
    graph_prob = text_graph_w * text_prob + (1.0 - text_graph_w) * dino_prob
    joint_prob = (1.0 - float(args.joint_inner_weight)) * lp_prob + float(args.joint_inner_weight) * graph_prob
    joint_prob = joint_prob * (1.0 + float(args.joint_mask_boost) * joint_gate)
    joint_prob = joint_prob / joint_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return joint_prob

def compute_single_graph_guided_score(lp_score, graph_score, graph_mask, args, graph_scale=1.0):
    """Single visual graph guided prediction.

    This is the standalone LP + MoCo visual graph branch. It follows the
    same internal design as the single-graph ablation in the joint dual graph:

        1) Convert LP score and visual graph score to probabilities.
        2) Mix LP probability and graph probability using joint_inner_weight.
        3) Use the visual clique mask as a gate to boost selected classes.
        4) Renormalize to a probability distribution.

    It is intentionally not the same as directly fusing LP with dino_clique_score.
    """
    lp_prob = torch.softmax(float(args.lp_scale) * lp_score.float(), dim=1)
    graph_prob = torch.softmax(float(graph_scale) * graph_score.float(), dim=1)

    graph_gate = graph_mask.float()
    if graph_gate.max().item() > 0:
        graph_gate = graph_gate / graph_gate.max().clamp(min=1e-6)

    guided_prob = (1.0 - float(args.joint_inner_weight)) * lp_prob + float(args.joint_inner_weight) * graph_prob
    guided_prob = guided_prob * (1.0 + float(args.joint_mask_boost) * graph_gate)
    guided_prob = guided_prob / guided_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return guided_prob

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
    """Mutual propagation between ECLAP/CLIP class graph and DINO visual class graph.

    We do not mix feature dimensions directly. Instead, both branches are first
    represented as class-level distributions, then we propagate class confidence
    over a two-layer graph:

        [ CLIP class graph      cross identity ]
        [ cross identity       DINO class graph]

    This makes ECLAP scores and DINO scores refine each other through same-label
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

    # Block adjacency for two score layers: CLIP/ECLAP nodes and DINO nodes.
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
