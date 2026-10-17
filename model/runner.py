# Experiment runner, ablations, sensitivity analysis, and result export.
import os
import csv
import time
import traceback
from datetime import datetime
from itertools import product
from collections import defaultdict
from types import SimpleNamespace

import torch
import clip

from config import *
from common import *
from aki import *
from msc import *
from jgp import *
from evaluation import *
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
        3) joint_dual_weight

    For shot=0, JGP cannot be built because there are no few-shot
    DINO prototypes, so we run LP baseline once for each text_mode.
    """
    fixed = search_config["fixed"]
    for shot in search_config["shot"]:
        for text_mode in search_config["text_mode"]:
            weights = [0.0] if int(shot) == 0 else search_config["joint_dual_weight"]
            for joint_w in weights:
                hp = dict(fixed)
                hp["joint_dual_weight"] = float(joint_w)
                branch_mode = "lp" if int(shot) == 0 else "joint_dual_graph"
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
        return ["baseline", "joint", "msc", "joint_msc"]
    aliases = {
        "base": "baseline",
        "lp": "baseline",
        "qwen": "aki",
        "baseline_qwen": "aki",
        "baseline_aki": "aki",
        "baseline_joint": "joint",
        "baseline_msc": "msc",
        "baseline_msc_qwen": "msc",
        "msc_qwen": "msc",
        "joint+msc": "joint_msc",
        "joint_msc_qwen": "joint_msc",
        "aki+joint+msc": "full",
        "joint+msc+aki": "full",
        "full_qwen": "full",
    }
    valid = {"baseline", "aki", "joint", "msc", "joint_msc", "full"}
    out = []
    for part in str(text).replace("/", ",").split(","):
        item = part.strip().lower()
        if not item:
            continue
        item = aliases.get(item, item)
        if item not in valid:
            raise ValueError(
                f"Unknown --modules item: {part.strip()}. "
                "Choose from baseline, aki, joint, msc, joint_msc, full."
            )
        if item not in out:
            out.append(item)
    return out or ["baseline", "joint", "msc", "joint_msc"]

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
    alpha_part = "searchalpha" if getattr(cli_args, "search_alpha", False) else f"alpha{float(getattr(cli_args, 'joint_graph_text_weight', 0.5)):g}"
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
        "use_joint_dual_graph": False,
        "use_cross_graph": False,
    }

    if branch_mode == "lp":
        return flags
    if branch_mode == "joint_dual_graph":
        flags["use_joint_dual_graph"] = True
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
        search_joint_weight=getattr(cli_args, "search_joint_weight", False),
        joint_weight_grid=getattr(
            cli_args,
            "joint_weight_grid",
            ",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["w"]),
        ),
        search_alpha_weight_matrix=getattr(cli_args, "search_alpha_weight_matrix", True),
        joint_graph_text_weight=getattr(cli_args, "joint_graph_text_weight", 0.5),
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
    if args.mode == "ZS" and (args.use_dino_cache or args.use_dino_clique or args.use_joint_dual_graph or getattr(args, "use_cross_graph", False)):
        return False, "DINO cache/clique/joint modes require FS support features"

    if (args.use_dino_cache or args.use_dino_clique or args.use_joint_dual_graph or getattr(args, "use_cross_graph", False)) and not args.dino_features_path:
        # If not specified, code falls back to features_path/task/DINO, so this is still valid.
        return True, ""

    return True, ""

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
            joint = base_item.get("baseline_joint")
            full_item = source_items.get(attribute_text_source, {})
            full_model = full_item.get("qwen_joint_patch")
            baseline_external = base_item.get("baseline_external")

            row = {
                "dataset": dataset_name,
                "shot": shot,
                "baseline": fmt(baseline),
                "baseline_joint": fmt(joint),
                f"aki_joint_msc_{attribute_text_source}": fmt(full_model),
                "joint_raw": fmt(base_item.get("baseline_joint_raw_patch", base_item.get("baseline_joint_raw"))),
                "moco_visual_raw": fmt(base_item.get("lp_moco_visual_raw_patch", base_item.get("lp_moco_visual_raw"))),
                "gain_joint": fmt(None if baseline is None or joint is None else joint - baseline),
                f"gain_aki_joint_msc_{attribute_text_source}": fmt(
                    None if baseline is None or full_model is None else full_model - baseline
                ),
                "best_weight_joint": best_weight(base_item, "best_weight_baseline_joint"),
                f"best_weight_aki_joint_msc_{attribute_text_source}": best_weight(
                    full_item,
                    "best_weight_qwen_joint_patch",
                ),
                "joint_graph_text_weight": fmt(base_item.get("joint_graph_text_weight", None)),
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
                joint_msc = item.get("baseline_joint_patch")
                row[f"baseline_msc_{source}"] = fmt(msc)
                row[f"joint_msc_{source}"] = fmt(joint_msc)
                row[f"gain_msc_{source}"] = fmt(None if baseline is None or msc is None else msc - baseline)
                row[f"gain_joint_msc_{source}"] = fmt(None if baseline is None or joint_msc is None else joint_msc - baseline)
                row[f"best_weight_joint_msc_{source}"] = best_weight(item, "best_weight_baseline_joint_patch")
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
    # external ECALP table.  Do not substitute the external number.
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
                    "the external ECALP reference. Check base_text_prompt, feature "
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
    """Save per-method timing and a direct comparison against ECALP.

    The raw measurements are kept unchanged.  When an ECALP/baseline row is
    available for the same dataset and shot, the derived columns make the
    JG-MSC overhead directly usable in the computational-efficiency table.
    """
    baseline_by_dataset_shot = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method == "ECALP" or (
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
            0 if str(item.get("method", "")) == "ECALP" else 1,
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
    """Print the same ECALP/JG-MSC comparison saved by write_timing_csv."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["shot"]))].append(row)

    if not grouped:
        return

    print("=" * 100)
    print("RUNTIME COMPARISON: ORIGINAL ECALP VS JG-MSC")
    print("=" * 100)
    for (dataset_name, shot), group in sorted(grouped.items()):
        baseline = next((row for row in group if row.get("method") == "ECALP"), None)
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
                f"| {ratio:.3f}x ECALP"
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
    """Run AKI / Joint Graph / MSC ablations and save a compact CSV.

    Final compact columns:
        1) baseline                 = Original text + LP
        2) baseline_aki             = AKI / Qwen auxiliary node + LP
        3) baseline_joint           = Original text + Joint Graph
        4) baseline_msc_qwen        = Original text + MSC patch nodes selected by Qwen attributes
        5) joint_msc_qwen           = Original text + Joint Graph + MSC selected by Qwen attributes

    Notes:
        - For shot=0, JGP cannot be built because DINO few-shot prototypes do not exist.
          Therefore Joint columns fall back to the LP-side result.
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
        #   1) original ECALP (cached text features + LP only),
        #   2) complete JG-MSC (AKI + MSC + JGP) at fixed alpha/w/tau_K.
        # Sensitivity searches are disabled because they must not inflate the
        # reported inference time or replace the fixed-parameter accuracy.
        cli_args.save_timing = True
        cli_args.run_sensitivity = False
        cli_args.search_alpha = False
        cli_args.search_joint_weight = False
        cli_args.search_alpha_weight_matrix = False
        cli_args.modules = "baseline,full"
        cli_args.patch_variant = "on"
        cli_args.support_grounded_aki = True
        # Fixed paper timing configuration.  Keep the one-command runtime
        # comparison simple and prevent accidental per-run parameter changes.
        cli_args.joint_graph_text_weight = 0.5
        cli_args.runtime_compare_w = 0.9
        cli_args.runtime_compare_tau_k = 0.0
        cli_args.timing_warmup_queries = 5

    if bool(getattr(cli_args, "run_sensitivity", False)):
        # Sensitivity is evaluated on the complete JG-MSC model.  The fixed
        # values are used for one-at-a-time curves; the two-dimensional grid
        # remains available as a diagnostic in the normal result object.
        cli_args.search_alpha = True
        cli_args.search_joint_weight = True
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
        "baseline_joint",
        f"aki_joint_msc_{attribute_text_source}",
        "joint_raw",
        "moco_visual_raw",
        "gain_joint",
        f"gain_aki_joint_msc_{attribute_text_source}",
        "best_weight_joint",
        f"best_weight_aki_joint_msc_{attribute_text_source}",
        "joint_graph_text_weight",
        "baseline_external",
    ]
    if aki_enabled and not compare_support_grounding:
        fieldnames.insert(fieldnames.index("baseline_joint"), "baseline_aki")
        fieldnames.insert(fieldnames.index("gain_joint"), "gain_aki")
    if compare_support_grounding:
        aki_insert = fieldnames.index("baseline_joint")
        for name in ["baseline_aki_unfiltered", "baseline_aki_grounded"]:
            fieldnames.insert(aki_insert, name)
            aki_insert += 1
        gain_insert = fieldnames.index("gain_joint")
        for name in ["gain_aki_unfiltered", "gain_aki_grounded", "aki_attributes_total", "aki_attributes_retained", "aki_retention_rate"]:
            fieldnames.insert(gain_insert, name)
            gain_insert += 1
    insert_at = 5
    for source in requested_patch_text_sources:
        fieldnames.insert(insert_at, f"baseline_msc_{source}")
        insert_at += 1
        fieldnames.insert(insert_at, f"joint_msc_{source}")
        insert_at += 1
    gain_insert_at = fieldnames.index("best_weight_joint")
    for source in requested_patch_text_sources:
        fieldnames.insert(gain_insert_at, f"gain_msc_{source}")
        gain_insert_at += 1
        fieldnames.insert(gain_insert_at, f"gain_joint_msc_{source}")
        gain_insert_at += 1
    weight_insert_at = fieldnames.index("joint_graph_text_weight")
    for source in requested_patch_text_sources:
        fieldnames.insert(weight_insert_at, f"best_weight_joint_msc_{source}")
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
    requested_joint_dual_weights = parse_float_list(getattr(cli_args, "joint_dual_weights", ""))
    if requested_joint_dual_weights:
        search_config["joint_dual_weight"] = requested_joint_dual_weights
    if runtime_compare:
        search_config["joint_dual_weight"] = [
            float(getattr(cli_args, "runtime_compare_w", 0.9))
        ]
    elif getattr(cli_args, "run_sensitivity", False):
        # alpha and w are swept internally only once on the full model at the
        # reference tau_K.  All outer ablation/tau_K runs therefore use the
        # fixed paper value instead of creating another weight-search round.
        search_config["joint_dual_weight"] = [
            float(getattr(cli_args, "sensitivity_fixed_w", 0.9))
        ]
    elif getattr(cli_args, "search_joint_weight", False):
        # The internal search determines the reported best w.  Keep the outer
        # run at the alpha-search fixed w instead of 0.0 so diagnostic branches
        # (e.g. LP+MoCoVisual) do not silently collapse to pure LP.
        search_config["joint_dual_weight"] = [
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
    if any(m in requested_modules for m in ["aki", "joint", "msc", "joint_msc", "full"]) and "baseline" not in requested_modules:
        requested_modules = ["baseline"] + requested_modules
    print(f"Requested modules: {','.join(requested_modules)}")

    # Controlled module plan:
    #   baseline, AKI, Joint, MSC(Qwen), Joint+MSC(Qwen).
    # Baseline/Joint share the no-patch original run; MSC/Joint+MSC share the
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
    lp_aki_only = not any(m in requested_modules for m in ["joint", "msc", "joint_msc", "full"])
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
            config["params"]["joint_dual_weight"] = 0.0
        shot = int(config["shot"])
        branch_mode = config["branch_mode"]
        base_source = requested_patch_text_sources[0]
        if "baseline" in requested_modules or "joint" in requested_modules:
            if runtime_compare:
                baseline_config = dict(config)
                baseline_config["branch_mode"] = "lp"
                baseline_config["params"] = dict(config["params"])
                baseline_config["params"]["joint_dual_weight"] = 0.0
                planned_runs.append(
                    (baseline_config, "original", base_source, False, "baseline", None, None)
                )
            else:
                planned_runs.append((config, "original", base_source, False, "baseline_joint" if branch_mode == "joint_dual_graph" else "baseline", None, None))
        if "aki" in requested_modules:
            if compare_support_grounding:
                planned_runs.append((config, aki_text_mode, base_source, False, "baseline_aki_unfiltered", False, None))
                for threshold in thresholds_for_current_run(shot):
                    planned_runs.append((config, aki_text_mode, base_source, False, "baseline_aki_grounded", True, threshold))
            else:
                thresholds = thresholds_for_current_run(shot) if getattr(cli_args, "support_grounded_aki", False) else [None]
                for threshold in thresholds:
                    planned_runs.append((config, aki_text_mode, base_source, False, "baseline_aki", None, threshold))
        if shot > 0 and ("msc" in requested_modules or "joint_msc" in requested_modules):
            for source in requested_patch_text_sources:
                ablation_name = f"joint_msc_{source}" if branch_mode == "joint_dual_graph" else f"baseline_msc_{source}"
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
                    f"full_aki_joint_msc_{attribute_text_source}",
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
            hp.setdefault("joint_graph_text_weight", float(getattr(cli_args, "joint_graph_text_weight", 0.5)))
            joint_w = float(hp.get("joint_dual_weight", 0.0))

            if patch_variant == "off" and use_patch:
                continue
            if patch_variant == "on" and not use_patch and ablation_name not in {"baseline", "baseline_aki", "baseline_joint"}:
                continue

            run_id += 1
            dataset_cli_args = SimpleNamespace(**vars(cli_args))
            dataset_cli_args.prompt_json = dataset_prompt_json
            dataset_cli_args.cupl_json = dataset_cupl_json
            dataset_cli_args.meta_file = dataset_meta_file
            dataset_cli_args.patch_text_source = patch_text_source
            dataset_cli_args.use_support_patch_nodes = bool(use_patch)

            # Controlled ablation must start from exactly the same original
            # graph in every row.  By default we use the cached ECALP class
            # text features, which is also the protocol expected by the
            # external ECALP baseline table (e.g. ImageNet 16-shot 63.20).
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
            if branch_mode == "joint_dual_graph" and bool(
                getattr(cli_args, "tune_jgp_per_ablation", True)
            ):
                run_args.search_alpha = True
                run_args.search_joint_weight = True
                run_args.search_alpha_weight_matrix = True
            elif getattr(cli_args, "run_sensitivity", False):
                internal_sensitivity_run = (
                    ablation_name == f"full_aki_joint_msc_{attribute_text_source}"
                    and aki_threshold is not None
                    and abs(float(aki_threshold) - sensitivity_reference_tau) < 1e-12
                )
                run_args.search_alpha = bool(internal_sensitivity_run)
                run_args.search_joint_weight = bool(internal_sensitivity_run)
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
                f"joint_dual_weight={joint_w} "
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

                selected_alpha = result.get("best_joint_alpha")
                if selected_alpha is None:
                    selected_alpha = result.get("best_alpha")
                if selected_alpha is None:
                    selected_alpha = float(getattr(run_args, "joint_graph_text_weight", 0.5))
                selected_w = result.get("best_joint_weight")
                if selected_w is None:
                    selected_w = float(getattr(run_args, "joint_dual_weight", 0.0))

                aki_component = text_mode in {"cupl_aux_node", "qwen_aux_node"}
                msc_component = bool(use_patch)
                jgp_component = branch_mode == "joint_dual_graph"

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
                        timing_method = "ECALP"
                    elif runtime_compare and ablation_name == f"full_aki_joint_msc_{attribute_text_source}":
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
                            float(getattr(run_args, "joint_graph_text_weight", 0.5))
                            if jgp_component else ""
                        ),
                        "w": (
                            float(getattr(run_args, "joint_dual_weight", 0.0))
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
                    and ablation_name == f"full_aki_joint_msc_{attribute_text_source}"
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
                        "baseline_joint": None,
                        "baseline_joint_patch": None,
                        "baseline_joint_raw": None,
                        "baseline_joint_raw_patch": None,
                        "qwen_joint": None,
                        "qwen_joint_patch": None,
                        "lp_moco_visual": None,
                        "lp_moco_visual_patch": None,
                        "lp_moco_visual_raw": None,
                        "lp_moco_visual_raw_patch": None,
                        "best_weight_lp_moco_visual": None,
                        "best_weight_lp_moco_visual_patch": None,
                        "best_weight_baseline_joint": None,
                        "best_weight_baseline_joint_patch": None,
                        "best_weight_qwen_joint": None,
                        "best_weight_qwen_joint_patch": None,
                        "joint_graph_text_weight": getattr(run_args, "joint_graph_text_weight", None),
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

                if text_mode == "original" and branch_mode == "joint_dual_graph":
                    best_joint_w = result.get("best_joint_weight")
                    stored_joint_w = best_joint_w if best_joint_w is not None else joint_w
                    improved = update_best(
                        item,
                        f"baseline_joint{patch_suffix}",
                        result["acc_final"],
                        f"best_weight_baseline_joint{patch_suffix}",
                        stored_joint_w,
                    )
                    if improved:
                        if result.get("best_joint_alpha") is not None:
                            item["joint_graph_text_weight"] = result["best_joint_alpha"]
                        elif result.get("best_alpha") is not None:
                            item["joint_graph_text_weight"] = result["best_alpha"]
                    update_best(item, f"baseline_joint_raw{patch_suffix}", result.get("acc_joint_dual"))
                    visual_acc = result.get("acc_lp_moco_visual", 0.0)
                    update_best(
                        item,
                        f"lp_moco_visual{patch_suffix}",
                        visual_acc,
                        f"best_weight_lp_moco_visual{patch_suffix}",
                        stored_joint_w,
                    )
                    update_best(item, f"lp_moco_visual_raw{patch_suffix}", result.get("acc_lp_moco_visual"))

                if text_mode in {"cupl_aux_node", "qwen_aux_node"} and branch_mode == "joint_dual_graph":
                    best_joint_w = result.get("best_joint_weight")
                    stored_joint_w = best_joint_w if best_joint_w is not None else joint_w
                    improved = update_best(
                        item,
                        f"qwen_joint{patch_suffix}",
                        result["acc_final"],
                        f"best_weight_qwen_joint{patch_suffix}",
                        stored_joint_w,
                    )
                    if improved:
                        if result.get("best_joint_alpha") is not None:
                            item["joint_graph_text_weight"] = result["best_joint_alpha"]
                        elif result.get("best_alpha") is not None:
                            item["joint_graph_text_weight"] = result["best_alpha"]

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
                if result.get("best_joint_weight") is not None:
                    best_alpha_msg += f" | best_w={float(result['best_joint_weight']):.2f}"
                print(
                    f"[DONE-{tag}] ablation={ablation_name} | text={text_mode} | mode={branch_mode} | "
                    f"LP={result['acc_lp']:.2f} | "
                    f"JointRaw={result['acc_joint_dual']:.2f} | "
                    f"JointFinal={result['acc_final']:.2f} | "
                    f"LP+MoCoVisual={result.get('acc_lp_moco_visual', 0.0):.2f} | "
                    f"Final={result['acc_final']:.2f} | "
                    f"weight={joint_w:.1f}"
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
                    f"patch={'on' if use_patch else 'off'} weight={joint_w}: {repr(e)}"
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
        joint = base_item.get("baseline_joint")
        full_model = source_items.get(attribute_text_source, {}).get("qwen_joint_patch")

        summary = f"{dataset_name:15s} shot={shot:<2} | Baseline={sfmt(baseline)}"
        if "joint" in requested_modules:
            summary += f" | Joint={sfmt(joint)}({gfmt(None if baseline is None or joint is None else joint - baseline)})"
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
        for source in requested_patch_text_sources if any(m in requested_modules for m in ["msc", "joint_msc"]) else []:
            item = source_items.get(source, {})
            msc = item.get("baseline_patch")
            joint_msc = item.get("baseline_joint_patch")
            print(
                f"{'':15s}         | "
                f"MSC-{source}={sfmt(msc)}({gfmt(None if baseline is None or msc is None else msc - baseline)}) | "
                f"Joint+MSC-{source}={sfmt(joint_msc)}"
                f"({gfmt(None if baseline is None or joint_msc is None else joint_msc - baseline)})"
            )
        if "full" in requested_modules:
            print(
                f"{'':15s}         | AKI+Joint+MSC-{attribute_text_source}={sfmt(full_model)}"
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
