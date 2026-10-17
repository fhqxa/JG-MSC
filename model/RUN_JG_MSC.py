# Main command-line entry point for JG-MSC.
from config import *
from runner import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JG-MSC experiment runner. Default visual path: features_path/fine_grained/MOCO/R50."
    )

    # You only need to specify these commonly changed arguments.
    parser.add_argument("--datasets", type=str, default="ALL",
                        help="Dataset names separated by '/', e.g., Aircraft/Cars/DTD. Default: ALL from SUPPORTED_DATASETS.")
    parser.add_argument("--shots", type=str, default="",
                        help="Optional shot list separated by comma or slash, e.g., 16 or 1,2,4,8,16. Default uses SEARCH_CONFIG.")
    parser.add_argument("--modules", type=str, default="baseline,aki,joint,msc,joint_msc,full",
                        help=(
                            "Comma/slash list of modules to run: baseline, aki, joint, msc, joint_msc, full. "
                            "Aliases: lp=baseline, qwen=aki, joint+msc=joint_msc, joint+msc+aki=full."
                        ))
    parser.add_argument("--experiment_suite", action="store_true",
                        help=("Run and save the complete paper experiment suite: detailed component ablation, "
                              "per-query runtime, and tau_K/alpha/w sensitivity CSVs."))
    parser.add_argument(
        "--runtime_compare",
        action="store_true",
        help=(
            "Run exactly two timed methods per dataset/shot: original ECALP "
            "(cached text + LP only) and full JG-MSC (AKI + MSC + JGP). "
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
            "Default cached reproduces the original ECALP graph as closely as possible."
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
    parser.add_argument("--joint_dual_weights", type=str, default="",
                        help="Optional external JGP weights, e.g., 0.9 or 0.8,0.9. Default uses SEARCH_CONFIG.")
    parser.add_argument("--joint_graph_text_weight", type=float, default=0.5,
                        help="Alpha for Joint internal graph fusion: alpha * TEXT_GRAPH + (1-alpha) * MOCO_GRAPH.")
    parser.add_argument("--search_alpha", action="store_true",
                        help="Search joint_graph_text_weight inside one evaluation pass.")
    parser.add_argument(
        "--alpha_grid",
        type=str,
        default=",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["alpha"]),
        help="Grid for --search_alpha; defaults to SEARCH_CONFIG['sensitivity']['alpha'].",
    )
    parser.add_argument("--search_joint_weight", dest="search_joint_weight", action="store_true", default=True,
                        help="Search external Joint weight inside one evaluation pass (default: enabled).")
    parser.add_argument("--no_search_joint_weight", dest="search_joint_weight", action="store_false",
                        help="Disable internal Joint-weight search and use --joint_dual_weights as fixed outer runs.")
    parser.add_argument(
        "--joint_weight_grid",
        type=str,
        default=",".join(str(value) for value in SEARCH_CONFIG["sensitivity"]["w"]),
        help=("Grid for --search_joint_weight; defaults to SEARCH_CONFIG['sensitivity']['w']. "
              "Formula: final=(1-w)*LP+w*JointGraph."),
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
            "MoCo query rows follow the original ECALP cache order. Empty auto-resolves "
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
                        help=("Optional directory for exact per-query ECALP/MGP probability caches. "
                              "Empty disables export."))
    parser.add_argument("--prediction_export_per_class", type=int, default=0,
                        help="Queries saved per class in prediction caches; 0 saves all queries.")
    parser.add_argument("--prediction_export_seed", type=int, default=1,
                        help="Sampling seed used by --prediction_export_per_class.")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)