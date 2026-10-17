# JG-MSC configuration and dataset resources.
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

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
    # Formula: P_final = (1 - w) * P_LP + w * P_JointDualGraph
    # This is the same mathematical w as sensitivity["w"], but it serves a
    # different purpose: these are candidate operating points for ordinary
    # runs.  Under --experiment_suite/--run_sensitivity, the outer run is
    # fixed to sensitivity["fixed_w"] and the full sensitivity["w"] curve is
    # evaluated internally in that single pass.
    "joint_dual_weight": [0.8,0.9],

    # Parameter-sensitivity grids used by --experiment_suite or
    # --run_sensitivity.  tau_k is the normalized AKI pruning threshold
    # denoted by tau_K in the paper; alpha mixes the two graph spaces inside
    # JGP; w fuses the JGP prediction with the original LP prediction. Alpha
    # and w are different parameters; joint_dual_weight above is another use
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
        # ECLAP / LP
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
        "joint_gate_mode": "soft",
        "joint_text_gate": 1.0,
        "joint_dino_gate": 1.0,
        "joint_inner_weight": 0.5,
        "joint_mask_boost": 1.0,
        "joint_dual_scale": 1.0,

        # Inactive DINO cache branch defaults
        "dino_cache_beta": 5.5,
        "dino_cache_weight": 0.0,
    },

}

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

DEFAULT_FEATURES_PATH = "/home/zh/chy/chy/MLLM/ECALP-master/features"

DEFAULT_MOCO_FEATURES_PATH = "/home/zh/chy/chy/MLLM/ECALP-master/features_mocov3_aligned_16shot/fine_grained/MOCO/R50"

DEFAULT_QWEN_ATTR_ROOT = "/home/zh/chy/chy/MLLM/ECALP-master/qwen_attrs_universal"

DEFAULT_CUPL_ROOT = "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/Prompt_CuPL"

DEFAULT_META_ROOT = "/home/zh/chy/chy/MLLM/ECALP-master/features_mocov3_aligned_16shot/fine_grained"

DEFAULT_MOCOV3_CKPT = "/home/zh/chy/chy/AMU-Tuning/AMU-Tuning/r-50-1000ep.pth.tar"

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