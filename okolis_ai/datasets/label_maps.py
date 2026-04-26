"""Unified label taxonomy and per-dataset remaps.

Target classes:
    0 unlabeled | 1 ground | 2 road | 3 wall | 4 nature | 5 object
"""
UNIFIED = {
    "unlabeled": 0, "ground": 1, "road": 2, "wall": 3, "nature": 4, "object": 5
}

# ---- Toronto-3D (8 classes in original) ----
# 0 unclassified, 1 road, 2 road marking, 3 natural, 4 building,
# 5 utility line, 6 pole, 7 car, 8 fence
TORONTO3D_MAP = {
    0: UNIFIED["unlabeled"],
    1: UNIFIED["road"],
    2: UNIFIED["road"],
    3: UNIFIED["nature"],
    4: UNIFIED["wall"],
    5: UNIFIED["object"],
    6: UNIFIED["object"],
    7: UNIFIED["object"],
    8: UNIFIED["wall"],
}

# ---- SemanticKITTI (20 classes learning set) ----
# kept only relevant ids, extend as needed
SEMKITTI_MAP = {
    0:  UNIFIED["unlabeled"],
    9:  UNIFIED["road"],
    10: UNIFIED["ground"],    # sidewalk
    11: UNIFIED["ground"],    # other-ground
    13: UNIFIED["wall"],      # building
    14: UNIFIED["wall"],      # fence
    15: UNIFIED["nature"],    # vegetation
    16: UNIFIED["nature"],    # trunk
    17: UNIFIED["nature"],    # terrain
    18: UNIFIED["object"],    # pole
    19: UNIFIED["object"],    # traffic sign
    1: UNIFIED["object"],     # car
}

# ---- BotanicGarden (continuous classes) ----
BOTANIC_MAP = {
    0: UNIFIED["unlabeled"],
    1: UNIFIED["ground"],
    2: UNIFIED["nature"],
    3: UNIFIED["object"],
    4: UNIFIED["wall"],
}

# ---- S3DIS (indoor, used for wall/floor priors) ----
S3DIS_MAP = {
    "ceiling": UNIFIED["unlabeled"],
    "floor":   UNIFIED["ground"],
    "wall":    UNIFIED["wall"],
    "beam":    UNIFIED["wall"],
    "column":  UNIFIED["wall"],
    "window":  UNIFIED["wall"],
    "door":    UNIFIED["wall"],
    "table":   UNIFIED["object"],
    "chair":   UNIFIED["object"],
    "sofa":    UNIFIED["object"],
    "bookcase":UNIFIED["object"],
    "board":   UNIFIED["object"],
    "clutter": UNIFIED["object"],
}


def apply_map(labels, mapping):
    import numpy as np
    out = np.zeros_like(labels)
    for k, v in mapping.items():
        out[labels == k] = v
    return out
