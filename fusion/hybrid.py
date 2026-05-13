"""Hybrid geometry + ML label assignment at the segment level."""
from __future__ import annotations
from typing import Iterable
import numpy as np

from ..segments.segment import Segment, SemanticLabel

# index → label  (must match trained model's 8-class taxonomy)
CLASSES: list[SemanticLabel] = [
    "unlabeled",   # 0
    "ground",      # 1
    "road",        # 2
    "sidewalk",    # 3
    "building",    # 4
    "fence",       # 5
    "vegetation",  # 6
    "pole",        # 7
]
IDX = {c: i for i, c in enumerate(CLASSES)}


def _segment_votes(probs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Mean softmax over a segment's points → (C,)."""
    return probs[indices].mean(axis=0)


def _apply_geom_prior(seg: Segment, scores: np.ndarray) -> np.ndarray:
    """Nudge scores using geometry-derived priors and vetoes."""
    s = scores.copy()
    f = seg.features

    # Planes
    if seg.kind == "plane":
        # near-vertical planes → boost building, suppress ground/road
        if f.verticality > 0.7 and f.planarity > 0.3:
            s[IDX["building"]] *= 1.3
            s[IDX["fence"]] *= 1.1
            s[IDX["ground"]] *= 0.3
            s[IDX["road"]] *= 0.3
            s[IDX["sidewalk"]] *= 0.3
        # tiny planes can't be buildings
        if max(f.extent) < 0.4:
            s[IDX["building"]] *= 0.2
        # horizontal planes cannot be buildings or fences
        if f.verticality < 0.3:
            s[IDX["building"]] *= 0.0
            s[IDX["fence"]] *= 0.0

    # Ground-origin segments
    if seg.kind == "ground":
        s[IDX["ground"]] *= 2.0
        s[IDX["road"]] *= 1.5
        s[IDX["sidewalk"]] *= 1.5
        s[IDX["building"]] *= 0.0
        s[IDX["fence"]] *= 0.0
        s[IDX["vegetation"]] *= 0.5
        s[IDX["pole"]] *= 0.0

    # Clusters
    if seg.kind == "cluster":
        # compact, short → likely pole or vegetation
        if f.height_range < 0.3 and max(f.extent[:2]) < 1.0:
            s[IDX["building"]] *= 0.0
            s[IDX["fence"]] *= 0.0
        # tall, thin, non-planar → vegetation (tree/bush)
        if f.linearity > 0.6 and f.height_range > 1.5 and f.planarity < 0.3:
            s[IDX["vegetation"]] *= 1.4
        # very thin and tall → pole
        if f.linearity > 0.8 and f.height_range > 1.0 and max(f.extent[:2]) < 0.5:
            s[IDX["pole"]] *= 1.5

    # Renormalize
    s = np.maximum(s, 0)
    if s.sum() > 0:
        s = s / s.sum()
    return s


def fuse(segments: Iterable[Segment],
         probs: np.ndarray | None) -> list[Segment]:
    """Assign semantic label + confidence to each segment.

    `probs` is per-point softmax of shape (N, C) in the SAME cloud frame that
    segment indices refer to. If None, labels come from geometry priors alone
    (uniform ML prior)."""
    out = []
    for seg in segments:
        if probs is not None:
            ml = _segment_votes(probs, seg.indices)
        else:
            ml = np.ones(len(CLASSES)) / len(CLASSES)
        fused = _apply_geom_prior(seg, ml)
        top = int(np.argmax(fused))
        seg.semantic = CLASSES[top]
        # confidence: top score × agreement(ml, fused)
        agreement = float(1.0 - 0.5 * np.abs(ml - fused).sum())
        seg.confidence = float(fused[top] * max(0.0, agreement))
        out.append(seg)
    return out
