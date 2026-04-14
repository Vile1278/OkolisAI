"""End-to-end CLI: .ply → segmented, wall-extracted Scene.

Examples:
    python -m okolis_ai.scripts.run_pipeline --input scans/yard.ply \
        --output outputs/yard_scene --view

    python -m okolis_ai.scripts.run_pipeline --input scans/yard.ply \
        --output outputs/yard_scene --model weights/randlanet_best.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from ..io.ply_loader import load_ply, to_numpy
from ..geometry.preprocess import preprocess
from ..geometry.ground import extract_ground, height_above_ground
from ..geometry.planes import extract_planes
from ..geometry.clusters import cluster_residual
from ..geometry import features as feat_mod
from ..segments.segment import Segment
from ..walls.wall_extractor import extract_wall
from ..fusion.hybrid import fuse, CLASSES, IDX
from ..ml.base import UniformSegmenter
from ..ml.inference import segment_cloud
from ..scene.scene import Scene


def build_scene(ply_path: Path, model_weights: Path | None = None,
                voxel: float = 0.03) -> Scene:
    # 1. Load + preprocess
    pcd = load_ply(ply_path)
    pcd = preprocess(pcd, voxel=voxel)
    arr = to_numpy(pcd)
    xyz = arr["xyz"]; rgb = arr["rgb"]

    # 2. Geometry
    ground_mask = extract_ground(pcd, method="grid", cell=0.3, z_tol=0.15)
    h_above = height_above_ground(pcd, ground_mask, cell=0.3)

    ground_idx = np.where(ground_mask)[0]
    planes = extract_planes(pcd, exclude=ground_idx,
                            distance=voxel * 1.5, min_inliers=500,
                            max_planes=30)

    used = np.zeros(len(xyz), dtype=bool); used[ground_idx] = True
    for p in planes: used[p.indices] = True
    remaining = np.where(~used)[0]
    clusters = cluster_residual(pcd, remaining, eps=voxel * 5, min_points=20)

    # 3. Build Segment objects
    segments: list[Segment] = []
    if len(ground_idx):
        segments.append(Segment.new(
            kind="ground", indices=ground_idx,
            features=feat_mod.compute(xyz[ground_idx])))
    for p in planes:
        segments.append(Segment.new(
            kind="plane", indices=p.indices,
            features=feat_mod.compute(xyz[p.indices]),
            normal=p.normal, plane=p.plane))
    for c in clusters:
        segments.append(Segment.new(
            kind="cluster", indices=c.indices,
            features=feat_mod.compute(xyz[c.indices])))

    # 4. ML segmentation (or uniform placeholder)
    if model_weights is not None:
        from ..ml.randlanet.segmenter import RandLANetSegmenter
        segmenter = RandLANetSegmenter(weights=model_weights)
    else:
        segmenter = UniformSegmenter()

    # Build ML features in the unified 5-dim layout
    # [R, G, B, intensity(=0 for iPhone), height-above-ground]
    from ..datasets.common import pack_features
    feats = pack_features(rgb=rgb, intensity=None, height_above_ground=h_above)
    probs = segment_cloud(segmenter, xyz, feats)

    # 5. Fuse
    segments = fuse(segments, probs)

    # 6. Walls
    walls = []
    plane_refs = [(p.normal, p.centroid) for p in planes]
    for s in segments:
        if s.semantic == "wall" and s.kind == "plane" and s.normal is not None:
            pts = xyz[s.indices]
            try:
                w = extract_wall(pts, s.normal, s.id,
                                 confidence=s.confidence,
                                 all_planes=plane_refs)
                walls.append(w)
            except Exception as e:
                print(f"[warn] wall extraction failed for {s.id}: {e}")

    return Scene(points=xyz, colors=rgb, segments=segments, walls=walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    scene = build_scene(args.input, args.model, voxel=args.voxel)
    scene.save(args.output)
    print(f"Saved scene → {args.output}  "
          f"({len(scene.segments)} segments, {len(scene.walls)} walls)")
    if args.view:
        from ..interaction.viewer import show
        show(scene)


if __name__ == "__main__":
    main()
