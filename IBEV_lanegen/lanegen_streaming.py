#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming Stage1 evidence aggregation for IBEV LaneGen Production V6.

The module keeps only Frenet grid accumulators and the current PCD chunk in
memory.  It intentionally does not concatenate the complete XYZ/RGB/intensity
point cloud.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import lanegen_core as core
from lanegen_pcd_io import PcdChunkReader


@dataclass(frozen=True)
class GridSpec:
    nd: int
    ns: int
    ncell: int
    d_min: float
    d_resolution: float

    @classmethod
    def from_args(cls, trajectory: core.Trajectory, args) -> "GridSpec":
        nd = int(round(2.0 * args.corridor_half_width / args.d_resolution))
        ns = int(len(trajectory.s))
        return cls(
            nd=nd,
            ns=ns,
            ncell=nd * ns,
            d_min=-float(args.corridor_half_width),
            d_resolution=float(args.d_resolution),
        )


class StreamingTopK:
    """Per-cell exact top/bottom-k accumulator.

    Memory is O(ncell*k), independent of point count.  Updates are performed by
    sorting the current chunk by cell and value, taking at most k candidates per
    cell, and merging them with the existing k values.
    """

    def __init__(self, ncell: int, k: int, *, largest: bool) -> None:
        self.ncell = int(ncell)
        self.k = max(1, int(k))
        self.largest = bool(largest)
        fill = -np.inf if self.largest else np.inf
        self.values = np.full((self.ncell, self.k), fill, dtype=np.float32)
        self.total_count = np.zeros(self.ncell, dtype=np.int32)

    def update(self, flat: np.ndarray, values: np.ndarray) -> None:
        flat = np.asarray(flat, dtype=np.int64).reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(flat) == 0:
            return
        valid = (
            (flat >= 0)
            & (flat < self.ncell)
            & np.isfinite(values)
        )
        if not np.any(valid):
            return
        flat = flat[valid]
        values = values[valid]

        # cell primary, requested value order secondary
        secondary = -values.astype(np.float64) if self.largest else values.astype(np.float64)
        order = np.lexsort((secondary, flat))
        sf = flat[order]
        sv = values[order]
        cells, starts, counts = np.unique(
            sf, return_index=True, return_counts=True
        )
        self.total_count[cells] += counts.astype(np.int32)

        repeated_starts = np.repeat(starts, counts)
        ranks = np.arange(len(sf), dtype=np.int64) - repeated_starts
        keep = ranks < self.k
        if not np.any(keep):
            return

        # row index in the compact candidate matrix
        group_index = np.repeat(np.arange(len(cells), dtype=np.int64), counts)
        compact = np.full(
            (len(cells), self.k),
            -np.inf if self.largest else np.inf,
            dtype=np.float32,
        )
        compact[group_index[keep], ranks[keep]] = sv[keep]

        merged = np.concatenate([self.values[cells], compact], axis=1)
        if self.largest:
            # k largest, descending
            part = np.partition(merged, merged.shape[1] - self.k, axis=1)[:, -self.k:]
            part.sort(axis=1)
            self.values[cells] = part[:, ::-1]
        else:
            # k smallest, ascending
            part = np.partition(merged, self.k - 1, axis=1)[:, :self.k]
            part.sort(axis=1)
            self.values[cells] = part

    def mean_and_count(self) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(self.values)
        count = finite.sum(axis=1).astype(np.int32)
        vals = np.where(finite, self.values, 0.0)
        mean = np.divide(
            vals.sum(axis=1, dtype=np.float64),
            np.maximum(count, 1),
        ).astype(np.float32)
        mean[count == 0] = 0.0
        return mean, count


class StreamingMoments:
    """Per-cell sum/count aggregator for RGB means."""

    def __init__(self, ncell: int, channels: int = 1) -> None:
        self.ncell = int(ncell)
        self.channels = int(channels)
        self.count = np.zeros(self.ncell, dtype=np.int32)
        self.sum = np.zeros((self.channels, self.ncell), dtype=np.float64)

    def update(self, flat: np.ndarray, values: np.ndarray) -> None:
        flat = np.asarray(flat, dtype=np.int64).reshape(-1)
        if len(flat) == 0:
            return
        vals = np.asarray(values)
        if self.channels == 1:
            vals = vals.reshape(-1, 1)
        if vals.shape != (len(flat), self.channels):
            raise ValueError(
                f"values shape mismatch: {vals.shape}, expected {(len(flat), self.channels)}"
            )
        valid = (flat >= 0) & (flat < self.ncell) & np.isfinite(vals).all(axis=1)
        flat = flat[valid]
        vals = vals[valid]
        if len(flat) == 0:
            return
        np.add.at(self.count, flat, 1)
        for c in range(self.channels):
            np.add.at(self.sum[c], flat, vals[:, c].astype(np.float64))

    def mean(self) -> np.ndarray:
        out = np.zeros((self.ncell, self.channels), dtype=np.float32)
        ok = self.count > 0
        if np.any(ok):
            out[ok] = (self.sum[:, ok] / self.count[ok][None, :]).T.astype(np.float32)
        return out


@dataclass
class ProjectionTotals:
    input_points: int = 0
    within_corridor: int = 0
    ambiguous: int = 0
    accepted: int = 0

    def add(self, proj: core.FrenetProjection, n_input: int) -> None:
        self.input_points += int(n_input)
        within = np.asarray(proj.within_corridor_all, dtype=bool)
        ambiguous = np.asarray(proj.ambiguous_input, dtype=bool)
        self.within_corridor += int(within.sum())
        self.ambiguous += int((within & ambiguous).sum())
        self.accepted += int(np.asarray(proj.keep, dtype=bool).sum())

    def as_dict(self) -> dict:
        return {
            "input_points": int(self.input_points),
            "within_corridor_points_before_ambiguity_filter": int(self.within_corridor),
            "ambiguous_points": int(self.ambiguous),
            "ambiguous_ratio_in_corridor": float(self.ambiguous / max(self.within_corridor, 1)),
            "accepted_points": int(self.accepted),
        }


class BBoxAccumulator:
    def __init__(self) -> None:
        self.minimum = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        self.maximum = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        self.count = 0

    def update(self, xyz: np.ndarray) -> None:
        if len(xyz) == 0:
            return
        self.minimum = np.minimum(self.minimum, np.min(xyz, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(xyz, axis=0))
        self.count += int(len(xyz))

    def world_xy_dict(self) -> dict:
        if self.count == 0:
            return {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0}
        return {
            "x_min": float(self.minimum[0]),
            "x_max": float(self.maximum[0]),
            "y_min": float(self.minimum[1]),
            "y_max": float(self.maximum[1]),
        }


def project_chunk(xyz: np.ndarray, trajectory: core.Trajectory, args) -> core.FrenetProjection:
    return core.project_to_frenet(
        xyz,
        trajectory,
        args.corridor_half_width,
        candidates=args.frenet_candidates,
        z_weight=args.frenet_z_weight,
        heading_weight=args.frenet_heading_weight,
        ambiguity_margin=args.frenet_ambiguity_margin,
        ambiguity_s_separation=args.frenet_ambiguity_s_separation,
        drop_ambiguous=not args.keep_ambiguous_frenet,
    )


def cell_indices(proj: core.FrenetProjection, spec: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_index = np.floor(
        (proj.d_value - spec.d_min) / spec.d_resolution
    ).astype(np.int64)
    ok = (d_index >= 0) & (d_index < spec.nd)
    d_index = d_index[ok]
    s_index = proj.s_index[ok]
    flat = d_index * spec.ns + s_index
    return flat, ok, d_index


def update_ambiguous_grid(
    accumulator: np.ndarray,
    proj: core.FrenetProjection,
    spec: GridSpec,
) -> None:
    amb = (
        np.asarray(proj.ambiguous_input, dtype=bool)
        & np.asarray(proj.within_corridor_all, dtype=bool)
    )
    if not np.any(amb):
        return
    amb_d = np.floor(
        (proj.best_d_value_all[amb] - spec.d_min) / spec.d_resolution
    ).astype(np.int64)
    amb_s = proj.best_s_index_all[amb]
    ok = (amb_d >= 0) & (amb_d < spec.nd)
    if not np.any(ok):
        return
    flat = amb_d[ok] * spec.ns + amb_s[ok]
    np.add.at(accumulator, flat, 1)


def packed_rgb_channels(rgb: np.ndarray) -> np.ndarray:
    return core.decode_rgb_channels(rgb)


def chunk_intensity(
    rgb: Optional[np.ndarray],
    native: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    if native is not None:
        return np.asarray(native, dtype=np.float32)
    if rgb is None:
        raise ValueError("反射強度に使えるフィールドがありません")
    channels = packed_rgb_channels(rgb).astype(np.float32)
    if mode == "packed_rgb_grayscale_red":
        return channels[:, 0]
    return channels.mean(axis=1, dtype=np.float32)


def detect_intensity_mode(reader: PcdChunkReader) -> tuple[str, dict]:
    """Determine native/packed intensity mode without retaining point arrays."""
    if bool(reader.info.get("has_native_intensity")):
        # The exact cell evidence statistics are recorded after aggregation.
        # Avoid an otherwise redundant full PCD pass just for raw percentiles.
        return "native_intensity_field", {
            "decode_mode": "native_intensity_field",
            "streaming_raw_percentiles": "not_materialized",
        }

    native_seen = False
    rgb_seen = False
    gray_equal = 0
    rgb_count = 0
    minimum = np.inf
    maximum = -np.inf
    # Exact 0..255 histogram for packed RGB intensity metadata.
    histogram = np.zeros(256, dtype=np.int64)

    for xyz, rgb, native in reader.iter_chunks(phase="intensity_mode"):
        if native is not None:
            native_seen = True
            if len(native):
                minimum = min(minimum, float(np.min(native)))
                maximum = max(maximum, float(np.max(native)))
        elif rgb is not None:
            rgb_seen = True
            ch = packed_rgb_channels(rgb)
            eq = (ch[:, 0] == ch[:, 1]) & (ch[:, 1] == ch[:, 2])
            gray_equal += int(eq.sum())
            rgb_count += int(len(ch))

    if native_seen:
        return "native_intensity_field", {
            "decode_mode": "native_intensity_field",
            "min": 0.0 if not np.isfinite(minimum) else float(minimum),
            "max": 0.0 if not np.isfinite(maximum) else float(maximum),
        }
    if not rgb_seen:
        raise ValueError("反射強度に使えるフィールドがありません")
    ratio = float(gray_equal / max(rgb_count, 1))
    mode = "packed_rgb_grayscale_red" if ratio >= 0.999 else "packed_rgb_channel_mean"

    # One more bounded-memory pass gives exact histogram percentiles.
    for xyz, rgb, native in reader.iter_chunks(phase="intensity_histogram"):
        vals = chunk_intensity(rgb, native, mode)
        q = np.clip(np.rint(vals), 0, 255).astype(np.int64)
        histogram += np.bincount(q, minlength=256)

    def percentile(p: float) -> float:
        total = int(histogram.sum())
        if total == 0:
            return 0.0
        threshold = p / 100.0 * max(total - 1, 0)
        return float(np.searchsorted(np.cumsum(histogram), threshold, side="right"))

    return mode, {
        "decode_mode": mode,
        "grayscale_ratio": ratio,
        "min": float(np.flatnonzero(histogram)[0]) if histogram.any() else 0.0,
        "p50": percentile(50),
        "p90": percentile(90),
        "p99": percentile(99),
        "max": float(np.flatnonzero(histogram)[-1]) if histogram.any() else 0.0,
    }


def build_intensity_streaming(
    reader: PcdChunkReader,
    trajectory: core.Trajectory,
    args,
    intensity_mode: str,
) -> tuple[dict, BBoxAccumulator]:
    spec = GridSpec.from_args(trajectory, args)
    bbox = BBoxAccumulator()
    totals = ProjectionTotals()
    ambiguity_count = np.zeros(spec.ncell, dtype=np.int32)
    low_z = StreamingTopK(spec.ncell, args.surface_low_k, largest=False)

    # Pass 1: robust lower-surface reference and Frenet ambiguity distribution.
    for xyz, rgb, native in reader.iter_chunks(phase="intensity_surface_pass"):
        bbox.update(xyz)
        proj = project_chunk(xyz, trajectory, args)
        totals.add(proj, len(xyz))
        update_ambiguous_grid(ambiguity_count, proj, spec)
        flat, ok, _ = cell_indices(proj, spec)
        z = xyz[proj.keep, 2][ok]
        low_z.update(flat, z)

    z_base, z_base_count = low_z.mean_and_count()
    del low_z

    top_i = StreamingTopK(spec.ncell, args.topk, largest=True)
    surface_count = np.zeros(spec.ncell, dtype=np.int32)
    corridor_points = 0
    surface_points = 0

    # Pass 2: select points near robust surface and aggregate intensity top-k.
    for xyz, rgb, native in reader.iter_chunks(phase="intensity_evidence_pass"):
        proj = project_chunk(xyz, trajectory, args)
        flat, ok, _ = cell_indices(proj, spec)
        z = xyz[proj.keep, 2][ok]
        vals = chunk_intensity(rgb, native, intensity_mode)[proj.keep][ok]
        corridor_points += int(len(flat))
        surface = z <= z_base[flat] + float(args.surface_band)
        sf = flat[surface]
        sv = vals[surface]
        surface_points += int(len(sf))
        top_i.update(sf, sv)
        if len(sf):
            np.add.at(surface_count, sf, 1)

    i_topk_flat, k_count = top_i.mean_and_count()
    del top_i

    i_topk = i_topk_flat.reshape(spec.nd, spec.ns)
    count = surface_count.reshape(spec.nd, spec.ns)
    occupied = count > 0
    ambiguous_count_grid = ambiguity_count.reshape(spec.nd, spec.ns)

    inner = max(1, int(round(args.bg_inner / args.d_resolution)))
    outer = max(1, int(round(args.bg_outer / args.d_resolution)))
    i_bg, bg_n = core.annulus_background(i_topk, occupied, inner, outer, axis=0)
    bg_valid = np.isfinite(i_bg) & (bg_n >= 4) & occupied
    i_excess = np.where(
        bg_valid, i_topk - np.nan_to_num(i_bg), 0.0
    ).astype(np.float32)
    smooth_cells = max(1, int(round(args.scale_smooth / args.s_resolution)))
    scale = core.row_robust_scale(i_excess, bg_valid, smooth_cells, args.scale_floor)
    i_sigma = (i_excess / scale[None, :]).astype(np.float32)
    i_sigma[~bg_valid] = 0.0

    valid_vals = i_topk[occupied]
    stats = {
        "aggregation_mode": "fully_streaming_grid",
        "full_point_arrays_materialized": False,
        "pcd_passes": 2,
        "corridor_points": int(corridor_points),
        "surface_points": int(surface_points),
        "surface_low_k": int(args.surface_low_k),
        "surface_base_used_mean": float(z_base_count[z_base_count > 0].mean())
            if np.any(z_base_count > 0) else 0.0,
        "occupied_cells": int(occupied.sum()),
        "bg_valid_cells": int(bg_valid.sum()),
        "mean_points_per_cell": float(count[occupied].mean()) if occupied.any() else 0.0,
        "topk_used_mean": float(k_count[k_count > 0].mean()) if np.any(k_count > 0) else 0.0,
        "intensity_p50": float(np.percentile(valid_vals, 50)) if valid_vals.size else 0.0,
        "intensity_p90": float(np.percentile(valid_vals, 90)) if valid_vals.size else 0.0,
        "intensity_p99": float(np.percentile(valid_vals, 99)) if valid_vals.size else 0.0,
        "excess_p50": float(np.percentile(i_excess[bg_valid], 50)) if bg_valid.any() else 0.0,
        "excess_p99": float(np.percentile(i_excess[bg_valid], 99)) if bg_valid.any() else 0.0,
        "scale_median": float(np.median(scale)),
        "sigma_p99": float(np.percentile(i_sigma[bg_valid], 99)) if bg_valid.any() else 0.0,
        "sigma_p999": float(np.percentile(i_sigma[bg_valid], 99.9)) if bg_valid.any() else 0.0,
        "frenet_projection": totals.as_dict(),
    }
    return ({
        "occupied": occupied,
        "count": count.astype(np.int32),
        "i_topk": i_topk.astype(np.float32),
        "i_bg": np.nan_to_num(i_bg).astype(np.float32),
        "i_excess": i_excess,
        "i_sigma": i_sigma,
        "i_scale_s": scale.astype(np.float32),
        "bg_valid": bg_valid,
        "frenet_ambiguous": ambiguous_count_grid > 0,
        "frenet_ambiguous_count": ambiguous_count_grid.astype(np.int32),
        "_stats": stats,
    }, bbox)


def build_z_streaming(
    reader: PcdChunkReader,
    trajectory: core.Trajectory,
    args,
) -> dict:
    spec = GridSpec.from_args(trajectory, args)
    totals = ProjectionTotals()
    low_z = StreamingTopK(spec.ncell, args.z_low_k, largest=False)
    accepted = 0
    for xyz, rgb, native in reader.iter_chunks(phase="z_evidence_pass"):
        proj = project_chunk(xyz, trajectory, args)
        totals.add(proj, len(xyz))
        flat, ok, _ = cell_indices(proj, spec)
        z = xyz[proj.keep, 2][ok]
        accepted += int(len(flat))
        low_z.update(flat, z)

    z_low, _ = low_z.mean_and_count()
    count = low_z.total_count.copy()
    del low_z
    z_surface = z_low.reshape(spec.nd, spec.ns)
    z_occupied = count.reshape(spec.nd, spec.ns) > 0

    d_axis = (
        spec.d_min + (np.arange(spec.nd) + 0.5) * args.d_resolution
    ).astype(np.float64)
    fit_mask = z_occupied & (np.abs(d_axis)[:, None] <= args.road_fit_half_width)
    w = fit_mask.astype(np.float64)
    n = w.sum(axis=0)
    sd = (w * d_axis[:, None]).sum(axis=0)
    sdd = (w * (d_axis ** 2)[:, None]).sum(axis=0)
    sz = (w * z_surface).sum(axis=0)
    sdz = (w * d_axis[:, None] * z_surface).sum(axis=0)
    det = n * sdd - sd * sd
    alpha = np.zeros(spec.ns)
    beta = np.zeros(spec.ns)
    good = (n >= 20) & (np.abs(det) > 1e-9)
    alpha[good] = (n[good] * sdz[good] - sd[good] * sz[good]) / det[good]
    beta[good] = (sdd[good] * sz[good] - sd[good] * sdz[good]) / det[good]
    idx = np.arange(spec.ns)
    if np.any(good):
        alpha = np.interp(idx, idx[good], alpha[good])
        beta = np.interp(idx, idx[good], beta[good])
    z_detrend = (
        z_surface - (alpha[None, :] * d_axis[:, None] + beta[None, :])
    ).astype(np.float32)
    z_detrend[~z_occupied] = 0.0

    inner = max(1, int(round(args.level_inner / args.d_resolution)))
    outer = max(1, int(round(args.level_outer / args.d_resolution)))
    sv_hi, sm_hi = core._masked_running_sum(z_detrend, z_occupied, inner, outer, axis=0)
    sv_lo, sm_lo = core._masked_running_sum(z_detrend, z_occupied, -outer, -inner, axis=0)
    min_n = max(3, (outer - inner) // 3)
    pair_ok = (sm_hi >= min_n) & (sm_lo >= min_n)
    level_hi = np.divide(sv_hi, np.maximum(sm_hi, 1e-9))
    level_lo = np.divide(sv_lo, np.maximum(sm_lo, 1e-9))
    z_level_diff = np.where(pair_ok, level_hi - level_lo, 0.0).astype(np.float32)

    h = max(1, int(round(args.grad_half_window / args.d_resolution)))
    dz_dd = np.zeros((spec.nd, spec.ns), dtype=np.float32)
    up = np.roll(z_detrend, -h, axis=0)
    dn = np.roll(z_detrend, h, axis=0)
    up_ok = np.roll(z_occupied, -h, axis=0)
    dn_ok = np.roll(z_occupied, h, axis=0)
    both = up_ok & dn_ok
    both[:h, :] = False
    both[-h:, :] = False
    dz_dd[both] = ((up - dn)[both] / (2.0 * h * args.d_resolution)).astype(np.float32)

    stats = {
        "aggregation_mode": "fully_streaming_grid",
        "full_point_arrays_materialized": False,
        "pcd_passes": 1,
        "corridor_points": int(accepted),
        "occupied_cells": int(z_occupied.sum()),
        "cross_slope_median": float(np.median(alpha)),
        "level_diff_p50": float(np.percentile(np.abs(z_level_diff[pair_ok]), 50)) if pair_ok.any() else 0.0,
        "level_diff_p99": float(np.percentile(np.abs(z_level_diff[pair_ok]), 99)) if pair_ok.any() else 0.0,
        "grad_p99": float(np.percentile(np.abs(dz_dd[both]), 99)) if both.any() else 0.0,
        "frenet_projection": totals.as_dict(),
    }
    return {
        "z_surface": z_surface.astype(np.float32),
        "z_detrend": z_detrend,
        "z_level_diff": z_level_diff,
        "dz_dd": dz_dd,
        "z_occupied": z_occupied,
        "z_pair_ok": pair_ok,
        "_stats": stats,
    }


def inspect_rgb(reader: PcdChunkReader) -> tuple[bool, float]:
    rgb_points = 0
    grayscale = 0
    for xyz, rgb, native in reader.iter_chunks(phase="rgb_inspect"):
        if rgb is None:
            continue
        ch = packed_rgb_channels(rgb)
        rgb_points += int(len(ch))
        grayscale += int(((ch[:, 0] == ch[:, 1]) & (ch[:, 1] == ch[:, 2])).sum())
    ratio = float(grayscale / max(rgb_points, 1))
    return rgb_points > 0, ratio


def build_rgb_streaming(
    reader: PcdChunkReader,
    trajectory: core.Trajectory,
    args,
) -> dict:
    spec = GridSpec.from_args(trajectory, args)
    low_z = StreamingTopK(spec.ncell, args.surface_low_k, largest=False)
    totals = ProjectionTotals()

    for xyz, rgb, native in reader.iter_chunks(phase="rgb_surface_pass"):
        if rgb is None:
            continue
        proj = project_chunk(xyz, trajectory, args)
        totals.add(proj, len(xyz))
        flat, ok, _ = cell_indices(proj, spec)
        z = xyz[proj.keep, 2][ok]
        low_z.update(flat, z)

    z_base, _ = low_z.mean_and_count()
    del low_z
    moments = StreamingMoments(spec.ncell, channels=3)

    for xyz, rgb, native in reader.iter_chunks(phase="rgb_evidence_pass"):
        if rgb is None:
            continue
        proj = project_chunk(xyz, trajectory, args)
        flat, ok, _ = cell_indices(proj, spec)
        z = xyz[proj.keep, 2][ok]
        ch = packed_rgb_channels(rgb)[proj.keep][ok]
        surface = z <= z_base[flat] + float(args.surface_band)
        moments.update(flat[surface], ch[surface].astype(np.float32))

    mean = moments.mean().reshape(spec.nd, spec.ns, 3)
    rgb_mean = np.clip(mean, 0, 255).astype(np.uint8)
    rgb_occupied = moments.count.reshape(spec.nd, spec.ns) > 0
    rf = rgb_mean.astype(np.float32)
    luminance = 0.299 * rf[:, :, 0] + 0.587 * rf[:, :, 1] + 0.114 * rf[:, :, 2]
    mx = rf.max(axis=2)
    mn = rf.min(axis=2)
    saturation = np.divide(mx - mn, np.maximum(mx, 1.0)).astype(np.float32)
    stats = {
        "aggregation_mode": "fully_streaming_grid",
        "full_point_arrays_materialized": False,
        "pcd_passes": 2,
        "occupied_cells": int(rgb_occupied.sum()),
        "luminance_p50": float(np.percentile(luminance[rgb_occupied], 50)) if rgb_occupied.any() else 0.0,
        "luminance_p99": float(np.percentile(luminance[rgb_occupied], 99)) if rgb_occupied.any() else 0.0,
        "frenet_projection": totals.as_dict(),
    }
    return {
        "rgb_mean": rgb_mean,
        "rgb_occupied": rgb_occupied,
        "rgb_luminance": luminance.astype(np.float32),
        "rgb_saturation": saturation.astype(np.float32),
        "_stats": stats,
    }
