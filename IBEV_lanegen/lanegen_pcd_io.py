#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production PCD I/O for IBEV LaneGen.

Supports PCD DATA modes:
- ascii
- binary
- binary_compressed (LZF)

The public API intentionally returns only the fields used by LaneGen:
XYZ, packed RGB/RGBA and native intensity/reflectivity.
ASCII input is decoded in line chunks to avoid the former whole-file text copy.
Binary input is read through numpy.memmap. binary_compressed is decompressed
directly into a temporary memmap and reused across repeated point passes.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional
import os
import tempfile

import numpy as np

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class PcdReadCancelled(RuntimeError):
    """Raised when a caller-provided cancellation callback requests stop."""


@dataclass(frozen=True)
class PcdHeader:
    path: Path
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    width: int
    height: int
    points: int
    data: str
    data_offset: int

    @property
    def scalar_columns(self) -> int:
        return int(sum(self.counts))


def _normalise_vector(values: list[str], n: int, default: str) -> tuple[str, ...]:
    if not values:
        values = [default] * n
    if len(values) == 1 and n > 1:
        values = values * n
    if len(values) != n:
        raise ValueError(f"PCD header vector length mismatch: expected {n}, got {len(values)}")
    return tuple(values)


def read_pcd_header(path: Path) -> PcdHeader:
    path = Path(path).expanduser().resolve()
    raw_header: dict[str, list[str]] = {}
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PCDヘッダの途中でEOFになりました: {path}")
            line = raw.decode("ascii", "strict").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].upper()
            raw_header[key] = parts[1:]
            if key == "DATA":
                data_offset = f.tell()
                break

    fields = tuple(x.lower() for x in raw_header.get("FIELDS", []))
    if not fields:
        raise ValueError(f"PCD FIELDSがありません: {path}")
    n = len(fields)
    sizes = tuple(int(x) for x in _normalise_vector(raw_header.get("SIZE", []), n, "4"))
    types = tuple(x.upper() for x in _normalise_vector(raw_header.get("TYPE", []), n, "F"))
    counts = tuple(int(x) for x in _normalise_vector(raw_header.get("COUNT", []), n, "1"))
    width = int(raw_header.get("WIDTH", ["0"])[0])
    height = int(raw_header.get("HEIGHT", ["1"])[0])
    points = int(raw_header.get("POINTS", [str(width * height)])[0])
    data = raw_header.get("DATA", [""])[0].lower()
    if data not in {"ascii", "binary", "binary_compressed"}:
        raise NotImplementedError(f"未対応のPCD DATA形式です: {data} ({path})")
    for required in ("x", "y", "z"):
        if required not in fields:
            raise ValueError(f"FIELDS に {required} がありません: {fields}")
    return PcdHeader(path, fields, sizes, types, counts, width, height,
                     points, data, data_offset)


def _numpy_scalar_dtype(type_code: str, size: int) -> np.dtype:
    key = (type_code.upper(), int(size))
    table = {
        ("F", 4): np.dtype("<f4"), ("F", 8): np.dtype("<f8"),
        ("I", 1): np.dtype("i1"), ("I", 2): np.dtype("<i2"),
        ("I", 4): np.dtype("<i4"), ("I", 8): np.dtype("<i8"),
        ("U", 1): np.dtype("u1"), ("U", 2): np.dtype("<u2"),
        ("U", 4): np.dtype("<u4"), ("U", 8): np.dtype("<u8"),
    }
    if key not in table:
        raise NotImplementedError(f"未対応のPCD TYPE/SIZEです: TYPE={key[0]} SIZE={key[1]}")
    return table[key]


def _structured_dtype(header: PcdHeader) -> np.dtype:
    specs = []
    for name, size, typ, count in zip(header.fields, header.sizes,
                                      header.types, header.counts):
        dt = _numpy_scalar_dtype(typ, size)
        specs.append((name, dt) if count == 1 else (name, dt, (count,)))
    return np.dtype(specs, align=False)


def _expanded_column_slices(header: PcdHeader) -> dict[str, slice]:
    out: dict[str, slice] = {}
    col = 0
    for name, count in zip(header.fields, header.counts):
        out[name] = slice(col, col + count)
        col += count
    return out


def _packed_rgb_as_float32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "f":
        return values.astype(np.float32, copy=False).reshape(-1)
    # PCD also permits rgb/rgba as UINT32. Preserve its bit pattern so the
    # existing packed-float decoder remains compatible.
    u = values.astype(np.uint32, copy=False).reshape(-1)
    return np.ascontiguousarray(u).view(np.float32)


def _select_fields(field_arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    xyz = np.column_stack([
        np.asarray(field_arrays["x"]).reshape(-1),
        np.asarray(field_arrays["y"]).reshape(-1),
        np.asarray(field_arrays["z"]).reshape(-1),
    ]).astype(np.float32, copy=False)

    rgb = None
    for name in ("rgb", "rgba"):
        if name in field_arrays:
            rgb = _packed_rgb_as_float32(field_arrays[name])
            break

    intensity = None
    for name in ("intensity", "reflectivity", "i", "intensities"):
        if name in field_arrays:
            intensity = np.asarray(field_arrays[name]).reshape(-1).astype(np.float32, copy=False)
            break

    finite = np.isfinite(xyz).all(axis=1)
    if rgb is not None:
        finite &= np.isfinite(rgb)
    if intensity is not None:
        finite &= np.isfinite(intensity)
    return (
        np.ascontiguousarray(xyz[finite], dtype=np.float32),
        None if rgb is None else np.ascontiguousarray(rgb[finite], dtype=np.float32),
        None if intensity is None else np.ascontiguousarray(intensity[finite], dtype=np.float32),
    )


def _check_cancel(cancel_callback: Optional[CancelCallback]) -> None:
    if cancel_callback is not None and bool(cancel_callback()):
        raise PcdReadCancelled("PCD読み込みがキャンセルされました")


def _notify(cb: Optional[ProgressCallback], done: int, total: int, phase: str) -> None:
    if cb is not None:
        cb(int(done), int(total), str(phase))


def _iter_ascii(header: PcdHeader, chunk_points: int,
                progress_callback: Optional[ProgressCallback],
                cancel_callback: Optional[CancelCallback]):
    slices = _expanded_column_slices(header)
    total_cols = header.scalar_columns
    done = 0
    with header.path.open("rb") as f:
        f.seek(header.data_offset)
        lines: list[bytes] = []
        _check_cancel(cancel_callback)
        while True:
            raw = f.readline()
            if raw:
                if raw.strip() and not raw.lstrip().startswith(b"#"):
                    lines.append(raw)
            if (not raw) or len(lines) >= chunk_points:
                if lines:
                    text = b"".join(lines).decode("ascii", "ignore")
                    arr = np.fromstring(text, sep=" ", dtype=np.float64)
                    if arr.size % total_cols != 0:
                        raise ValueError(
                            f"ASCII PCD列数がFIELDS/COUNTと不一致です: values={arr.size}, cols={total_cols}"
                        )
                    arr = arr.reshape(-1, total_cols)
                    fields = {}
                    for name, sl in slices.items():
                        value = arr[:, sl]
                        fields[name] = value[:, 0] if value.shape[1] == 1 else value
                    xyz, rgb, intensity = _select_fields(fields)
                    done += len(lines)
                    _notify(progress_callback, min(done, header.points), header.points, "ascii")
                    yield xyz, rgb, intensity
                    lines.clear()
                    # Cancellation is checked at chunk boundaries, not for every
                    # ASCII line.  Per-line filesystem stat calls make million-point
                    # files unusably slow.
                    _check_cancel(cancel_callback)
                if not raw:
                    break


def _iter_binary(header: PcdHeader, chunk_points: int,
                 progress_callback: Optional[ProgressCallback],
                 cancel_callback: Optional[CancelCallback]):
    dtype = _structured_dtype(header)
    available = max(0, (header.path.stat().st_size - header.data_offset) // dtype.itemsize)
    n = min(header.points, int(available)) if header.points > 0 else int(available)
    mm = np.memmap(header.path, mode="r", dtype=dtype,
                   offset=header.data_offset, shape=(n,))
    try:
        for start in range(0, n, chunk_points):
            _check_cancel(cancel_callback)
            stop = min(n, start + chunk_points)
            block = mm[start:stop]
            fields = {name: np.asarray(block[name]) for name in header.fields}
            yield _select_fields(fields)
            _notify(progress_callback, stop, n, "binary")
    finally:
        del mm


def lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Pure Python LZF decompressor used by PCD binary_compressed."""
    src = memoryview(data)
    out = bytearray(expected_size)
    ip = 0
    op = 0
    n = len(src)
    while ip < n:
        ctrl = int(src[ip]); ip += 1
        if ctrl < 32:
            length = ctrl + 1
            if ip + length > n or op + length > expected_size:
                raise ValueError("壊れたLZF literal block")
            out[op:op + length] = src[ip:ip + length]
            ip += length
            op += length
        else:
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                if ip >= n:
                    raise ValueError("壊れたLZF length block")
                length += int(src[ip]); ip += 1
            if ip >= n:
                raise ValueError("壊れたLZF back reference")
            ref -= int(src[ip]); ip += 1
            length += 2
            if ref < 0 or op + length > expected_size:
                raise ValueError("LZF back reference範囲外")
            for _ in range(length):
                out[op] = out[ref]
                op += 1
                ref += 1
    if op != expected_size:
        raise ValueError(f"LZF展開サイズ不一致: {op} != {expected_size}")
    return bytes(out)


def _lzf_decompress_to_memmap(
    compressed: np.ndarray,
    output: np.memmap,
    expected_size: int,
) -> None:
    """LZF decompression directly into a temporary memmap.

    This avoids allocating the full uncompressed binary_compressed payload as a
    Python ``bytes``/``bytearray`` object.  Back references read from the output
    memmap, which is valid because LZF only references already-produced bytes.
    """
    src = np.asarray(compressed, dtype=np.uint8)
    ip = 0
    op = 0
    n = int(len(src))
    while ip < n:
        ctrl = int(src[ip]); ip += 1
        if ctrl < 32:
            length = ctrl + 1
            if ip + length > n or op + length > expected_size:
                raise ValueError("壊れたLZF literal block")
            output[op:op + length] = src[ip:ip + length]
            ip += length
            op += length
        else:
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                if ip >= n:
                    raise ValueError("壊れたLZF length block")
                length += int(src[ip]); ip += 1
            if ip >= n:
                raise ValueError("壊れたLZF back reference")
            ref -= int(src[ip]); ip += 1
            length += 2
            if ref < 0 or op + length > expected_size:
                raise ValueError("LZF back reference範囲外")
            # Overlap is allowed, therefore copy in forward order.
            for _ in range(length):
                output[op] = output[ref]
                op += 1
                ref += 1
    if op != expected_size:
        raise ValueError(f"LZF展開サイズ不一致: {op} != {expected_size}")
    output.flush()


class PcdChunkReader:
    """Reusable, bounded-memory PCD chunk source.

    A reader can be iterated multiple times.  For ``binary_compressed`` input,
    the field-major uncompressed block is materialised once into a temporary
    memmap and reused across all Stage1 passes.  The temporary file is deleted
    on ``close()`` or context-manager exit.
    """

    def __init__(
        self,
        path: Path,
        chunk_points: int = 500_000,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_callback: Optional[CancelCallback] = None,
        temp_dir: Optional[Path] = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.header = read_pcd_header(self.path)
        self.chunk_points = max(1, int(chunk_points))
        self.progress_callback = progress_callback
        self.cancel_callback = cancel_callback
        self.temp_dir = None if temp_dir is None else Path(temp_dir).expanduser().resolve()
        self._temp_path: Optional[Path] = None
        self._raw_memmap: Optional[np.memmap] = None
        self._compressed_memmap: Optional[np.memmap] = None
        self._prepared = False

    def __enter__(self) -> "PcdChunkReader":
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def info(self) -> dict:
        h = self.header
        return {
            "path": str(h.path),
            "fields": list(h.fields),
            "sizes": list(h.sizes),
            "types": list(h.types),
            "counts": list(h.counts),
            "data_type": h.data,
            "header_points": int(h.points),
            "chunk_points": int(self.chunk_points),
            "has_native_intensity": any(
                x in h.fields for x in ("intensity", "reflectivity", "i", "intensities")
            ),
            "has_rgb": any(x in h.fields for x in ("rgb", "rgba")),
            "binary_compressed_temp_memmap": bool(h.data == "binary_compressed"),
        }

    def prepare(self) -> None:
        if self._prepared:
            return
        _check_cancel(self.cancel_callback)
        if self.header.data != "binary_compressed":
            self._prepared = True
            return

        with self.path.open("rb") as f:
            f.seek(self.header.data_offset)
            sizes = f.read(8)
            if len(sizes) != 8:
                raise ValueError("binary_compressedサイズヘッダがありません")
            compressed_size, uncompressed_size = struct.unpack("<II", sizes)
            compressed_offset = f.tell()
        available = self.path.stat().st_size - compressed_offset
        if available < compressed_size:
            raise ValueError("binary_compressedデータが途中で終了しています")

        self._compressed_memmap = np.memmap(
            self.path,
            mode="r",
            dtype=np.uint8,
            offset=compressed_offset,
            shape=(int(compressed_size),),
        )
        temp_parent = self.temp_dir
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f".{self.path.stem}.lanegen_decompressed.",
            suffix=".bin",
            dir=None if temp_parent is None else str(temp_parent),
        )
        os.close(fd)
        self._temp_path = Path(name)
        with self._temp_path.open("wb") as f:
            f.truncate(int(uncompressed_size))
        self._raw_memmap = np.memmap(
            self._temp_path,
            mode="r+",
            dtype=np.uint8,
            shape=(int(uncompressed_size),),
        )
        _lzf_decompress_to_memmap(
            self._compressed_memmap,
            self._raw_memmap,
            int(uncompressed_size),
        )
        self._prepared = True

    def close(self) -> None:
        if self._raw_memmap is not None:
            try:
                self._raw_memmap.flush()
            except Exception:
                pass
            self._raw_memmap = None
        self._compressed_memmap = None
        if self._temp_path is not None:
            try:
                self._temp_path.unlink(missing_ok=True)
            finally:
                self._temp_path = None
        self._prepared = False

    def _notify(self, done: int, total: int, phase: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(int(done), int(total), str(phase))

    def iter_chunks(
        self,
        *,
        phase: str = "pcd",
    ) -> Iterator[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        self.prepare()
        h = self.header
        if h.data == "ascii":
            def cb(done: int, total: int, _old_phase: str) -> None:
                self._notify(done, total, phase)
            yield from _iter_ascii(
                h, self.chunk_points, cb, self.cancel_callback
            )
            return
        if h.data == "binary":
            def cb(done: int, total: int, _old_phase: str) -> None:
                self._notify(done, total, phase)
            yield from _iter_binary(
                h, self.chunk_points, cb, self.cancel_callback
            )
            return

        if self._raw_memmap is None:
            raise RuntimeError("binary_compressed temp memmap is not prepared")
        raw = self._raw_memmap
        arrays: dict[str, np.ndarray] = {}
        offset = 0
        for name, size, typ, count in zip(h.fields, h.sizes, h.types, h.counts):
            dt = _numpy_scalar_dtype(typ, size)
            nvals = h.points * count
            nbytes = nvals * dt.itemsize
            if offset + nbytes > len(raw):
                raise ValueError(f"binary_compressed field範囲外: {name}")
            arr = np.ndarray(
                shape=(h.points, count) if count > 1 else (h.points,),
                dtype=dt,
                buffer=raw,
                offset=offset,
            )
            arrays[name] = arr
            offset += nbytes

        n = h.points
        for start in range(0, n, self.chunk_points):
            _check_cancel(self.cancel_callback)
            stop = min(n, start + self.chunk_points)
            fields = {name: arr[start:stop] for name, arr in arrays.items()}
            yield _select_fields(fields)
            self._notify(stop, n, phase)


def _iter_binary_compressed(header: PcdHeader, chunk_points: int,
                            progress_callback: Optional[ProgressCallback],
                            cancel_callback: Optional[CancelCallback],
                            temp_dir: Optional[Path] = None):
    with PcdChunkReader(
        header.path,
        chunk_points=chunk_points,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        temp_dir=temp_dir,
    ) as reader:
        yield from reader.iter_chunks(phase="binary_compressed")


def iter_pcd_chunks(path: Path, chunk_points: int = 500_000,
                    progress_callback: Optional[ProgressCallback] = None,
                    cancel_callback: Optional[CancelCallback] = None,
                    temp_dir: Optional[Path] = None,
                    ) -> Iterator[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Yield filtered XYZ/RGB/intensity chunks from any supported PCD format."""
    with PcdChunkReader(
        path,
        chunk_points=chunk_points,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        temp_dir=temp_dir,
    ) as reader:
        yield from reader.iter_chunks()


def load_pcd(path: Path, chunk_points: int = 500_000,
             progress_callback: Optional[ProgressCallback] = None,
             cancel_callback: Optional[CancelCallback] = None,
             temp_dir: Optional[Path] = None,
             ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], dict]:
    """Compatibility full-load API. Production V6 Stage1 does not use it."""
    with PcdChunkReader(
        path,
        chunk_points=chunk_points,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        temp_dir=temp_dir,
    ) as reader:
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        intensity_parts: list[np.ndarray] = []
        has_rgb = False
        has_intensity = False
        valid_points = 0
        for xyz, rgb, intensity in reader.iter_chunks():
            xyz_parts.append(xyz)
            valid_points += len(xyz)
            if rgb is not None:
                has_rgb = True
                rgb_parts.append(rgb)
            elif has_rgb:
                rgb_parts.append(np.full(len(xyz), np.nan, dtype=np.float32))
            if intensity is not None:
                has_intensity = True
                intensity_parts.append(intensity)
            elif has_intensity:
                intensity_parts.append(np.full(len(xyz), np.nan, dtype=np.float32))
        xyz = np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3), np.float32)
        rgb = np.concatenate(rgb_parts) if has_rgb and rgb_parts else None
        intensity = np.concatenate(intensity_parts) if has_intensity and intensity_parts else None
        info = dict(reader.info)
        info["valid_points"] = int(valid_points)
    print(
        f"[info] PCD {Path(path).name}: {len(xyz):,} points, "
        f"DATA={info['data_type']}, fields={info['fields']}"
    )
    return xyz, rgb, intensity, info


def load_ascii_pcd(path: Path, chunk_points: int = 500_000, **kwargs):
    """Backward-compatible name. V3 accepts all supported DATA modes."""
    return load_pcd(path, chunk_points=chunk_points, **kwargs)
