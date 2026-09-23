"""
Band cache – keeps track of cropped / processed raster bands.

The cache prevents redundant work when the user requests multiple indices
that share common bands (e.g. NDVI and SAVI both need RED + NIR).

Persistence: a lightweight JSON manifest stored next to the cropped files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "band_cache_manifest.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CachedBand:
    """Metadata for a single cached band file."""

    scene_id: str
    band_name: str
    bbox_hash: str
    path: str  # relative to cache root
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "band_name": self.band_name,
            "bbox_hash": self.bbox_hash,
            "path": self.path,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CachedBand:
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# BandCache
# ─────────────────────────────────────────────────────────────────────────────


class BandCache:
    """
    Track which Landsat bands have already been cropped for a given AOI.

    Usage::

        cache = BandCache(root=Path("data/processed/landsat"))
        if not cache.has_band("LC09_...", "SR_B4", bbox_hash):
            path = processor.crop_band(...)
            cache.register_band("LC09_...", "SR_B4", bbox_hash, path)
        else:
            path = cache.get_band_path("LC09_...", "SR_B4", bbox_hash)
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, CachedBand] = {}
        self._load()

    # ── Key helpers ─────────────────────────────────────────────────────

    @staticmethod
    def make_bbox_hash(bbox: dict[str, float]) -> str:
        """Deterministic hash for a bbox dict ``{west, south, east, north}``."""
        raw = f"{bbox['west']:.6f},{bbox['south']:.6f},{bbox['east']:.6f},{bbox['north']:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _key(scene_id: str, band_name: str, bbox_hash: str) -> str:
        return f"{scene_id}::{band_name}::{bbox_hash}"

    # ── Query ───────────────────────────────────────────────────────────

    def has_band(self, scene_id: str, band_name: str, bbox_hash: str) -> bool:
        """Return True if this band has already been cropped & registered."""
        key = self._key(scene_id, band_name, bbox_hash)
        if key not in self._entries:
            return False
        # verify file still exists on disk
        band_path = self.root / self._entries[key].path
        if not band_path.exists():
            logger.warning("Cached band file missing: %s — removing entry", band_path)
            del self._entries[key]
            self._save()
            return False
        return True

    def get_band_path(
        self, scene_id: str, band_name: str, bbox_hash: str
    ) -> Path | None:
        """Return the absolute path to a cached band, or None."""
        key = self._key(scene_id, band_name, bbox_hash)
        entry = self._entries.get(key)
        if entry is None:
            return None
        abs_path = self.root / entry.path
        return abs_path if abs_path.exists() else None

    def list_bands(self, scene_id: str, bbox_hash: str) -> list[str]:
        """List band names cached for the given scene + bbox."""
        prefix = f"{scene_id}::"
        suffix = f"::{bbox_hash}"
        return [
            e.band_name
            for k, e in self._entries.items()
            if k.startswith(prefix) and k.endswith(suffix)
        ]

    def get_bands_for_index(
        self,
        scene_id: str,
        band_names: list[str],
        bbox_hash: str,
    ) -> dict[str, Path | None]:
        """
        For a set of required band names, return which are cached.

        Returns a dict ``{band_name: Path | None}``.
        """
        return {bn: self.get_band_path(scene_id, bn, bbox_hash) for bn in band_names}

    # ── Mutation ────────────────────────────────────────────────────────

    def register_band(
        self,
        scene_id: str,
        band_name: str,
        bbox_hash: str,
        path: Path,
        width: int = 0,
        height: int = 0,
    ) -> None:
        """Register a newly-cropped band file."""
        try:
            rel_path = path.relative_to(self.root)
        except ValueError:
            rel_path = path
        entry = CachedBand(
            scene_id=scene_id,
            band_name=band_name,
            bbox_hash=bbox_hash,
            path=str(rel_path),
            width=width,
            height=height,
        )
        key = self._key(scene_id, band_name, bbox_hash)
        self._entries[key] = entry
        self._save()
        logger.info("Cached band: %s / %s (bbox=%s)", scene_id, band_name, bbox_hash)

    def clear(self, scene_id: str | None = None) -> int:
        """Remove entries (optionally filter by scene). Returns count removed."""
        if scene_id is None:
            n = len(self._entries)
            self._entries.clear()
        else:
            keys = [k for k in self._entries if k.startswith(f"{scene_id}::")]
            n = len(keys)
            for k in keys:
                del self._entries[k]
        self._save()
        return n

    # ── Persistence ─────────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self.root / _MANIFEST_NAME

    def _load(self) -> None:
        mp = self._manifest_path()
        if not mp.exists():
            self._entries = {}
            return
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            self._entries = {k: CachedBand.from_dict(v) for k, v in data.items()}
            logger.debug("Loaded band cache: %d entries", len(self._entries))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Corrupt band cache manifest – starting fresh: %s", exc)
            self._entries = {}

    def _save(self) -> None:
        mp = self._manifest_path()
        data = {k: v.to_dict() for k, v in self._entries.items()}
        mp.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── Info ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"BandCache(root={self.root!r}, entries={len(self._entries)})"

    def summary(self) -> dict[str, Any]:
        """High-level summary for logging / display."""
        scenes = {e.scene_id for e in self._entries.values()}
        return {
            "total_entries": len(self._entries),
            "unique_scenes": len(scenes),
            "scenes": sorted(scenes),
            "root": str(self.root),
        }

    def list_all_entries(
        self,
        scene_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return every cached entry as a list of dicts.

        Optionally filtered by *scene_id*.
        """
        entries = []
        for e in self._entries.values():
            if scene_id and e.scene_id != scene_id:
                continue
            entries.append(
                {
                    "scene_id": e.scene_id,
                    "band_name": e.band_name,
                    "bbox_hash": e.bbox_hash,
                    "path": str(self.root / e.path),
                    "width": e.width,
                    "height": e.height,
                }
            )
        return entries
