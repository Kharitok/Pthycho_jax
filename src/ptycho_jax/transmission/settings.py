"""Strict JSON settings for reconstruction scripts, independent of JAX."""

import hashlib
import json
import math
import types
from dataclasses import fields
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key: {name!r}.")
        result[name] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number {value!r} is not supported.")


def _convert(value, annotation, name, base_dir):
    """Type-check JSON without accepting strings as booleans or bools as ints."""
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin in (types.UnionType, Union):
        for candidate in arguments:
            try:
                return _convert(value, candidate, name, base_dir)
            except ValueError:
                pass
    elif annotation is type(None):
        if value is None:
            return None
    elif annotation is Path:
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            return (path if path.is_absolute() else base_dir / path).resolve()
    elif annotation is bool:
        if type(value) is bool:
            return value
    elif annotation is int:
        if type(value) is int:
            return value
    elif annotation is float:
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    elif annotation is str:
        if isinstance(value, str):
            return value
    elif annotation is dict:
        if isinstance(value, dict):
            return value
    elif origin is tuple and isinstance(value, list):
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_convert(v, arguments[0], name, base_dir) for v in value)
        if len(value) == len(arguments):
            return tuple(
                _convert(v, hint, name, base_dir)
                for v, hint in zip(value, arguments, strict=True)
            )
    raise ValueError(f"Invalid JSON value for {name!r}; expected {annotation}.")


def validate_setting_values(settings):
    """Check remaining physical ranges shared by JSON and merged CLI values."""
    for name in (
        "detector_distance_m",
        "detector_pixel_size_m",
        "scan_coords_unit",
        "energy_unit",
    ):
        if name in settings and (
            not math.isfinite(settings[name]) or settings[name] <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")
    for name in (
        "sample_learning_rate",
        "probe_learning_rate",
        "weights_learning_rate",
        "positions_learning_rate",
    ):
        if name in settings and (
            not math.isfinite(settings[name]) or settings[name] < 0
        ):
            raise ValueError(
                f"{name} must be finite and nonnegative; zero disables its gradient updates."
            )
    if "defocus_m" in settings and not math.isfinite(settings["defocus_m"]):
        raise ValueError("defocus_m must be finite.")
    if (
        "relative_probe_threshold" in settings
        and not 0 <= settings["relative_probe_threshold"] < 1
    ):
        raise ValueError("relative_probe_threshold must lie in [0, 1).")
    if "crop_shape" in settings and any(v < 1 for v in settings["crop_shape"]):
        raise ValueError("crop_shape must contain positive integers.")
    if "seed" in settings and settings["seed"] < 0:
        raise ValueError("seed must be nonnegative.")
    if "modes" in settings:
        modes = settings["modes"]
        if not modes or len(set(modes)) != len(modes) or min(modes) < 1:
            raise ValueError("modes must contain distinct positive integers.")
    if settings.get("device") not in (None, "cpu", "gpu"):
        raise ValueError("device must be 'cpu', 'gpu' or null.")


def load_json_settings(path, config_type):
    """Load config fields plus modes/device; return values and exact source bytes.

    Keys use dataclass names, with 'steps' accepted as an alias for 'n_steps'.
    Runtime start timestamps are generated afresh and cannot be supplied here.
    Explicit path values are relative to the settings file; omitted fields keep
    normal script defaults. Domain validation of merged settings remains with
    the reconstruction config, allowing CLI overrides of ordinary JSON values.
    """
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    try:
        values = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON settings in {path}: {error}") from error
    if not isinstance(values, dict):
        raise ValueError("JSON settings must be a single object.")  # noqa: TRY004 - invalid document contents
    if "steps" in values:
        if "n_steps" in values:
            raise ValueError("Use only one of 'n_steps' and its alias 'steps'.")
        values["n_steps"] = values.pop("steps")
    annotations = get_type_hints(config_type)
    allowed = {item.name for item in fields(config_type)} - {"started_utc"}
    annotations = {name: annotations[name] for name in allowed}
    annotations.update(modes=tuple[int, ...], device=str | None)
    unknown = values.keys() - annotations.keys()
    if unknown:
        raise ValueError(f"Unknown JSON settings: {', '.join(sorted(unknown))}.")
    # null tag requests the usual randomized label.
    if values.get("tag", 1) is None:
        del values["tag"]
    converted = {
        name: _convert(value, annotations[name], name, path.parent)
        for name, value in values.items()
    }
    limits = converted.get("scan_coord_limits")
    if limits is not None:
        if set(limits) - {"0", "1"}:
            raise ValueError("scan_coord_limits keys must be '0' (sx) and/or '1' (sy).")
        converted["scan_coord_limits"] = {
            axis: _convert(
                value, tuple[float, float], f"scan_coord_limits.{axis}", path.parent
            )
            for axis, value in limits.items()
        }
    pairs = list((converted.get("scan_coord_limits") or {}).values())
    if converted.get("intensity_limits") is not None:
        pairs.append(converted["intensity_limits"])
    if any(low > high for low, high in pairs):
        raise ValueError("Scan/intensity limits must be ordered [lower, upper] pairs.")
    return converted, {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "content": content,
    }
