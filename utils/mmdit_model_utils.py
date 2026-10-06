from collections.abc import Callable
from typing import Any

from models.MMDiT import build_model as build_mmdit
from models.MMDiT_xyz import build_model as build_mmdit_xyz


def _normalize_name(value: object, default: str) -> str:
    normalized = str(value or default).strip().lower().replace("-", "_")
    return normalized or default


def _resolve_mmdit_variant(variant: str) -> tuple[Callable[..., Any], str]:
    if variant == "mmdit":
        return build_mmdit, "MMDiT"
    if variant == "mmdit_xyz":
        return build_mmdit_xyz, "MMDiT_xyz"

    raise ValueError(
        "Unsupported model_variant. Expected MMDiT or MMDiT_xyz; " f"got {variant}."
    )


def resolve_mmdit_factory(
    model_variant: object,
) -> tuple[Callable[..., Any], str]:
    variant = _normalize_name(model_variant, "mmdit")
    return _resolve_mmdit_variant(variant)
