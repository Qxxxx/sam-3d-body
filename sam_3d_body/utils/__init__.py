# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import TYPE_CHECKING, Any

__all__ = ["recursive_to"]

if TYPE_CHECKING:
    from .dist import recursive_to


def __getattr__(name: str) -> Any:
    if name == "recursive_to":
        from .dist import recursive_to

        return recursive_to
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
