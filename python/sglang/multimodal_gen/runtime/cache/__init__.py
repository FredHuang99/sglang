# SPDX-License-Identifier: Apache-2.0
"""
Cache acceleration module for SGLang-diffusion

This module provides various caching strategies to accelerate
diffusion transformer (DiT) inference:

- TeaCache: Temporal similarity-based caching for diffusion models
- cache-dit integration: Block-level caching with DBCache and TaylorSeer

"""

from importlib import import_module

from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheContext, TeaCacheMixin

_CACHE_DIT_EXPORTS = {
    "CacheDitConfig",
    "enable_cache_on_transformer",
    "enable_cache_on_dual_transformer",
    "get_scm_mask",
}

__all__ = [
    # TeaCache (always available)
    "TeaCacheContext",
    "TeaCacheMixin",
    # cache-dit integration (lazy-loaded, requires cache-dit package)
    "CacheDitConfig",
    "enable_cache_on_transformer",
    "enable_cache_on_dual_transformer",
    "get_scm_mask",
]


def __getattr__(name: str):
    if name in _CACHE_DIT_EXPORTS:
        module = import_module(
            "sglang.multimodal_gen.runtime.cache.cache_dit_integration"
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
