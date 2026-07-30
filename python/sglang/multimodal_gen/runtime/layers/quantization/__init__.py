# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal, get_args

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)

QuantizationMethods = Literal[
    "fp8",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "bitsandbytes",
    "modelslim",
    "mxfp8",
    "mxfp4",
    "mxfp4_npu",
]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

_BUILTIN_METHOD_TO_IMPORT = {
    "modelopt": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8",
        "ModelOptFp8Config",
    ),
    "modelopt_fp8": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
        "ModelOptFp8Config",
    ),
    "modelopt_fp4": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
        "ModelOptFp4Config",
    ),
    "bitsandbytes": (
        "sglang.multimodal_gen.runtime.layers.quantization.bitsandbytes",
        "BitsAndBytesConfig",
    ),
    "modelslim": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelslim",
        "ModelSlimConfig",
    ),
    "fp8": (
        "sglang.multimodal_gen.runtime.layers.quantization.fp8",
        "Fp8Config",
    ),
    "mxfp4": (
        "sglang.multimodal_gen.runtime.layers.quantization.mxfp4",
        "Mxfp4Config",
    ),
    "mxfp8": (
        "sglang.multimodal_gen.runtime.layers.quantization.mxfp8_npu",
        "MXFP8Config",
    ),
    "mxfp4_npu": (
        "sglang.multimodal_gen.runtime.layers.quantization.mxfp4_npu",
        "NPUMXFP4Config",
    ),
}

# Preserve the previously importable class attributes without importing their
# CUDA backends during package initialization.
_CLASS_NAME_TO_IMPORT = {
    attribute: module_and_attribute
    for module_and_attribute in _BUILTIN_METHOD_TO_IMPORT.values()
    for attribute in (module_and_attribute[1],)
}
_CLASS_NAME_TO_IMPORT["ModelOptFp8DiffusionConfig"] = (
    "sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8",
    "ModelOptFp8Config",
)

# Third-party registrations remain eager because callers explicitly register
# them. Built-in backends stay lazy until selected.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG: dict[str, type[QuantizationConfig]] = {}


def _load_quantization_config(
    module_and_attribute: tuple[str, str],
) -> type[QuantizationConfig]:
    module_name, attribute = module_and_attribute
    return getattr(import_module(module_name), attribute)


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.


    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(
                f"The quantization method `{quantization}` is already exists."
            )
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    customized = _CUSTOMIZED_METHOD_TO_QUANT_CONFIG.get(quantization)
    if customized is not None:
        return customized
    return _load_quantization_config(_BUILTIN_METHOD_TO_IMPORT[quantization])


def __getattr__(name: str) -> Any:
    module_and_attribute = _CLASS_NAME_TO_IMPORT.get(name)
    if module_and_attribute is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _load_quantization_config(module_and_attribute)
    globals()[name] = value
    return value


__all__ = [
    "QuantizationMethods",
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
