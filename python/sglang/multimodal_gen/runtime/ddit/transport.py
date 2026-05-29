"""Point-to-point tensor transport helpers for concurrent DDiT."""

from __future__ import annotations

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group

_MAX_TENSOR_DIMS = 16
_DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
    torch.uint8: 5,
    torch.int8: 6,
    torch.int16: 7,
    torch.int32: 8,
    torch.int64: 9,
    torch.bool: 10,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _device() -> torch.device:
    return get_local_torch_device()


def _run_p2p_ops(ops: list[dist.P2POp]) -> None:
    if not ops:
        return
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def _tensor_meta(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in _DTYPE_TO_CODE:
        raise TypeError(f"Unsupported DDiT P2P tensor dtype: {tensor.dtype}")
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) > _MAX_TENSOR_DIMS:
        raise ValueError(f"Tensor rank {len(shape)} exceeds {_MAX_TENSOR_DIMS}")
    meta = torch.zeros(_MAX_TENSOR_DIMS + 2, dtype=torch.int64, device=_device())
    meta[0] = len(shape)
    meta[1] = _DTYPE_TO_CODE[tensor.dtype]
    if shape:
        meta[2 : 2 + len(shape)] = torch.tensor(
            shape, dtype=torch.int64, device=_device()
        )
    return meta


def send_tensor_p2p_many(tensor: torch.Tensor, dsts: tuple[int, ...]) -> None:
    """Send one tensor to multiple ranks with batched NCCL P2P ops."""
    if not dist.is_available() or not dist.is_initialized():
        return
    dsts = tuple(int(dst) for dst in dsts)
    if not dsts:
        return
    group = get_world_group().device_group
    tensor = tensor.detach().contiguous().to(_device())
    meta = _tensor_meta(tensor)
    _run_p2p_ops([dist.P2POp(dist.isend, meta, dst, group) for dst in dsts])
    _run_p2p_ops([dist.P2POp(dist.isend, tensor, dst, group) for dst in dsts])


def send_tensor_p2p(tensor: torch.Tensor, dst: int) -> None:
    """Send a tensor to another rank without involving the world collective."""
    send_tensor_p2p_many(tensor, (dst,))


def recv_tensor_p2p(src: int) -> torch.Tensor:
    """Receive a tensor sent by send_tensor_p2p."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("recv_tensor_p2p requires torch.distributed")
    group = get_world_group().device_group
    meta = torch.empty(_MAX_TENSOR_DIMS + 2, dtype=torch.int64, device=_device())
    _run_p2p_ops([dist.P2POp(dist.irecv, meta, src, group)])
    ndim = int(meta[0].item())
    dtype_code = int(meta[1].item())
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise TypeError(f"Unsupported DDiT P2P tensor dtype code: {dtype_code}")
    shape = tuple(int(dim.item()) for dim in meta[2 : 2 + ndim])
    tensor = torch.empty(shape, dtype=dtype, device=_device())
    _run_p2p_ops([dist.P2POp(dist.irecv, tensor, src, group)])
    return tensor
