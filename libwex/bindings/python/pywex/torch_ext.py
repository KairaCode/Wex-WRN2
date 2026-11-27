# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Zero-copy PyTorch CPU and CUDA integration for pywex (lazy torch import)."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from ._lib import WexStore


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pywex.torch_ext requires PyTorch (`pip install pywex[torch]` or `pip install torch>=2.1`)."
        ) from exc
    return torch


def _flatten_expert_tensors(
    torch: Any,
    tensors: Sequence[Any],
    n: int,
    arg_name: str,
) -> List[Any]:
    """Accept either flat [3*n] tensors or nested [[m0, m1, m2], ...] per expert."""
    items = list(tensors)
    flat: List[Any] = []
    if len(items) == n and n > 0 and all(isinstance(x, (list, tuple)) for x in items):
        for idx, group in enumerate(items):
            if len(group) != 3:
                raise ValueError(
                    f"{arg_name}[{idx}] must contain 3 tensors (w1, w2, w3), got {len(group)}"
                )
            flat.extend(group)
    else:
        flat = items

    if len(flat) != 3 * n:
        raise ValueError(
            f"{arg_name} must contain 3 * len(eids) = {3 * n} tensors "
            f"(or {n} groups of 3 tensors), got {len(flat)}"
        )
    for idx, t in enumerate(flat):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{arg_name}[{idx}] must be a torch.Tensor, got {type(t).__name__}")
        if not t.is_cuda:
            raise ValueError(f"{arg_name}[{idx}] must be a CUDA tensor (got device={t.device})")
        if not t.is_contiguous():
            raise ValueError(f"{arg_name}[{idx}] must be C-contiguous")
    return flat


def _validate_cuda_slabs(torch: Any, dst_slabs: Sequence[Any], n: int) -> List[Any]:
    slabs = list(dst_slabs)
    if len(slabs) != n:
        raise ValueError(f"dst_slabs length ({len(slabs)}) must match len(eids) ({n})")
    for idx, t in enumerate(slabs):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"dst_slabs[{idx}] must be a torch.Tensor, got {type(t).__name__}")
        if not t.is_cuda:
            raise ValueError(f"dst_slabs[{idx}] must be a CUDA tensor (got device={t.device})")
        if not t.is_contiguous():
            raise ValueError(f"dst_slabs[{idx}] must be C-contiguous")
    return slabs


def fetch_host_tensors(
    store: WexStore,
    layer: int,
    eids: Iterable[int],
    *,
    dtype: Optional[Any] = None,
    scratch: bool = True,
    bypass_l2: bool = False,
    by_record_id: bool = False,
) -> List[Any]:
    """Fetch experts or records directly into contiguous CPU PyTorch tensors (zero copy).

    Each returned tensor has ``dtype=torch.uint8`` and shape ``(decoded_bytes,)`` by
    default. When ``dtype`` is specified, each tensor is returned as a zero-copy
    ``.view(dtype)`` after verifying ``decoded_bytes`` is a multiple of ``dtype``'s
    element size.
    """
    torch = _require_torch()
    eid_list = [int(e) for e in eids]
    if not eid_list:
        return []

    elem_size = 1
    if dtype is not None:
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype or None, got {type(dtype).__name__}")
        elem_size = int(torch.empty(0, dtype=dtype).element_size())

    raw_tensors: List[Any] = []
    ptrs: List[int] = []
    caps: List[int] = []
    for eid in eid_list:
        layout = store.record_layout(layer, eid, by_record_id=by_record_id)
        if dtype is not None and (layout.decoded_bytes % elem_size) != 0:
            raise ValueError(
                f"Record decoded_bytes ({layout.decoded_bytes}) is not divisible by "
                f"{dtype} element size ({elem_size})"
            )
        t = torch.empty((layout.decoded_bytes,), dtype=torch.uint8, device="cpu")
        raw_tensors.append(t)
        ptrs.append(int(t.data_ptr()))
        caps.append(int(t.nbytes))

    store.fetch_host_ptrs(
        layer,
        eid_list,
        ptrs,
        caps,
        scratch=scratch,
        bypass_l2=bypass_l2,
        by_record_id=by_record_id,
    )

    if dtype is not None and dtype != torch.uint8:
        return [t.view(dtype) for t in raw_tensors]
    return raw_tensors


def allocate_expert_cuda_buffers(
    store: WexStore,
    layer: int,
    eid: int,
    *,
    device: Union[str, Any] = "cuda",
    expand_e8m0_f32: bool = False,
    by_record_id: bool = False,
) -> Tuple[List[Any], List[Any]]:
    """Allocate properly shaped ``(scales[3], weights[3])`` CUDA tensors for one expert.

    Matrix order follows libwex convention: ``0=w1/gate``, ``1=w2/down``, ``2=w3/up``.
    When ``expand_e8m0_f32=True``, scale tensors are allocated as ``torch.float32``
    for in-VRAM E8M0->float32 expansion during ``fetch_to_cuda``.
    """
    torch = _require_torch()
    layout = store.record_layout(layer, eid, by_record_id=by_record_id)
    if not layout.is_expert:
        raise ValueError(
            f"Record {layout.record_index} (codec={layout.codec_name}) is not a 6-tensor "
            f"MXFP4/NVFP4 expert record; allocate a contiguous uint8 CUDA slab of "
            f"{layout.decoded_bytes} bytes instead."
        )

    scale_dtype = torch.float32 if expand_e8m0_f32 else torch.uint8
    scales = [
        torch.empty((rows, cols), dtype=scale_dtype, device=device)
        for rows, cols in layout.scale_shapes
    ]
    weights = [
        torch.empty((rows, cols), dtype=torch.uint8, device=device)
        for rows, cols in layout.weight_shapes
    ]
    return scales, weights


def allocate_slab_cuda_buffer(
    store: WexStore,
    layer: int,
    eid: int,
    *,
    device: Union[str, Any] = "cuda",
    by_record_id: bool = False,
) -> Any:
    """Allocate a contiguous 1-D uint8 CUDA slab for non-expert records (WNR4-I4/GG)."""
    torch = _require_torch()
    layout = store.record_layout(layer, eid, by_record_id=by_record_id)
    return torch.empty((layout.decoded_bytes,), dtype=torch.uint8, device=device)


def fetch_to_cuda(
    store: WexStore,
    layer: int,
    eids: Iterable[int],
    *,
    dst_scales: Optional[Sequence[Any]] = None,
    dst_weights: Optional[Sequence[Any]] = None,
    dst_slabs: Optional[Sequence[Any]] = None,
    stream: Optional[Any] = None,
    expand_e8m0_f32: bool = False,
    bypass_l2: bool = False,
    by_record_id: bool = False,
) -> None:
    """Decode experts or records directly into caller PyTorch CUDA tensors (zero copy).

    For 6-tensor MXFP4/NVFP4 expert records (WNR3, WNR4-MX, WNR4-NV), pass
    ``dst_scales`` and ``dst_weights`` (either flat lists of ``3 * len(eids)``
    contiguous CUDA tensors or per-expert lists of 3 tensors). For contiguous
    slab records (WNR4-I4, WNR4-GG), pass ``dst_slabs`` (``len(eids)`` contiguous
    CUDA tensors).
    """
    torch = _require_torch()
    eid_list = [int(e) for e in eids]
    n = len(eid_list)
    if n == 0:
        return

    scale_ptrs: Optional[List[int]] = None
    weight_ptrs: Optional[List[int]] = None
    slab_ptrs: Optional[List[int]] = None
    ref_device = None

    if dst_scales is not None or dst_weights is not None:
        if dst_scales is None or dst_weights is None:
            raise ValueError("Both dst_scales and dst_weights must be provided together")
        flat_scales = _flatten_expert_tensors(torch, dst_scales, n, "dst_scales")
        flat_weights = _flatten_expert_tensors(torch, dst_weights, n, "dst_weights")
        if expand_e8m0_f32:
            for idx, st in enumerate(flat_scales):
                if (int(st.data_ptr()) & 3) != 0:
                    raise ValueError(
                        f"dst_scales[{idx}] pointer must be 4-byte aligned for expand_e8m0_f32=True"
                    )
        scale_ptrs = [int(t.data_ptr()) for t in flat_scales]
        weight_ptrs = [int(t.data_ptr()) for t in flat_weights]
        ref_device = flat_scales[0].device

    if dst_slabs is not None:
        slabs = _validate_cuda_slabs(torch, dst_slabs, n)
        slab_ptrs = [int(t.data_ptr()) for t in slabs]
        if ref_device is None:
            ref_device = slabs[0].device

    if scale_ptrs is None and slab_ptrs is None:
        raise ValueError("Provide either (dst_scales and dst_weights) or dst_slabs")

    if stream is not None:
        stream_ptr = int(stream) if isinstance(stream, int) else int(stream.cuda_stream)
    else:
        stream_ptr = int(torch.cuda.current_stream(ref_device).cuda_stream)

    store.fetch_device_ptrs(
        layer,
        eid_list,
        dst_scales=scale_ptrs,
        dst_weights=weight_ptrs,
        dst_slabs=slab_ptrs,
        stream_ptr=stream_ptr,
        scratch=False,
        bypass_l2=bypass_l2,
        by_record_id=by_record_id,
        expand_e8m0_f32=expand_e8m0_f32,
    )


__all__ = [
    "allocate_expert_cuda_buffers",
    "allocate_slab_cuda_buffer",
    "fetch_host_tensors",
    "fetch_to_cuda",
]
