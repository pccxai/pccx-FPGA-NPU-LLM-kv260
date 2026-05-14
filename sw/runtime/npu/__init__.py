"""PS-side dispatch helpers for the pccx v002 NPU.

The v002 control path sends 64-bit ISA words through AXIL_CMD_IN and moves
bulk tensor data through PS-issued AXI DataMover descriptors.  The six
DataMover command/status helpers stage one 72-bit descriptor as three
32-bit AXI-Lite writes, push it into a small FIFO, and expose the returned
8-bit status stream through a matching FIFO.  High-level calls in this
package keep a NumPy golden reference fallback so the Gemma daemon can run
on hosts without the KV260 bitstream, while the hardware path remains
experimental and golden-vector gated.
"""
from __future__ import annotations

from .npu_core import npu_available, npu_cvo, npu_gemm, npu_gemv, npu_status

__all__ = [
    "npu_available",
    "npu_cvo",
    "npu_gemm",
    "npu_gemv",
    "npu_status",
]
