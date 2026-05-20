"""DMA-safe buffer ownership helpers for PS-issued NPU transfers."""
from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
from pathlib import Path
from typing import BinaryIO


DMA_DEVICE_ENV = "PCCX_NPU_DMA_DEVICE"
DMA_PHYS_ENV = "PCCX_NPU_DMA_PHYS"
DMA_SIZE_ENV = "PCCX_NPU_DMA_SIZE"


class DmaBufferUnavailable(RuntimeError):
    """Raised when no PS DMA buffer provider can expose a physical address."""


@dataclass(frozen=True)
class DmaBufferSlice:
    """A view into one contiguous DMA region."""

    region: "MappedDmaRegion"
    name: str
    offset: int
    size: int
    phys_addr: int

    def write(self, data: bytes | bytearray | memoryview) -> None:
        payload = bytes(data)
        if len(payload) > self.size:
            raise ValueError(f"{self.name} payload exceeds DMA slice size")
        self.region.write(self.offset, payload)

    def read(self, size: int | None = None) -> bytes:
        read_size = self.size if size is None else int(size)
        if read_size < 0 or read_size > self.size:
            raise ValueError(f"{self.name} read exceeds DMA slice size")
        return self.region.read(self.offset, read_size)

    def sync_for_device(self) -> None:
        self.region.sync_for_device(self.offset, self.size)

    def sync_for_cpu(self) -> None:
        self.region.sync_for_cpu(self.offset, self.size)


class MappedDmaRegion:
    """One mmap'ed physically-contiguous DMA region with simple slice allocation."""

    def __init__(
        self,
        *,
        device_path: str | os.PathLike[str],
        phys_addr: int,
        size: int,
        coherent: bool = True,
        sysfs_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if phys_addr < 0 or phys_addr > 0xFFFF_FFFF:
            raise ValueError("DMA physical address must fit in the v12d 32-bit DataMover address field")
        if size <= 0:
            raise ValueError("DMA region size must be positive")
        self.device_path = str(device_path)
        self.phys_addr = int(phys_addr)
        self.size = int(size)
        self.coherent = bool(coherent)
        self.sysfs_path = Path(sysfs_path) if sysfs_path is not None else None
        self._file: BinaryIO | None = None
        self._mm: mmap.mmap | None = None
        self._next_offset = 0

    def __enter__(self) -> "MappedDmaRegion":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._mm is not None:
            return
        self._file = open(self.device_path, "r+b", buffering=0)
        self._mm = mmap.mmap(
            self._file.fileno(),
            self.size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def allocate(self, name: str, size: int, *, alignment: int = 64) -> DmaBufferSlice:
        if size <= 0:
            raise ValueError("DMA slice size must be positive")
        if alignment <= 0:
            raise ValueError("DMA slice alignment must be positive")
        offset = _align_up(self._next_offset, alignment)
        end = offset + int(size)
        if end > self.size:
            raise MemoryError(
                f"DMA region {self.device_path} cannot allocate {name!r}: "
                f"need 0x{end:x}, size 0x{self.size:x}"
            )
        self._next_offset = end
        return DmaBufferSlice(
            region=self,
            name=name,
            offset=offset,
            size=int(size),
            phys_addr=self.phys_addr + offset,
        )

    def write(self, offset: int, data: bytes) -> None:
        mm = self._require_mmap()
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise ValueError("DMA write outside mapped region")
        mm.seek(offset)
        mm.write(data)

    def read(self, offset: int, size: int) -> bytes:
        mm = self._require_mmap()
        end = offset + size
        if offset < 0 or end > self.size:
            raise ValueError("DMA read outside mapped region")
        mm.seek(offset)
        return mm.read(size)

    def sync_for_device(self, offset: int, size: int) -> None:
        self._sync("device", offset, size)

    def sync_for_cpu(self, offset: int, size: int) -> None:
        self._sync("cpu", offset, size)

    def _sync(self, direction: str, offset: int, size: int) -> None:
        if self.coherent or self.sysfs_path is None:
            return
        _write_optional_sysfs(self.sysfs_path / "sync_offset", str(offset))
        _write_optional_sysfs(self.sysfs_path / "sync_size", str(size))
        _write_optional_sysfs(self.sysfs_path / "sync_direction", direction)
        target = "sync_for_device" if direction == "device" else "sync_for_cpu"
        _write_optional_sysfs(self.sysfs_path / target, "1")

    def _require_mmap(self) -> mmap.mmap:
        if self._mm is None:
            raise RuntimeError("DMA region is not open")
        return self._mm


def open_dma_region_from_env() -> MappedDmaRegion:
    """Open a DMA region from env or the first udmabuf-style device."""

    explicit = os.getenv(DMA_DEVICE_ENV)
    if explicit:
        phys_text = os.getenv(DMA_PHYS_ENV)
        size_text = os.getenv(DMA_SIZE_ENV)
        if not phys_text or not size_text:
            raise DmaBufferUnavailable(
                f"{DMA_DEVICE_ENV} requires {DMA_PHYS_ENV} and {DMA_SIZE_ENV}"
            )
        region = MappedDmaRegion(
            device_path=explicit,
            phys_addr=int(phys_text, 0),
            size=int(size_text, 0),
            coherent=_env_truthy("PCCX_NPU_DMA_COHERENT", default=True),
        )
        region.open()
        return region

    discovered = _discover_udmabuf()
    if discovered is None:
        raise DmaBufferUnavailable(
            "no DMA buffer provider found; set PCCX_NPU_DMA_DEVICE, "
            "PCCX_NPU_DMA_PHYS, and PCCX_NPU_DMA_SIZE, or expose /dev/udmabufN"
        )
    discovered.open()
    return discovered


def _discover_udmabuf() -> MappedDmaRegion | None:
    for dev in sorted(Path("/dev").glob("udmabuf*")) + sorted(Path("/dev").glob("u-dma-buf*")):
        sysfs = _udmabuf_sysfs(dev.name)
        if sysfs is None:
            continue
        phys = _read_int_file(sysfs / "phys_addr")
        size = _read_int_file(sysfs / "size")
        if phys is None or size is None:
            continue
        coherent = _read_bool_file(sysfs / "sync_mode")
        return MappedDmaRegion(
            device_path=dev,
            phys_addr=phys,
            size=size,
            coherent=coherent,
            sysfs_path=sysfs,
        )
    return None


def _udmabuf_sysfs(name: str) -> Path | None:
    candidates = (
        Path("/sys/class/u-dma-buf") / name,
        Path("/sys/class/udmabuf") / name,
        Path("/sys/class/misc") / name,
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_int_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def _read_bool_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="ascii").strip().lower()
    except OSError:
        return True
    return text not in {"0", "false", "off", "disabled"}


def _write_optional_sysfs(path: Path, value: str) -> None:
    try:
        path.write_text(value, encoding="ascii")
    except OSError:
        return


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _env_truthy(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
