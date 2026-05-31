#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Generate the KV260 firmware bundle expected by xmutil/dfx-mgr after an SD
# card reimage.  Defaults are host-side only; board registration happens only
# when --register is supplied.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OVERLAY="pccx_npu_bd"
BIT_SRC=""
DTS_SRC=""
BUILD_DIR=""
NPU_BASE="0xA0000000"
NPU_SIZE="0x00010000"
REGISTER_ROOT=""
LOAD_APP=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
usage: scripts/kv260/generate_sd_platform_bundle.sh --bit <system.bit> [options]

Options:
  --bit <path>          Vivado-generated .bit file to convert to .bit.bin
  --dts <path>          Optional device-tree overlay source from DTG/XSCT
  --overlay <name>      Application/overlay name (default: pccx_npu_bd)
  --build <dir>         Output directory (default: sw/dtbo/build/<overlay>)
  --npu-base <hex>      Generic-UIO fallback base address (default: 0xA0000000)
  --npu-size <hex>      Generic-UIO fallback aperture size (default: 0x00010000)
  --register <dir>      Install bundle into <dir>/<overlay>, normally /lib/firmware/xilinx
  --load                After --register, run xmutil unloadapp/loadapp <overlay>
  --dry-run             Print register/load commands instead of executing them
  -h, --help            Print this help

The generated bundle contains:
  <overlay>.bit.bin
  <overlay>.dtbo
  shell.json
  manifest.json

Prefer --dts with a DTG/XSCT-generated pl.dtsi.  Without --dts, the script
creates a minimal full-bitstream + generic-uio overlay for the compiled v002
address-map default.
USAGE
}

while (($#)); do
  case "$1" in
    --bit)
      [[ $# -ge 2 ]] || { echo "error: --bit requires a value" >&2; exit 2; }
      BIT_SRC="$2"
      shift 2
      ;;
    --dts)
      [[ $# -ge 2 ]] || { echo "error: --dts requires a value" >&2; exit 2; }
      DTS_SRC="$2"
      shift 2
      ;;
    --overlay)
      [[ $# -ge 2 ]] || { echo "error: --overlay requires a value" >&2; exit 2; }
      OVERLAY="$2"
      shift 2
      ;;
    --build)
      [[ $# -ge 2 ]] || { echo "error: --build requires a value" >&2; exit 2; }
      BUILD_DIR="$2"
      shift 2
      ;;
    --npu-base)
      [[ $# -ge 2 ]] || { echo "error: --npu-base requires a value" >&2; exit 2; }
      NPU_BASE="$2"
      shift 2
      ;;
    --npu-size)
      [[ $# -ge 2 ]] || { echo "error: --npu-size requires a value" >&2; exit 2; }
      NPU_SIZE="$2"
      shift 2
      ;;
    --register)
      [[ $# -ge 2 ]] || { echo "error: --register requires a value" >&2; exit 2; }
      REGISTER_ROOT="$2"
      shift 2
      ;;
    --load)
      LOAD_APP=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${BIT_SRC}" ]]; then
  echo "error: --bit is required" >&2
  usage >&2
  exit 2
fi

if [[ ! "${OVERLAY}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: --overlay must contain only letters, numbers, '.', '_' or '-'" >&2
  exit 2
fi

if [[ ! -f "${BIT_SRC}" ]]; then
  echo "error: bitstream not found: ${BIT_SRC}" >&2
  exit 1
fi

if [[ -n "${DTS_SRC}" && ! -f "${DTS_SRC}" ]]; then
  echo "error: dts source not found: ${DTS_SRC}" >&2
  exit 1
fi

if ! command -v bootgen >/dev/null 2>&1; then
  echo "error: bootgen not found; source the Vitis/Vivado settings64.sh first" >&2
  exit 1
fi

if ! command -v dtc >/dev/null 2>&1; then
  echo "error: dtc not found; install device-tree-compiler on the host or board" >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/sw/dtbo/build/${OVERLAY}}"
mkdir -p "${BUILD_DIR}"

BIT_SRC_ABS="$(readlink -f "${BIT_SRC}")"
DTS_SRC_ABS=""
if [[ -n "${DTS_SRC}" ]]; then
  DTS_SRC_ABS="$(readlink -f "${DTS_SRC}")"
fi

BITBIN="${BUILD_DIR}/${OVERLAY}.bit.bin"
DTS_OUT="${BUILD_DIR}/${OVERLAY}.dts"
DTBO="${BUILD_DIR}/${OVERLAY}.dtbo"
SHELL_JSON="${BUILD_DIR}/shell.json"
MANIFEST_JSON="${BUILD_DIR}/manifest.json"
BIF="${BUILD_DIR}/${OVERLAY}.bootgen.bif"

printf '[1/4] convert .bit to .bit.bin\n'
cat >"${BIF}" <<BIF
all:
{
    [destination_device=pl] "${BIT_SRC_ABS}"
}
BIF
bootgen -w -arch zynqmp -process_bitstream bin -image "${BIF}" -o "${BITBIN}"

printf '[2/4] generate dtbo\n'
if [[ -n "${DTS_SRC_ABS}" ]]; then
  if grep -q 'firmware-name[[:space:]]*=' "${DTS_SRC_ABS}"; then
    sed -E "s/firmware-name[[:space:]]*=[[:space:]]*\"[^\"]*\"/firmware-name = \"${OVERLAY}.bit.bin\"/" \
      "${DTS_SRC_ABS}" >"${DTS_OUT}"
  else
    cp "${DTS_SRC_ABS}" "${DTS_OUT}"
    printf 'warning: %s has no firmware-name property; copied unchanged\n' "${DTS_SRC_ABS}" >&2
  fi
else
  node_addr="$(printf '%s' "${NPU_BASE#0x}" | tr 'A-F' 'a-f')"
  cat >"${DTS_OUT}" <<DTS
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target = <&fpga_full>;
        __overlay__ {
            firmware-name = "${OVERLAY}.bit.bin";
        };
    };

    fragment@1 {
        target-path = "/amba";
        __overlay__ {
            #address-cells = <2>;
            #size-cells = <2>;

            pccx_npu_uio: fabric@${node_addr} {
                compatible = "generic-uio";
                reg = <0x0 ${NPU_BASE} 0x0 ${NPU_SIZE}>;
            };
        };
    };
};
DTS
fi
dtc -@ -O dtb -o "${DTBO}" "${DTS_OUT}"

printf '[3/4] write shell.json and manifest\n'
cat >"${SHELL_JSON}" <<JSON
{
  "shell_type": "XRT_FLAT",
  "num_slots": "1"
}
JSON

bit_sha256="$(sha256sum "${BIT_SRC_ABS}" | awk '{print $1}')"
bundle_sha256="$(sha256sum "${BITBIN}" | awk '{print $1}')"
git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
bitstream_source_name="$(basename "${BIT_SRC_ABS}")"
dt_source_name="generated-generic-uio"
if [[ -n "${DTS_SRC_ABS}" ]]; then
  dt_source_name="$(basename "${DTS_SRC_ABS}")"
fi
cat >"${MANIFEST_JSON}" <<JSON
{
  "overlay": "${OVERLAY}",
  "generated_at_utc": "${generated_at}",
  "git_commit": "${git_commit}",
  "bitstream_source": "${bitstream_source_name}",
  "bitstream_sha256": "${bit_sha256}",
  "bitbin_sha256": "${bundle_sha256}",
  "dt_source": "${dt_source_name}",
  "npu_base": "${NPU_BASE}",
  "npu_size": "${NPU_SIZE}",
  "files": [
    "${OVERLAY}.bit.bin",
    "${OVERLAY}.dtbo",
    "shell.json"
  ]
}
JSON

run_or_print() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

printf '[4/4] bundle ready: %s\n' "${BUILD_DIR}"
if [[ -n "${REGISTER_ROOT}" ]]; then
  APP_DIR="${REGISTER_ROOT%/}/${OVERLAY}"
  printf 'register bundle: %s\n' "${APP_DIR}"
  run_or_print sudo mkdir -p "${APP_DIR}"
  run_or_print sudo install -m 0644 "${BITBIN}" "${APP_DIR}/${OVERLAY}.bit.bin"
  run_or_print sudo install -m 0644 "${DTBO}" "${APP_DIR}/${OVERLAY}.dtbo"
  run_or_print sudo install -m 0644 "${SHELL_JSON}" "${APP_DIR}/shell.json"
  if [[ "${LOAD_APP}" -eq 1 ]]; then
    run_or_print sudo xmutil unloadapp
    run_or_print sudo xmutil loadapp "${OVERLAY}"
  else
    printf 'next: sudo xmutil listapps\n'
    printf 'load only when intended: sudo xmutil loadapp %s\n' "${OVERLAY}"
  fi
fi

printf 'bitbin=%s\n' "${BITBIN}"
printf 'dtbo=%s\n' "${DTBO}"
printf 'shell_json=%s\n' "${SHELL_JSON}"
printf 'manifest=%s\n' "${MANIFEST_JSON}"
