#!/usr/bin/env bash
# build.sh — thin wrapper around the Vivado TCL flow.
#
#   ./build.sh project   # just create_project.tcl
#   ./build.sh synth     # create_project + synth (OOC)
#   ./build.sh impl      # full impl + write_bitstream (long job)
#   ./build.sh clean     # wipe build/ + generated .jou/.log
#
# Target: Vivado 2023.2+ on Linux. Will attempt Vivado 2025.2 if available.
# Set PCCX_VIVADO_JOBS=1 or 2 on memory-constrained hosts.

set -euo pipefail

HW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$HW_DIR/build"
VIVADO_BIN="$(command -v vivado || true)"

if [[ -z "$VIVADO_BIN" ]]; then
    for candidate in /tools/Xilinx/2025.2/Vivado/bin/vivado \
                     /tools/Xilinx/2024.1/Vivado/bin/vivado \
                     /tools/Xilinx/2023.2/Vivado/bin/vivado; do
        if [[ -x "$candidate" ]]; then
            VIVADO_BIN="$candidate"
            break
        fi
    done
fi

if [[ -z "$VIVADO_BIN" ]]; then
    echo "error: vivado not in PATH and no install found under /tools/Xilinx" >&2
    exit 1
fi

cd "$HW_DIR"

case "${1:-synth}" in
    project)
        mkdir -p "$BUILD_DIR"
        "$VIVADO_BIN" -mode batch \
            -log    "$BUILD_DIR/vivado_project.log" \
            -journal "$BUILD_DIR/vivado_project.jou" \
            -source vivado/create_project.tcl
        ;;
    synth)
        mkdir -p "$BUILD_DIR"
        "$VIVADO_BIN" -mode batch \
            -log    "$BUILD_DIR/vivado_synth.log" \
            -journal "$BUILD_DIR/vivado_synth.jou" \
            -source vivado/create_project.tcl \
            -source vivado/synth.tcl
        ;;
    impl)
        mkdir -p "$BUILD_DIR"
        "$VIVADO_BIN" -mode batch \
            -log    "$BUILD_DIR/vivado_impl.log" \
            -journal "$BUILD_DIR/vivado_impl.jou" \
            -source vivado/impl.tcl
        ;;
    clean)
        rm -rf "$BUILD_DIR"
        rm -f  "$HW_DIR/vivado.jou" "$HW_DIR/vivado.log"
        ;;
    *)
        echo "unknown command: $1" >&2
        echo "usage: $0 {project|synth|impl|clean}" >&2
        exit 1
        ;;
esac
