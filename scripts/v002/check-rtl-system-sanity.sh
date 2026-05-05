#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
# Lightweight RTL relationship checks before expensive Vivado runs.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HW_DIR="$ROOT_DIR/hw"
FILELIST="$HW_DIR/vivado/filelist.f"

failures=0
report_fail() {
    printf 'FAIL: %s\n' "$*"
    failures=$((failures + 1))
}

report_pass() {
    printf 'PASS: %s\n' "$*"
}

if [[ ! -f "$FILELIST" ]]; then
    report_fail "missing filelist: hw/vivado/filelist.f"
    exit 1
fi

python3 - "$ROOT_DIR" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
hw = root / "hw"
filelist = hw / "vivado" / "filelist.f"
failures = 0

def fail(msg):
    global failures
    print(f"FAIL: {msg}")
    failures += 1

def ok(msg):
    print(f"PASS: {msg}")

entries = []
for line in filelist.read_text().splitlines():
    item = line.strip()
    if not item or item.startswith("#"):
        continue
    entries.append(hw / item)

missing = [p for p in entries if not p.exists()]
if missing:
    for path in missing:
        fail(f"filelist entry missing: {path.relative_to(root)}")
else:
    ok(f"filelist entries exist ({len(entries)})")

listed = {p.resolve() for p in entries}
rtl_sv = sorted((hw / "rtl").rglob("*.sv"))
unlisted = [p for p in rtl_sv if p.resolve() not in listed]
if unlisted:
    for path in unlisted:
        fail(f"rtl sv not in filelist: {path.relative_to(root)}")
else:
    ok(f"all RTL .sv files are in filelist ({len(rtl_sv)})")

module_defs = {}
scan_files = rtl_sv + sorted((hw / "vivado").glob("*.sv"))
for path in scan_files:
    text = path.read_text(errors="ignore")
    for match in re.finditer(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b", text, re.MULTILINE):
        module_defs.setdefault(match.group(1), []).append(path)

dups = {name: paths for name, paths in module_defs.items() if len(paths) > 1}
if dups:
    for name, paths in sorted(dups.items()):
        rels = ", ".join(str(p.relative_to(root)) for p in paths)
        fail(f"duplicate module {name}: {rels}")
else:
    ok(f"no duplicate module definitions ({len(module_defs)})")

mem_dispatcher = (hw / "rtl/MEM_control/top/mem_dispatcher.sv").read_text(errors="ignore")
mem_global = (hw / "rtl/MEM_control/memory/mem_GLOBAL_cache.sv").read_text(errors="ignore")
required_dispatcher_terms = [
    ".IN_npu_direct_en",
    ".IN_npu_direct_we",
    ".IN_npu_direct_addr",
    ".IN_npu_direct_wdata",
]
required_global_terms = [
    "IN_npu_direct_en",
    "IN_npu_direct_we",
    "IN_npu_direct_addr",
    "IN_npu_direct_wdata",
    "l2_npu_addr",
]
missing_terms = [term for term in required_dispatcher_terms if term not in mem_dispatcher]
missing_terms += [term for term in required_global_terms if term not in mem_global]
if missing_terms:
    for term in missing_terms:
        fail(f"missing CVO/L2 direct port boundary term: {term}")
else:
    ok("CVO/L2 direct port-B boundary terms present")

sys.exit(1 if failures else 0)
PY
python_status=$?
if (( python_status != 0 )); then
    failures=$((failures + 1))
fi

if (( failures != 0 )); then
    printf 'RTL_SYSTEM_SANITY=FAIL\n'
    exit 1
fi

printf 'RTL_SYSTEM_SANITY=PASS\n'
