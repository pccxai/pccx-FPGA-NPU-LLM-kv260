#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

run_valid_handoff() {
  local tmpdir="$1"
  local run_id="$2"
  local program_json="$tmpdir/program.json"
  local program_memh="$tmpdir/program.memh"
  local handoff_json="$tmpdir/handoff.json"
  local validation_json="$tmpdir/validate.json"

  python3 "$ROOT_DIR/tools/v002/generate_smoke_program.py" \
    --preset tiny-shape-lookup \
    --out "$program_json" \
    --memh "$program_memh"
  python3 "$ROOT_DIR/tools/v002/generate_driver_handoff.py" \
    --program-json "$program_json" \
    --memh "$program_memh" \
    --run-id "$run_id" \
    --mode dry-run \
    --out "$handoff_json"
  python3 -m json.tool "$handoff_json" >/dev/null
  python3 "$ROOT_DIR/tools/v002/generate_driver_handoff.py" \
    --validate-handoff "$handoff_json" \
    --mode dry-run \
    --out "$validation_json"
  python3 -m json.tool "$validation_json" >/dev/null
}

run_invalid_handoff() {
  local tmpdir="$1"
  local run_id="$2"
  local bad_memh="$tmpdir/program_bad.memh"
  local program_json="$tmpdir/program.json"

  python3 "$ROOT_DIR/tools/v002/generate_smoke_program.py" \
    --preset tiny-shape-lookup \
    --out "$program_json" \
    --memh "$tmpdir/program.memh"

  awk 'NR == 1 { next } { print }' "$tmpdir/program.memh" >"$bad_memh"

  if python3 "$ROOT_DIR/tools/v002/generate_driver_handoff.py" \
      --program-json "$program_json" \
      --memh "$bad_memh" \
      --run-id "$run_id" \
      --mode dry-run \
      --out "$tmpdir/invalid_handoff.json" >"$tmpdir/invalid_handoff.log" 2>&1; then
    echo "invalid handoff test failed: generator unexpectedly accepted malformed memh" >&2
    exit 1
  fi
}

main() {
  local run_tmpdir
  local run_id

  run_tmpdir="$(mktemp -d)"
  TMP_WORK_DIR="${run_tmpdir}"
  trap 'rm -rf "${TMP_WORK_DIR}"' EXIT
  run_id="test-$(date -u +%Y%m%dT%H%M%SZ)"

  run_valid_handoff "$run_tmpdir" "$run_id"
  run_invalid_handoff "$run_tmpdir" "$run_id"
  echo "runtime handoff tests: PASS"
}

main "$@"
