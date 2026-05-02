#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT_DIR="${REPO_ROOT}"
EVIDENCE_ROOT="${REPO_ROOT}/docs/evidence/kv260-gemma3n-e4b"
DRY_RUN=0
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
usage: scripts/kv260/run_gemma3n_e4b_smoke.sh [--dry-run]

Options:
  --dry-run   validate host-side handoff inputs and write evidence without contacting the board
  --handoff <path>
              validate a runtime handoff artifact in dry-run mode
  -h, --help  print this help
USAGE
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --handoff)
      if (($# < 2)); then
        echo "error: --handoff requires a value" >&2
        usage >&2
        exit 2
      fi
      export PCCX_RUNTIME_HANDOFF_JSON="$2"
      shift 2
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

PCCX_KV260_HOST="${PCCX_KV260_HOST:-${PCCX_KV260_BOARD_ADDR:-}}"
PCCX_MODEL_DIR="${PCCX_MODEL_DIR:-${PCCX_GEMMA3N_E4B_MODEL_DIR:-}}"
PCCX_BITSTREAM_PATH="${PCCX_BITSTREAM_PATH:-${PCCX_KV260_BITSTREAM:-}}"
PCCX_BOARD_RUNTIME_DIR="${PCCX_BOARD_RUNTIME_DIR:-${PCCX_RUNTIME_DIR:-}}"
PCCX_RUNTIME_HANDOFF_JSON="${PCCX_RUNTIME_HANDOFF_JSON:-${PCCX_RUNTIME_HANDOFF:-}}"
export PCCX_KV260_HOST PCCX_MODEL_DIR PCCX_BITSTREAM_PATH PCCX_BOARD_RUNTIME_DIR
export PCCX_RUNTIME_HANDOFF_JSON

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
GIT_SHORT="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || printf 'nogit')"
GIT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')"
RUN_ID="${PCCX_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${GIT_SHORT}}"
EVIDENCE_REL="docs/evidence/kv260-gemma3n-e4b/${RUN_ID}"
EVIDENCE_DIR="${EVIDENCE_ROOT}/${RUN_ID}"

mkdir -p "${EVIDENCE_DIR}"

HOST_STDOUT_LOG="${EVIDENCE_DIR}/host_stdout.log"
HOST_STDERR_LOG="${EVIDENCE_DIR}/host_stderr.log"
SUMMARY_FILE="${EVIDENCE_DIR}/summary.txt"
BLOCKER_FILE="${EVIDENCE_DIR}/blocker.txt"

exec > >(tee -a "${HOST_STDOUT_LOG}") 2> >(tee -a "${HOST_STDERR_LOG}" >&2)

START_NS="$(date +%s%N)"

RESULT_STATUS="FAIL_RUNTIME"
RUNTIME_PATH="blocked"
RUNTIME_HANDOFF_PATH="${PCCX_RUNTIME_HANDOFF_JSON:-none}"
RUNTIME_HANDOFF_MODE="unknown"
RUNTIME_HANDOFF_TRANSPORT="unknown"
RUNTIME_HANDOFF_BOARD_CONTACTED="no"
RUNTIME_HANDOFF_BOARD_INPUTS="unknown"
RUNTIME_HANDOFF_UNSUPPORTED="unknown"
RUNTIME_HANDOFF_COMMAND_COUNT="unknown"
BOARD_REACHABLE="no"
MODEL_FOUND="no"
BITSTREAM_FOUND="no"
BITSTREAM_LOADED="no"
TOKEN_COUNT="unknown"
TOK_PER_SEC="unknown"
ELAPSED_SEC="unknown"
BLOCKER_REASON=""
REMOTE_RUN_DIR=""
REMOTE_MODEL_DIR=""
REMOTE_BITSTREAM_PATH=""
BITSTREAM_SHA256="unknown"
BITSTREAM_BASENAME="unknown"
MODEL_LOCATION="unknown"

required_env=(
  PCCX_KV260_HOST
  PCCX_KV260_USER
  PCCX_MODEL_DIR
  PCCX_BITSTREAM_PATH
  PCCX_RUN_PROMPT
  PCCX_RUN_TOKENS
  PCCX_BOARD_RUNTIME_DIR
)

remote_quote() {
  local value="$1"
  printf "'%s'" "$(printf '%s' "${value}" | sed "s/'/'\\\\''/g")"
}

escape_sed_pattern() {
  printf '%s' "$1" | sed -e 's/[.[\*^$()+?{}|\/]/\\&/g' -e 's/\]/\\]/g'
}

sanitize_file() {
  local src="$1"
  local dst="$2"
  local value
  cp "${src}" "${dst}"
  for value in \
    "${PCCX_KV260_HOST:-}" \
    "${PCCX_KV260_USER:-}" \
    "${PCCX_MODEL_DIR:-}" \
    "${PCCX_BITSTREAM_PATH:-}" \
    "${PCCX_BOARD_RUNTIME_DIR:-}" \
    "${REMOTE_RUN_DIR:-}" \
    "${REMOTE_MODEL_DIR:-}" \
    "${REMOTE_BITSTREAM_PATH:-}"; do
    if [[ -n "${value}" ]]; then
      local pattern
      pattern="$(escape_sed_pattern "${value}")"
      sed -i "s/${pattern}/<redacted>/g" "${dst}" || true
    fi
  done
}

elapsed_since_start() {
  local end_ns elapsed_ns
  end_ns="$(date +%s%N)"
  elapsed_ns=$((end_ns - START_NS))
  awk -v ns="${elapsed_ns}" 'BEGIN { printf "%.3f", ns / 1000000000 }'
}

write_summary() {
  ELAPSED_SEC="$(elapsed_since_start)"
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'branch=%s\n' "${GIT_BRANCH}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
    printf 'board_interface=%s\n' "${PCCX_KV260_BOARD_IFACE:-unknown}"
    printf 'model_manifest=%s\n' "${PCCX_GEMMA3N_E4B_MANIFEST:-unknown}"
    printf 'command_line=%s %s\n' "$0" "${ORIGINAL_ARGS[*]:-}"
    printf 'result_status=%s\n' "${RESULT_STATUS}"
    printf 'runtime_path=%s\n' "${RUNTIME_PATH}"
    printf 'runtime_handoff_path=%s\n' "${RUNTIME_HANDOFF_PATH}"
    printf 'runtime_handoff_mode=%s\n' "${RUNTIME_HANDOFF_MODE}"
    printf 'runtime_handoff_transport=%s\n' "${RUNTIME_HANDOFF_TRANSPORT}"
    printf 'runtime_handoff_board_inputs=%s\n' "${RUNTIME_HANDOFF_BOARD_INPUTS}"
    printf 'runtime_handoff_unsupported=%s\n' "${RUNTIME_HANDOFF_UNSUPPORTED}"
    printf 'runtime_handoff_command_count=%s\n' "${RUNTIME_HANDOFF_COMMAND_COUNT}"
    printf 'board_reachable=%s\n' "${BOARD_REACHABLE}"
    printf 'model_found=%s\n' "${MODEL_FOUND}"
    printf 'model_location=%s\n' "${MODEL_LOCATION}"
    printf 'bitstream_found=%s\n' "${BITSTREAM_FOUND}"
    printf 'bitstream_loaded=%s\n' "${BITSTREAM_LOADED}"
    printf 'bitstream_basename=%s\n' "${BITSTREAM_BASENAME}"
    printf 'bitstream_sha256=%s\n' "${BITSTREAM_SHA256}"
    printf 'token_count=%s\n' "${TOKEN_COUNT}"
    printf 'tok_per_sec=%s\n' "${TOK_PER_SEC}"
    printf 'elapsed_sec=%s\n' "${ELAPSED_SEC}"
    if [[ -n "${BLOCKER_REASON}" ]]; then
      printf 'blocker_reason=%s\n' "${BLOCKER_REASON}"
    fi
  } >"${SUMMARY_FILE}"
}

finish() {
  local exit_code="$1"
  write_summary
  printf 'summary=%s\n' "${EVIDENCE_REL}/summary.txt"
  printf 'evidence_dir=%s\n' "${EVIDENCE_REL}"
  printf 'result_status=%s\n' "${RESULT_STATUS}"
  case "${RESULT_STATUS}" in
    READY_FOR_BOARD_INPUTS|PASS_*)
      printf 'PASS\n'
      ;;
    BLOCKED_*|BLOCKED_BOARD_INPUTS)
      printf 'BLOCKED\n'
      ;;
    *) printf 'FAIL\n' ;;
  esac
  exit "${exit_code}"
}

block_with() {
  local status="$1"
  local reason="$2"
  shift 2
  RESULT_STATUS="${status}"
  RUNTIME_PATH="blocked"
  BLOCKER_REASON="${reason}"
  {
    printf 'status=%s\n' "${status}"
    printf 'reason=%s\n' "${reason}"
    printf 'instructions:\n'
    local line
    for line in "$@"; do
      printf -- '- %s\n' "${line}"
    done
  } >"${BLOCKER_FILE}"
  printf '%s: %s\n' "${status}" "${reason}" >&2
  finish 2
}

dry_run_blocked() {
  local reason="$1"
  shift
  RESULT_STATUS="BLOCKED_BOARD_INPUTS"
  RUNTIME_PATH="dry_run"
  BLOCKER_REASON="${reason}"
  {
    printf 'status=%s\n' "${RESULT_STATUS}"
    printf 'reason=%s\n' "${reason}"
    printf 'instructions:\n'
    local line
    for line in "$@"; do
      printf -- '- %s\n' "${line}"
    done
  } >"${BLOCKER_FILE}"
  printf 'dry_run_blocked=%s\n' "${reason}"
  finish 0
}

validate_runtime_handoff_dry_run() {
  local handoff_path="${1:-}"
  local validation_json="${EVIDENCE_DIR}/handoff_validation.json"
  local parsed=""

  if [[ -z "${handoff_path}" || ! -f "${handoff_path}" ]]; then
    dry_run_blocked \
      "Missing required handoff artifact for dry-run" \
      "Pass --handoff <path> or PCCX_RUNTIME_HANDOFF_JSON." \
      "The handoff artifact must be generated by tools/v002/generate_driver_handoff.py."
    return 1
  fi

  if ! python3 "$ROOT_DIR/tools/v002/generate_driver_handoff.py" \
      --validate-handoff "${handoff_path}" \
      --mode dry-run \
      --out "${validation_json}"; then
    dry_run_blocked \
      "Runtime handoff artifact validation failed" \
      "Regenerate with tools/v002/generate_driver_handoff.py." \
      "Use --mode dry-run when writing deterministic handoff artifacts for this lane."
    return 1
  fi

  parsed="$(python3 - "${validation_json}" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
mode = payload.get("mode", "unknown")
transport = payload.get("transport", "unknown")
board_inputs = payload.get("boardInputsRequired", []) or []
unsupported = payload.get("unsupportedClaims", []) or []
command_count = payload.get("commandCount", "unknown")
print(mode)
print(transport)
print(",".join(board_inputs) if board_inputs else "none")
print(",".join(unsupported) if unsupported else "none")
print(command_count)
PY
)"
  RUNTIME_HANDOFF_PATH="${handoff_path}"
  RUNTIME_HANDOFF_MODE="$(printf '%s' "${parsed%%$'\n'*}")"
  parsed="${parsed#*$'\n'}"
  RUNTIME_HANDOFF_TRANSPORT="$(printf '%s' "${parsed%%$'\n'*}")"
  parsed="${parsed#*$'\n'}"
  RUNTIME_HANDOFF_BOARD_INPUTS="$(printf '%s' "${parsed%%$'\n'*}")"
  parsed="${parsed#*$'\n'}"
  RUNTIME_HANDOFF_UNSUPPORTED="$(printf '%s' "${parsed%%$'\n'*}")"
  RUNTIME_HANDOFF_COMMAND_COUNT="$(printf '%s' "${parsed#*$'\n'}")"
  {
    printf 'result=%s\n' "READY_FOR_BOARD_INPUTS"
    printf 'handoff_path=%s\n' "${RUNTIME_HANDOFF_PATH}"
    printf 'mode=%s\n' "${RUNTIME_HANDOFF_MODE}"
    printf 'transport=%s\n' "${RUNTIME_HANDOFF_TRANSPORT}"
    printf 'board_inputs=%s\n' "${RUNTIME_HANDOFF_BOARD_INPUTS}"
    printf 'unsupported=%s\n' "${RUNTIME_HANDOFF_UNSUPPORTED}"
    printf 'command_count=%s\n' "${RUNTIME_HANDOFF_COMMAND_COUNT}"
  } >"${EVIDENCE_DIR}/handoff_summary.txt"
  RESULT_STATUS="READY_FOR_BOARD_INPUTS"
  RUNTIME_PATH="dry_run_handoff"
  BLOCKER_REASON=""
  finish 0
}

run_ssh_logged() {
  local name="$1"
  local command="$2"
  local raw_out="${EVIDENCE_DIR}/.${name}.stdout.raw"
  local raw_err="${EVIDENCE_DIR}/.${name}.stderr.raw"
  local clean_out="${EVIDENCE_DIR}/${name}_stdout.log"
  local clean_err="${EVIDENCE_DIR}/${name}_stderr.log"
  set +e
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "${command}" >"${raw_out}" 2>"${raw_err}"
  local rc=$?
  set -e
  sanitize_file "${raw_out}" "${clean_out}"
  sanitize_file "${raw_err}" "${clean_err}"
  rm -f "${raw_out}" "${raw_err}"
  return "${rc}"
}

run_scp_to_logged() {
  local name="$1"
  local src="$2"
  local remote_dst="$3"
  local raw_out="${EVIDENCE_DIR}/.${name}.stdout.raw"
  local raw_err="${EVIDENCE_DIR}/.${name}.stderr.raw"
  local clean_out="${EVIDENCE_DIR}/${name}_stdout.log"
  local clean_err="${EVIDENCE_DIR}/${name}_stderr.log"
  set +e
  scp "${SSH_OPTS[@]}" "${src}" "${SSH_TARGET}:${remote_dst}" >"${raw_out}" 2>"${raw_err}"
  local rc=$?
  set -e
  sanitize_file "${raw_out}" "${clean_out}"
  sanitize_file "${raw_err}" "${clean_err}"
  rm -f "${raw_out}" "${raw_err}"
  return "${rc}"
}

run_scp_from_logged() {
  local name="$1"
  local remote_src="$2"
  local dst="$3"
  local raw_out="${EVIDENCE_DIR}/.${name}.stdout.raw"
  local raw_err="${EVIDENCE_DIR}/.${name}.stderr.raw"
  local raw_dst="${dst}.raw"
  local clean_out="${EVIDENCE_DIR}/${name}_stdout.log"
  local clean_err="${EVIDENCE_DIR}/${name}_stderr.log"
  set +e
  scp "${SSH_OPTS[@]}" "${SSH_TARGET}:${remote_src}" "${raw_dst}" >"${raw_out}" 2>"${raw_err}"
  local rc=$?
  set -e
  sanitize_file "${raw_out}" "${clean_out}"
  sanitize_file "${raw_err}" "${clean_err}"
  if [[ "${rc}" -eq 0 ]]; then
    sanitize_file "${raw_dst}" "${dst}"
  fi
  rm -f "${raw_out}" "${raw_err}" "${raw_dst}"
  return "${rc}"
}

write_remote_smoke_script() {
  local dst="$1"
  cat >"${dst}" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

: "${MODEL_DIR:?}"
: "${RUN_PROMPT:?}"
: "${RUN_TOKENS:?}"
: "${BOARD_RUNTIME_DIR:?}"
: "${REMOTE_RUN_DIR:?}"
: "${BITSTREAM_PATH:?}"

mkdir -p "${REMOTE_RUN_DIR}"

RESULT_FILE="${REMOTE_RUN_DIR}/result.env"
GENERATED_FILE="${REMOTE_RUN_DIR}/generated_output.txt"
RUNTIME_STDOUT="${REMOTE_RUN_DIR}/runtime_stdout.log"
RUNTIME_STDERR="${REMOTE_RUN_DIR}/runtime_stderr.log"

write_result() {
  local status="$1"
  local runtime_path="$2"
  local token_count="$3"
  local elapsed_sec="$4"
  local tok_per_sec="$5"
  local reason="${6:-}"
  {
    printf 'RESULT_STATUS=%s\n' "${status}"
    printf 'RUNTIME_PATH=%s\n' "${runtime_path}"
    printf 'TOKEN_COUNT=%s\n' "${token_count}"
    printf 'ELAPSED_SEC=%s\n' "${elapsed_sec}"
    printf 'TOK_PER_SEC=%s\n' "${tok_per_sec}"
    if [[ -n "${reason}" ]]; then
      printf 'BLOCKER_REASON=%s\n' "${reason}"
    fi
  } >"${RESULT_FILE}"
}

elapsed_seconds() {
  local start_ns="$1"
  local end_ns="$2"
  awk -v s="${start_ns}" -v e="${end_ns}" 'BEGIN { printf "%.3f", (e - s) / 1000000000 }'
}

metric_value() {
  local key="$1"
  awk -F= -v key="${key}" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "${RUNTIME_STDOUT}" "${RUNTIME_STDERR}" 2>/dev/null || true
}

first_existing_gguf() {
  find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.gguf' 2>/dev/null | sort | head -n 1
}

run_npu_candidate() {
  local start_ns end_ns rc elapsed token_count tok_per_sec cmd_desc reported_status reported_reason
  export MODEL_DIR RUN_PROMPT RUN_TOKENS BITSTREAM_PATH BOARD_RUNTIME_DIR REMOTE_RUN_DIR
  start_ns="$(date +%s%N)"
  set +e
  if [[ -n "${PCCX_REMOTE_RUN_CMD:-}" ]]; then
    cmd_desc="PCCX_REMOTE_RUN_CMD"
    bash -lc "${PCCX_REMOTE_RUN_CMD}" >"${RUNTIME_STDOUT}" 2>"${RUNTIME_STDERR}"
    rc=$?
  elif [[ -x "${BOARD_RUNTIME_DIR}/run_gemma3n_e4b_npu.sh" ]]; then
    cmd_desc="${BOARD_RUNTIME_DIR}/run_gemma3n_e4b_npu.sh"
    "${BOARD_RUNTIME_DIR}/run_gemma3n_e4b_npu.sh" \
      --model "${MODEL_DIR}" \
      --max-new-tokens "${RUN_TOKENS}" \
      --input "${RUN_PROMPT}" >"${RUNTIME_STDOUT}" 2>"${RUNTIME_STDERR}"
    rc=$?
  elif [[ -x "${BOARD_RUNTIME_DIR}/runtime/run_gemma3n_e4b_npu.sh" ]]; then
    cmd_desc="${BOARD_RUNTIME_DIR}/runtime/run_gemma3n_e4b_npu.sh"
    "${BOARD_RUNTIME_DIR}/runtime/run_gemma3n_e4b_npu.sh" \
      --model "${MODEL_DIR}" \
      --max-new-tokens "${RUN_TOKENS}" \
      --input "${RUN_PROMPT}" >"${RUNTIME_STDOUT}" 2>"${RUNTIME_STDERR}"
    rc=$?
  elif [[ -x "${BOARD_RUNTIME_DIR}/bin/gemma3n_e4b_npu" ]]; then
    cmd_desc="${BOARD_RUNTIME_DIR}/bin/gemma3n_e4b_npu"
    "${BOARD_RUNTIME_DIR}/bin/gemma3n_e4b_npu" \
      --model "${MODEL_DIR}" \
      --max-new-tokens "${RUN_TOKENS}" \
      --input "${RUN_PROMPT}" >"${RUNTIME_STDOUT}" 2>"${RUNTIME_STDERR}"
    rc=$?
  elif [[ -x "${BOARD_RUNTIME_DIR}/gemma3n_e4b_npu" ]]; then
    cmd_desc="${BOARD_RUNTIME_DIR}/gemma3n_e4b_npu"
    "${BOARD_RUNTIME_DIR}/gemma3n_e4b_npu" \
      --model "${MODEL_DIR}" \
      --max-new-tokens "${RUN_TOKENS}" \
      --input "${RUN_PROMPT}" >"${RUNTIME_STDOUT}" 2>"${RUNTIME_STDERR}"
    rc=$?
  else
    set -e
    return 3
  fi
  set -e
  end_ns="$(date +%s%N)"
  elapsed="$(elapsed_seconds "${start_ns}" "${end_ns}")"
  reported_status="$(metric_value RESULT_STATUS)"
  reported_reason="$(metric_value BLOCKER_REASON)"

  if [[ "${reported_status}" == BLOCKED_* ]]; then
    write_result "${reported_status}" "FPGA_NPU" "unknown" "${elapsed}" "unknown" "${reported_reason:-NPU runtime reported ${reported_status}}"
    return 42
  fi

  if [[ "${rc}" -ne 0 ]]; then
    write_result "BLOCKED_RUNTIME" "FPGA_NPU" "unknown" "${elapsed}" "unknown" "NPU runtime command failed: ${cmd_desc}"
    return "${rc}"
  fi

  token_count="$(metric_value TOKEN_COUNT)"
  if [[ -z "${token_count}" ]]; then
    token_count="$(metric_value TOKENS_GENERATED)"
  fi
  if [[ -z "${token_count}" ]]; then
    token_count="$(metric_value NEW_TOKENS)"
  fi
  tok_per_sec="$(metric_value TOK_PER_SEC)"
  if [[ -z "${tok_per_sec}" ]]; then
    tok_per_sec="$(metric_value TOKENS_PER_SECOND)"
  fi
  if [[ -z "${tok_per_sec}" && "${token_count}" =~ ^[0-9]+$ ]]; then
    tok_per_sec="$(awk -v t="${token_count}" -v e="${elapsed}" 'BEGIN { if (e > 0) printf "%.3f", t / e; else print "unknown" }')"
  fi

  if [[ -z "${token_count}" ]]; then
    write_result "BLOCKED_RUNTIME" "FPGA_NPU" "unknown" "${elapsed}" "unknown" "NPU runtime exited without TOKEN_COUNT, TOKENS_GENERATED, or NEW_TOKENS"
    return 44
  fi

  cp "${RUNTIME_STDOUT}" "${GENERATED_FILE}" 2>/dev/null || true
  write_result "PASS_KV260_NPU" "FPGA_NPU" "${token_count}" "${elapsed}" "${tok_per_sec:-unknown}" ""
  return 0
}

run_transformers_fallback() {
  local start_ns end_ns rc elapsed token_count tok_per_sec
  if ! command -v python3 >/dev/null 2>&1; then
    return 3
  fi

  export MODEL_DIR RUN_PROMPT RUN_TOKENS GENERATED_FILE
  start_ns="$(date +%s%N)"
  set +e
  python3 - <<'PY' >"${RUNTIME_STDOUT}" 2>"${RUNTIME_STDERR}"
import os
import sys
import time

model_dir = os.environ["MODEL_DIR"]
prompt_text = os.environ["RUN_PROMPT"]
max_new_tokens = int(os.environ["RUN_TOKENS"])
generated_file = os.environ["GENERATED_FILE"]
trust_remote_code = os.environ.get("PCCX_TRUST_REMOTE_CODE", "0") == "1"

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:
    print(f"DEPENDENCY_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(40)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt")
    started = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    input_tokens = int(inputs["input_ids"].shape[-1])
    new_tokens = int(output_ids.shape[-1] - input_tokens)
    generated = tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True)
    with open(generated_file, "w", encoding="utf-8") as handle:
        handle.write(generated)
        handle.write("\n")
    print("PCCX_OUTPUT_BEGIN")
    print(generated)
    print("PCCX_OUTPUT_END")
    print(f"TOKEN_COUNT={new_tokens}")
    print(f"ELAPSED_SEC={elapsed:.3f}")
    if elapsed > 0:
        print(f"TOK_PER_SEC={new_tokens / elapsed:.3f}")
    else:
        print("TOK_PER_SEC=unknown")
except Exception as exc:
    print(f"RUNTIME_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(41)
PY
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  elapsed="$(elapsed_seconds "${start_ns}" "${end_ns}")"

  if [[ "${rc}" -ne 0 ]]; then
    write_result "BLOCKED_RUNTIME" "PS_FALLBACK" "unknown" "${elapsed}" "unknown" "Transformers PS fallback failed"
    return "${rc}"
  fi

  token_count="$(metric_value TOKEN_COUNT)"
  tok_per_sec="$(metric_value TOK_PER_SEC)"
  if [[ -z "${token_count}" ]]; then
    write_result "BLOCKED_RUNTIME" "PS_FALLBACK" "unknown" "${elapsed}" "unknown" "Transformers PS fallback did not report token count"
    return 45
  fi
  write_result "PASS_KV260_FALLBACK" "PS_FALLBACK" "${token_count}" "${elapsed}" "${tok_per_sec:-unknown}" ""
  return 0
}

run_llama_fallback() {
  local llama_bin gguf start_ns end_ns rc elapsed token_count tok_per_sec
  llama_bin="$(command -v llama-cli 2>/dev/null || command -v llama 2>/dev/null || command -v main 2>/dev/null || true)"
  if [[ -z "${llama_bin}" ]]; then
    return 3
  fi
  gguf="$(first_existing_gguf)"
  if [[ -z "${gguf}" ]]; then
    return 3
  fi

  start_ns="$(date +%s%N)"
  set +e
  "${llama_bin}" -m "${gguf}" -p "${RUN_PROMPT}" -n "${RUN_TOKENS}" --no-display-prompt \
    >"${RUNTIME_STDOUT}" 2>"${RUNTIME_STDERR}"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"
  elapsed="$(elapsed_seconds "${start_ns}" "${end_ns}")"
  if [[ "${rc}" -ne 0 ]]; then
    write_result "BLOCKED_RUNTIME" "PS_FALLBACK" "unknown" "${elapsed}" "unknown" "llama.cpp PS fallback failed"
    return "${rc}"
  fi

  cp "${RUNTIME_STDOUT}" "${GENERATED_FILE}" 2>/dev/null || true
  token_count="$(metric_value TOKEN_COUNT)"
  tok_per_sec="$(metric_value TOK_PER_SEC)"
  if [[ -z "${tok_per_sec}" ]]; then
    tok_per_sec="$(awk '
      /tokens per second|tok\/s/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^[0-9]+(\.[0-9]+)?$/) {
            value = $i
          }
        }
      }
      END { if (value != "") print value }
    ' "${RUNTIME_STDERR}" "${RUNTIME_STDOUT}" 2>/dev/null || true)"
  fi
  if [[ -z "${token_count}" ]]; then
    token_count="$(awk '
      /sample time/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "runs" && (i - 1) > 0 && $(i - 1) ~ /^[0-9]+$/) {
            print $(i - 1)
            exit
          }
        }
      }
    ' "${RUNTIME_STDERR}" "${RUNTIME_STDOUT}" 2>/dev/null || true)"
  fi
  if [[ -z "${token_count}" ]]; then
    write_result "BLOCKED_RUNTIME" "PS_FALLBACK" "unknown" "${elapsed}" "unknown" "llama.cpp fallback ran but did not report generated token count"
    return 45
  fi
  write_result "PASS_KV260_FALLBACK" "PS_FALLBACK" "${token_count}" "${elapsed}" "${tok_per_sec:-unknown}" ""
  return 0
}

if run_npu_candidate; then
  exit 0
fi

if [[ -f "${RESULT_FILE}" ]] && grep -q '^RESULT_STATUS=BLOCKED_RUNTIME$' "${RESULT_FILE}"; then
  exit 42
fi

if run_transformers_fallback; then
  exit 0
fi

if [[ -f "${RESULT_FILE}" ]] && grep -q '^RESULT_STATUS=BLOCKED_RUNTIME$' "${RESULT_FILE}"; then
  exit 42
fi

if run_llama_fallback; then
  exit 0
fi

write_result "BLOCKED_RUNTIME" "blocked" "unknown" "0.000" "unknown" "No NPU runner found and no usable PS fallback runtime found"
exit 42
REMOTE_SCRIPT
  chmod +x "${dst}"
}

result_value() {
  local key="$1"
  local file="$2"
  awk -F= -v key="${key}" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "${file}" 2>/dev/null || true
}

printf 'run_id=%s\n' "${RUN_ID}"
printf 'git_commit=%s\n' "${GIT_COMMIT}"
printf 'branch=%s\n' "${GIT_BRANCH}"
printf 'evidence_dir=%s\n' "${EVIDENCE_REL}"

if (( DRY_RUN )) && [[ -n "${PCCX_RUNTIME_HANDOFF_JSON}" ]]; then
  validate_runtime_handoff_dry_run "${PCCX_RUNTIME_HANDOFF_JSON}"
fi

missing=()
for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("${name}")
  fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
  missing_csv="$(IFS=,; printf '%s' "${missing[*]}")"
  if (( DRY_RUN )); then
    dry_run_blocked \
      "Missing required dry-run handoff inputs: ${missing_csv}" \
      "Set PCCX_KV260_HOST or PCCX_KV260_BOARD_ADDR for the board address." \
      "Set PCCX_KV260_USER for the board SSH user." \
      "Set PCCX_MODEL_DIR or PCCX_GEMMA3N_E4B_MODEL_DIR for the external model directory." \
      "Set PCCX_BITSTREAM_PATH or PCCX_KV260_BITSTREAM for the intended bitstream." \
      "Set PCCX_RUN_PROMPT, PCCX_RUN_TOKENS, and PCCX_BOARD_RUNTIME_DIR."
  fi
  for name in "${missing[@]}"; do
    case "${name}" in
      PCCX_KV260_HOST|PCCX_KV260_USER)
        block_with \
          "BLOCKED_BOARD" \
          "Missing required board SSH environment: ${missing_csv}" \
          "Export PCCX_KV260_HOST with the KV260 hostname or IP." \
          "Export PCCX_KV260_USER with the SSH user configured for key-based login." \
          "Re-run this script after the board is powered, booted, and reachable over SSH."
        ;;
    esac
  done
  for name in "${missing[@]}"; do
    case "${name}" in
      PCCX_MODEL_DIR)
        block_with \
          "BLOCKED_MODEL" \
          "Missing required model environment: ${missing_csv}" \
          "Export PCCX_MODEL_DIR to a readable Gemma 3N E4B model directory available on the KV260." \
          "If the model is only on the host, set PCCX_STAGE_MODEL_TO_BOARD=1 to copy it into PCCX_BOARD_RUNTIME_DIR/model."
        ;;
    esac
  done
  for name in "${missing[@]}"; do
    case "${name}" in
      PCCX_BITSTREAM_PATH)
        block_with \
          "BLOCKED_BITSTREAM" \
          "Missing required bitstream environment: ${missing_csv}" \
          "Export PCCX_BITSTREAM_PATH to the local or board-side bitstream file." \
          "Use PCCX_SKIP_BITSTREAM_LOAD=1 only when the intended image is already loaded and this run should not reprogram PL."
        ;;
    esac
  done
  block_with \
    "BLOCKED_RUNTIME" \
    "Missing required runtime environment: ${missing_csv}" \
    "Export PCCX_RUN_PROMPT with the smoke input text." \
    "Export PCCX_RUN_TOKENS with a positive integer max new token count." \
    "Export PCCX_BOARD_RUNTIME_DIR with the writable runtime directory on the KV260."
fi

printf '%s\n' "${PCCX_RUN_PROMPT}" >"${EVIDENCE_DIR}/input.txt"
printf '%s\n' "${PCCX_RUN_TOKENS}" >"${EVIDENCE_DIR}/requested_tokens.txt"

if ! [[ "${PCCX_RUN_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  if (( DRY_RUN )); then
    dry_run_blocked \
      "PCCX_RUN_TOKENS must be a positive integer" \
      "Set PCCX_RUN_TOKENS to a small positive integer for smoke testing, for example 8."
  fi
  block_with \
    "BLOCKED_RUNTIME" \
    "PCCX_RUN_TOKENS must be a positive integer" \
    "Set PCCX_RUN_TOKENS to a small positive integer for smoke testing, for example 8."
fi

if (( DRY_RUN )); then
  RESULT_STATUS="PASS_DRY_RUN_READY"
  RUNTIME_PATH="dry_run"
  BLOCKER_REASON="Dry-run did not contact the KV260 and did not execute hardware."
  {
    printf 'status=%s\n' "${RESULT_STATUS}"
    printf 'reason=%s\n' "${BLOCKER_REASON}"
    printf 'instructions:\n'
    printf -- '- Re-run without --dry-run only after board, model, bitstream, and runtime paths are intentional.\n'
    printf -- '- Use an available board interface such as USB serial/JTAG or Ethernet, depending on the target board.\n'
  } >"${BLOCKER_FILE}"
  finish 0
fi

SSH_TARGET="${PCCX_KV260_USER}@${PCCX_KV260_HOST}"
SSH_CONNECT_TIMEOUT="${PCCX_SSH_CONNECT_TIMEOUT:-10}"
SSH_STRICT_HOSTKEY_CHECKING="${PCCX_SSH_STRICT_HOSTKEY_CHECKING:-accept-new}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
  -o StrictHostKeyChecking="${SSH_STRICT_HOSTKEY_CHECKING}"
)

REMOTE_BASE="${PCCX_BOARD_RUNTIME_DIR%/}"
REMOTE_RUN_DIR="${REMOTE_BASE}/runs/${RUN_ID}"

printf 'checking_board_reachability=1\n'
if ! run_ssh_logged "board_probe" "printf 'kv260_ssh_ok\n'"; then
  block_with \
    "BLOCKED_BOARD" \
    "KV260 SSH probe failed" \
    "Verify board power, network route, and SSH key login for PCCX_KV260_HOST/PCCX_KV260_USER." \
    "Check logs: ${EVIDENCE_REL}/board_probe_stderr.log" \
    "Re-run after the command 'ssh \$PCCX_KV260_USER@\$PCCX_KV260_HOST true' succeeds without an interactive password."
fi
BOARD_REACHABLE="yes"
RUNTIME_HANDOFF_BOARD_CONTACTED="yes"

if ! run_ssh_logged "remote_mkdir" "mkdir -p $(remote_quote "${REMOTE_RUN_DIR}") $(remote_quote "${REMOTE_BASE}/bitstreams")"; then
  block_with \
    "BLOCKED_RUNTIME" \
    "Could not create KV260 runtime evidence directory" \
    "Ensure PCCX_BOARD_RUNTIME_DIR is writable by PCCX_KV260_USER." \
    "Check logs: ${EVIDENCE_REL}/remote_mkdir_stderr.log"
fi

printf 'checking_model_path=1\n'
LOCAL_MODEL_PRESENT="no"
REMOTE_MODEL_PRESENT="no"
if [[ -d "${PCCX_MODEL_DIR}" ]]; then
  LOCAL_MODEL_PRESENT="yes"
  find "${PCCX_MODEL_DIR}" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort >"${EVIDENCE_DIR}/model_manifest_local.txt" || true
fi

if run_ssh_logged "model_probe" "test -d $(remote_quote "${PCCX_MODEL_DIR}")"; then
  REMOTE_MODEL_PRESENT="yes"
  REMOTE_MODEL_DIR="${PCCX_MODEL_DIR}"
fi

if [[ "${REMOTE_MODEL_PRESENT}" != "yes" && "${LOCAL_MODEL_PRESENT}" == "yes" && "${PCCX_STAGE_MODEL_TO_BOARD:-0}" == "1" ]]; then
  REMOTE_MODEL_DIR="${REMOTE_BASE}/model"
  if ! run_ssh_logged "model_stage_mkdir" "mkdir -p $(remote_quote "${REMOTE_MODEL_DIR}")"; then
    block_with \
      "BLOCKED_MODEL" \
      "Could not create KV260 model staging directory" \
      "Ensure PCCX_BOARD_RUNTIME_DIR is writable and has enough free space for the model."
  fi
  if command -v rsync >/dev/null 2>&1; then
    RSYNC_RSH="ssh ${SSH_OPTS[*]}"
    export RSYNC_RSH
    model_stage_stdout_raw="${EVIDENCE_DIR}/.model_stage.stdout.raw"
    model_stage_stderr_raw="${EVIDENCE_DIR}/.model_stage.stderr.raw"
    if ! rsync -az --delete --exclude='.git' "${PCCX_MODEL_DIR%/}/" "${SSH_TARGET}:${REMOTE_MODEL_DIR}/" \
      >"${model_stage_stdout_raw}" 2>"${model_stage_stderr_raw}"; then
      sanitize_file "${model_stage_stdout_raw}" "${EVIDENCE_DIR}/model_stage_stdout.log"
      sanitize_file "${model_stage_stderr_raw}" "${EVIDENCE_DIR}/model_stage_stderr.log"
      rm -f "${model_stage_stdout_raw}" "${model_stage_stderr_raw}"
      block_with \
        "BLOCKED_MODEL" \
        "Model staging to KV260 failed" \
        "Check model_stage_stderr.log and confirm enough free space under PCCX_BOARD_RUNTIME_DIR."
    fi
    sanitize_file "${model_stage_stdout_raw}" "${EVIDENCE_DIR}/model_stage_stdout.log"
    sanitize_file "${model_stage_stderr_raw}" "${EVIDENCE_DIR}/model_stage_stderr.log"
    rm -f "${model_stage_stdout_raw}" "${model_stage_stderr_raw}"
  else
    block_with \
      "BLOCKED_MODEL" \
      "Model exists on host but rsync is unavailable for staging" \
      "Install rsync on the host or place the model directory directly on the KV260 at PCCX_MODEL_DIR."
  fi
  REMOTE_MODEL_PRESENT="yes"
fi

if [[ "${REMOTE_MODEL_PRESENT}" != "yes" ]]; then
  if [[ "${LOCAL_MODEL_PRESENT}" == "yes" ]]; then
    block_with \
      "BLOCKED_MODEL" \
      "Model directory exists on host but is not available on the KV260" \
      "Place the Gemma 3N E4B model at PCCX_MODEL_DIR on the KV260, or set PCCX_STAGE_MODEL_TO_BOARD=1 to stage the host directory." \
      "Do not commit model weights or private model paths."
  fi
  block_with \
    "BLOCKED_MODEL" \
    "Gemma 3N E4B model directory was not found" \
    "Set PCCX_MODEL_DIR to a Gemma 3N E4B model directory that exists on the KV260." \
    "If the model is on the host, set PCCX_STAGE_MODEL_TO_BOARD=1 and ensure the board has enough storage."
fi

MODEL_FOUND="yes"
MODEL_LOCATION="board"
if [[ "${LOCAL_MODEL_PRESENT}" == "yes" && "${PCCX_STAGE_MODEL_TO_BOARD:-0}" == "1" ]]; then
  MODEL_LOCATION="host_staged_to_board"
fi

if ! run_ssh_logged "model_file_probe" "d=$(remote_quote "${REMOTE_MODEL_DIR}"); test -f \"\$d/config.json\" || ls \"\$d\"/*.gguf \"\$d\"/*.safetensors \"\$d\"/*.bin >/dev/null 2>&1"; then
  block_with \
    "BLOCKED_MODEL" \
    "Model directory exists but no recognizable model files were found" \
    "Verify PCCX_MODEL_DIR contains config.json plus weights, or a GGUF model file." \
    "Check logs: ${EVIDENCE_REL}/model_file_probe_stderr.log"
fi

run_ssh_logged "model_manifest_remote" "find $(remote_quote "${REMOTE_MODEL_DIR}") -maxdepth 1 -type f -exec basename {} \\; | sort" || true

printf 'checking_bitstream_path=1\n'
if [[ -f "${PCCX_BITSTREAM_PATH}" ]]; then
  BITSTREAM_FOUND="host"
  BITSTREAM_BASENAME="$(basename "${PCCX_BITSTREAM_PATH}")"
  BITSTREAM_SHA256="$(sha256sum "${PCCX_BITSTREAM_PATH}" | awk '{print $1}')"
  REMOTE_BITSTREAM_PATH="${REMOTE_BASE}/bitstreams/${RUN_ID}/${BITSTREAM_BASENAME}"
  if ! run_ssh_logged "bitstream_mkdir" "mkdir -p $(remote_quote "${REMOTE_BASE}/bitstreams/${RUN_ID}")"; then
    block_with \
      "BLOCKED_BITSTREAM" \
      "Could not create KV260 bitstream staging directory" \
      "Ensure PCCX_BOARD_RUNTIME_DIR is writable by PCCX_KV260_USER."
  fi
  if ! run_scp_to_logged "bitstream_stage" "${PCCX_BITSTREAM_PATH}" "${REMOTE_BITSTREAM_PATH}"; then
    block_with \
      "BLOCKED_BITSTREAM" \
      "Could not copy bitstream to KV260" \
      "Check logs: ${EVIDENCE_REL}/bitstream_stage_stderr.log"
  fi
elif run_ssh_logged "bitstream_probe" "test -f $(remote_quote "${PCCX_BITSTREAM_PATH}")"; then
  BITSTREAM_FOUND="board"
  REMOTE_BITSTREAM_PATH="${PCCX_BITSTREAM_PATH}"
  BITSTREAM_BASENAME="$(basename "${PCCX_BITSTREAM_PATH}")"
  run_ssh_logged "bitstream_sha256_remote" "sha256sum $(remote_quote "${REMOTE_BITSTREAM_PATH}") | awk '{print \$1}'" || true
  if [[ -s "${EVIDENCE_DIR}/bitstream_sha256_remote_stdout.log" ]]; then
    BITSTREAM_SHA256="$(awk '{print $1; exit}' "${EVIDENCE_DIR}/bitstream_sha256_remote_stdout.log")"
  fi
else
  block_with \
    "BLOCKED_BITSTREAM" \
    "Bitstream file was not found on host or KV260" \
    "Set PCCX_BITSTREAM_PATH to a readable local bitstream file or a readable board-side bitstream file." \
    "Use PCCX_SKIP_BITSTREAM_LOAD=1 only when the intended bitstream is already loaded."
fi

if [[ "${PCCX_SKIP_BITSTREAM_LOAD:-0}" == "1" ]]; then
  BITSTREAM_LOADED="skipped"
else
  printf 'loading_bitstream=1\n'
  bitstream_load_cmd=$(cat <<BITSTREAM_LOAD
set -eu
p=$(remote_quote "${REMOTE_BITSTREAM_PATH}")
ext="\${p##*.}"
if command -v fpgautil >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo -n fpgautil -b "\${p}" || fpgautil -b "\${p}"
  else
    fpgautil -b "\${p}"
  fi
elif command -v xbutil >/dev/null 2>&1 && [ "\${ext}" = "xclbin" ]; then
  xbutil program --device 0 --base --path "\${p}" || xbutil program -d 0 -u "\${p}"
else
  echo "No supported bitstream loader found: fpgautil or xbutil" >&2
  exit 42
fi
BITSTREAM_LOAD
)
  if ! run_ssh_logged "bitstream_load" "${bitstream_load_cmd}"; then
    block_with \
      "BLOCKED_BITSTREAM" \
      "Bitstream load failed on KV260" \
      "Install/use fpgautil for .bit/.bin flows or xbutil for .xclbin flows, or set PCCX_SKIP_BITSTREAM_LOAD=1 if the correct image is already loaded." \
      "Check logs: ${EVIDENCE_REL}/bitstream_load_stderr.log"
  fi
  BITSTREAM_LOADED="yes"
fi

REMOTE_SCRIPT_LOCAL="${EVIDENCE_DIR}/run_remote_smoke.sh"
REMOTE_SCRIPT_PATH="${REMOTE_RUN_DIR}/run_remote_smoke.sh"
write_remote_smoke_script "${REMOTE_SCRIPT_LOCAL}"

printf 'staging_runtime_runner=1\n'
if ! run_scp_to_logged "remote_runner_stage" "${REMOTE_SCRIPT_LOCAL}" "${REMOTE_SCRIPT_PATH}"; then
  block_with \
    "BLOCKED_RUNTIME" \
    "Could not copy remote smoke runner to KV260" \
    "Check logs: ${EVIDENCE_REL}/remote_runner_stage_stderr.log"
fi

printf 'running_remote_smoke=1\n'
remote_env_cmd=$(
  printf 'MODEL_DIR=%s RUN_PROMPT=%s RUN_TOKENS=%s BOARD_RUNTIME_DIR=%s REMOTE_RUN_DIR=%s BITSTREAM_PATH=%s ' \
    "$(remote_quote "${REMOTE_MODEL_DIR}")" \
    "$(remote_quote "${PCCX_RUN_PROMPT}")" \
    "$(remote_quote "${PCCX_RUN_TOKENS}")" \
    "$(remote_quote "${REMOTE_BASE}")" \
    "$(remote_quote "${REMOTE_RUN_DIR}")" \
    "$(remote_quote "${REMOTE_BITSTREAM_PATH}")"
)
if [[ -n "${PCCX_REMOTE_RUN_CMD:-}" ]]; then
  remote_env_cmd+="PCCX_REMOTE_RUN_CMD=$(remote_quote "${PCCX_REMOTE_RUN_CMD}") "
fi
if [[ -n "${PCCX_TRUST_REMOTE_CODE:-}" ]]; then
  remote_env_cmd+="PCCX_TRUST_REMOTE_CODE=$(remote_quote "${PCCX_TRUST_REMOTE_CODE}") "
fi

set +e
run_ssh_logged "remote_smoke" "${remote_env_cmd} bash $(remote_quote "${REMOTE_SCRIPT_PATH}")"
remote_smoke_rc=$?
set -e

run_scp_from_logged "remote_result_fetch" "${REMOTE_RUN_DIR}/result.env" "${EVIDENCE_DIR}/remote_result.env" || true
run_scp_from_logged "remote_generated_fetch" "${REMOTE_RUN_DIR}/generated_output.txt" "${EVIDENCE_DIR}/generated_output.txt" || true
run_scp_from_logged "remote_runtime_stdout_fetch" "${REMOTE_RUN_DIR}/runtime_stdout.log" "${EVIDENCE_DIR}/runtime_stdout.log" || true
run_scp_from_logged "remote_runtime_stderr_fetch" "${REMOTE_RUN_DIR}/runtime_stderr.log" "${EVIDENCE_DIR}/runtime_stderr.log" || true

if [[ ! -s "${EVIDENCE_DIR}/remote_result.env" ]]; then
  block_with \
    "BLOCKED_RUNTIME" \
    "Remote smoke run did not produce a result file" \
    "Check logs: ${EVIDENCE_REL}/remote_smoke_stdout.log and ${EVIDENCE_REL}/remote_smoke_stderr.log."
fi

RESULT_STATUS="$(result_value RESULT_STATUS "${EVIDENCE_DIR}/remote_result.env")"
RUNTIME_PATH="$(result_value RUNTIME_PATH "${EVIDENCE_DIR}/remote_result.env")"
TOKEN_COUNT="$(result_value TOKEN_COUNT "${EVIDENCE_DIR}/remote_result.env")"
ELAPSED_SEC="$(result_value ELAPSED_SEC "${EVIDENCE_DIR}/remote_result.env")"
TOK_PER_SEC="$(result_value TOK_PER_SEC "${EVIDENCE_DIR}/remote_result.env")"
BLOCKER_REASON="$(result_value BLOCKER_REASON "${EVIDENCE_DIR}/remote_result.env")"

if [[ -z "${RESULT_STATUS}" ]]; then
  RESULT_STATUS="BLOCKED_RUNTIME"
  RUNTIME_PATH="blocked"
  BLOCKER_REASON="Remote result file did not contain RESULT_STATUS"
fi

if [[ "${remote_smoke_rc}" -ne 0 && "${RESULT_STATUS}" == PASS_* ]]; then
  RESULT_STATUS="BLOCKED_RUNTIME"
  RUNTIME_PATH="blocked"
  BLOCKER_REASON="Remote smoke command returned nonzero after reporting pass"
fi

case "${RESULT_STATUS}" in
  PASS_KV260_NPU|PASS_KV260_FALLBACK)
    finish 0
    ;;
  BLOCKED_RTL)
    finish 2
    ;;
  BLOCKED_*)
    if [[ -n "${BLOCKER_REASON}" ]]; then
      {
        printf 'status=%s\n' "${RESULT_STATUS}"
        printf 'reason=%s\n' "${BLOCKER_REASON}"
        printf 'instructions:\n'
        printf -- '- Inspect remote runtime stdout/stderr in this evidence directory.\n'
        printf -- '- If the failing command is an RTL/Vivado readiness issue, hand the log to the RTL/Vivado worker.\n'
      } >"${BLOCKER_FILE}"
    fi
    finish 2
    ;;
  *)
    finish 1
    ;;
esac
