#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-ablate_axmodel}"

TEXT="${TEXT:-Friends fell out often because life was changing so fast}"
VOICE="${VOICE:-af_heart}"
LANG_CODE="${LANG_CODE:-a}"
SPEED="${SPEED:-1.0}"
SCRIPT="${SCRIPT:-examples/infer_mixed_static_backend.py}"
OUTPUT_DIR="${OUTPUT_DIR:-examples/ablation_outputs}"
DYNAMIC_ONNX_PATH="${DYNAMIC_ONNX_PATH:-onnx_modules/duration_predictor.onnx}"
STATIC_ONNX_DIR="${STATIC_ONNX_DIR:-onnx_modules_static_frontend}"
AXMODEL_DIR="${AXMODEL_DIR:-kokoro-axmodel}"
DYNAMIC_PROVIDERS="${DYNAMIC_PROVIDERS:-CPUExecutionProvider}"
STATIC_ONNX_PROVIDERS="${STATIC_ONNX_PROVIDERS:-CPUExecutionProvider}"
DRY_RUN="${DRY_RUN:-0}"

MODULES=(encoder text_encoder f0n_shared f0n_head decoder vocoder)

mkdir -p "$OUTPUT_DIR"

run_case() {
    local label="$1"
    shift

    local output_path="$OUTPUT_DIR/${label}.wav"
    local cmd=(
        python "$SCRIPT"
        --text "$TEXT"
        --voice "$VOICE"
        --lang_code "$LANG_CODE"
        --speed "$SPEED"
        --dynamic_onnx_path "$DYNAMIC_ONNX_PATH"
        --static_onnx_dir "$STATIC_ONNX_DIR"
        --axmodel_dir "$AXMODEL_DIR"
        --dynamic_providers "$DYNAMIC_PROVIDERS"
        --static_onnx_providers "$STATIC_ONNX_PROVIDERS"
        --output "$output_path"
    )
    cmd+=("$@")

    printf '\n[%s]\n' "$label"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi

    "${cmd[@]}"
}

run_exhaustive() {
    local default_backend="$1"
    local switched_backend="$2"
    local max_mask=$((1 << ${#MODULES[@]}))

    for ((mask = 0; mask < max_mask; mask++)); do
        local label="${default_backend}"
        local args=(--default_backend "$default_backend")

        for ((idx = 0; idx < ${#MODULES[@]}; idx++)); do
            local module_name="${MODULES[$idx]}"
            if ((mask & (1 << idx))); then
                args+=("--${module_name}_backend" "$switched_backend")
                label+="_${module_name}-${switched_backend}"
            fi
        done

        run_case "$label" "${args[@]}"
    done
}

case "$MODE" in
    baseline)
        run_case all_axmodel --default_backend axmodel
        run_case all_onnx --default_backend onnx
        ;;
    ablate_axmodel)
        run_case all_axmodel --default_backend axmodel
        for module_name in "${MODULES[@]}"; do
            run_case "axmodel_${module_name}-onnx" --default_backend axmodel "--${module_name}_backend" onnx
        done
        ;;
    ablate_onnx)
        run_case all_onnx --default_backend onnx
        for module_name in "${MODULES[@]}"; do
            run_case "onnx_${module_name}-axmodel" --default_backend onnx "--${module_name}_backend" axmodel
        done
        ;;
    compare)
        run_case all_axmodel --default_backend axmodel
        run_case all_onnx --default_backend onnx
        for module_name in "${MODULES[@]}"; do
            run_case "axmodel_${module_name}-onnx" --default_backend axmodel "--${module_name}_backend" onnx
            run_case "onnx_${module_name}-axmodel" --default_backend onnx "--${module_name}_backend" axmodel
        done
        ;;
    exhaustive_from_axmodel)
        run_exhaustive axmodel onnx
        ;;
    exhaustive_from_onnx)
        run_exhaustive onnx axmodel
        ;;
    *)
        echo "Unsupported mode: $MODE" >&2
        echo "Supported modes: baseline, ablate_axmodel, ablate_onnx, compare, exhaustive_from_axmodel, exhaustive_from_onnx" >&2
        exit 1
        ;;
esac