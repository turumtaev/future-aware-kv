#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/local/many_to_one}"
DEVICE="${DEVICE:-cpu}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

"$PYTHON_BIN" -m pytest -q

for members in 2 4 8 16 32 64 128; do
    batch_size=$((512 / members))
    for architecture in 2a fakv; do
        for seed in 20 21 22; do
            output="$RESULT_ROOT/m${members}/${architecture}/seed_${seed}"
            if [[ -f "$output/summary.json" ]]; then
                echo "SKIP m=$members model=$architecture seed=$seed"
                continue
            fi
            mkdir -p "$output"
            echo "RUN  m=$members model=$architecture seed=$seed batch=$batch_size"
            "$PYTHON_BIN" experiments/many_to_one.py \
                --architecture "$architecture" \
                --members "$members" \
                --seed "$seed" \
                --batch-size "$batch_size" \
                --steps 3000 \
                --eval-every 100 \
                --eval-batches 16 \
                --test-batches 64 \
                --device "$DEVICE" \
                --output "$output" \
                2>&1 | tee "$output/stdout.log"
        done
    done
done

"$PYTHON_BIN" experiments/summarize_many_to_one.py \
    --input "$RESULT_ROOT" \
    --csv results/many_to_one.csv \
    --report results/many_to_one.md

echo "Complete. See results/many_to_one.md and results/many_to_one.csv."
