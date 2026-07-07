#!/bin/bash

set -euo pipefail

mkdir -p qsub_logs

EMAIL="shahyadk@bu.edu"

NN_CONFIG="benchmarking/configs_nn.txt"
RVFL_MEDIUM_CONFIG="benchmarking/configs_rvfl_medium.txt"
RVFL_HEAVY_CONFIG="benchmarking/configs_rvfl_heavy.txt"

NN_N=$(wc -l < "$NN_CONFIG" | tr -d ' ')
RVFL_MEDIUM_N=$(wc -l < "$RVFL_MEDIUM_CONFIG" | tr -d ' ')
RVFL_HEAVY_N=$(wc -l < "$RVFL_HEAVY_CONFIG" | tr -d ' ')

MEDIUM_GPU_QUEUES="l40s,h200,a100,a40,v100,p100"
HEAVY_GPU_QUEUES="l40s,h200,a100,a40"

echo "Submitting NN array with ${NN_N} jobs"
NN_SUBMIT_OUTPUT=$(
qsub \
    -N bench_nn \
    -t 1-${NN_N} \
    -tc 4 \
    -q "$MEDIUM_GPU_QUEUES" \
    -l gpus=1 \
    -l h_rt=04:00:00 \
    -pe omp 1 \
    -l mem_per_core=8G \
    -v CONFIG_FILE="$NN_CONFIG" \
    benchmarking/run_config_array.qsub
)
echo "$NN_SUBMIT_OUTPUT"
NN_JOB_ID=$(echo "$NN_SUBMIT_OUTPUT" | awk '{print $3}' | cut -d. -f1)

echo "Submitting medium RVFL array with ${RVFL_MEDIUM_N} jobs"
RVFL_MEDIUM_SUBMIT_OUTPUT=$(
qsub \
    -N bench_rvfl_mid \
    -t 1-${RVFL_MEDIUM_N} \
    -tc 2 \
    -q "$MEDIUM_GPU_QUEUES" \
    -l gpus=1 \
    -l h_rt=08:00:00 \
    -pe omp 1 \
    -l mem_per_core=16G \
    -v CONFIG_FILE="$RVFL_MEDIUM_CONFIG" \
    benchmarking/run_config_array.qsub
)
echo "$RVFL_MEDIUM_SUBMIT_OUTPUT"
RVFL_MEDIUM_JOB_ID=$(echo "$RVFL_MEDIUM_SUBMIT_OUTPUT" | awk '{print $3}' | cut -d. -f1)

echo "Submitting heavy RVFL array with ${RVFL_HEAVY_N} jobs"
RVFL_HEAVY_SUBMIT_OUTPUT=$(
qsub \
    -N bench_rvfl_big \
    -t 1-${RVFL_HEAVY_N} \
    -tc 1 \
    -q "$HEAVY_GPU_QUEUES" \
    -l gpus=1 \
    -l h_rt=24:00:00 \
    -pe omp 1 \
    -l mem_per_core=16G \
    -v CONFIG_FILE="$RVFL_HEAVY_CONFIG" \
    benchmarking/run_config_array.qsub
)
echo "$RVFL_HEAVY_SUBMIT_OUTPUT"
RVFL_HEAVY_JOB_ID=$(echo "$RVFL_HEAVY_SUBMIT_OUTPUT" | awk '{print $3}' | cut -d. -f1)

echo "Submitted job IDs:"
echo "  NN:          $NN_JOB_ID"
echo "  RVFL medium: $RVFL_MEDIUM_JOB_ID"
echo "  RVFL heavy:  $RVFL_HEAVY_JOB_ID"

echo "Benchmark arrays submitted at $(date)

Job IDs:
NN:          $NN_JOB_ID
RVFL medium: $RVFL_MEDIUM_JOB_ID
RVFL heavy:  $RVFL_HEAVY_JOB_ID

Counts:
NN jobs:          $NN_N
RVFL medium jobs: $RVFL_MEDIUM_N
RVFL heavy jobs:  $RVFL_HEAVY_N
" | mail -s "RVFL benchmark sweep submitted" "$EMAIL" || true

echo "Submitting one final notification job held on all arrays"
DONE_SUBMIT_OUTPUT=$(
qsub \
    -N bench_done_email \
    -hold_jid "$NN_JOB_ID,$RVFL_MEDIUM_JOB_ID,$RVFL_HEAVY_JOB_ID" \
    benchmarking/notify_done.qsub
)
echo "$DONE_SUBMIT_OUTPUT"

echo "Done. You should get one email now and one when all arrays finish."