#!/usr/bin/env bash
# Pareto sweep: runs all w_bike x w_ped combinations one after another.
# Each python call blocks in the foreground, so runs are naturally
# back-to-back -- no fixed sleep needed. `timeout` below is only a safety
# net in case a run ever hangs (kills it after 1h30m and moves on); it does
# NOT delay a run that finishes normally.

set -uo pipefail   # NOTE: no -e, so one crashed/timed-out run doesn't abort the whole sweep

BIKE_WEIGHTS=(0.4 0.8 1.2 1.6)
PED_WEIGHTS=(60 120 200 300)

MAX_RUN_SECONDS=$((90 * 60))   # 1h30m safety cap per run (1:15 + 15min buffer)

mkdir -p logs

sweep_start=$(date +%s)
run_num=0
total_runs=$(( ${#BIKE_WEIGHTS[@]} * ${#PED_WEIGHTS[@]} ))

for bike in "${BIKE_WEIGHTS[@]}"; do
  for ped in "${PED_WEIGHTS[@]}"; do
    run_num=$((run_num + 1))
    log_file="logs/idqn_mm2_bike${bike}_ped${ped}.log"

    echo "=== [$run_num/$total_runs] Starting w_bike=${bike} w_ped=${ped} at $(date) ==="

    timeout "${MAX_RUN_SECONDS}" python main.py \
      --agent IDQN_MM2 --map kbh_j1_442m --eps 100 --tr 0 --strategy 1 \
      --w_bike "${bike}" --w_ped "${ped}" --max_green_hold 12 \
      > "${log_file}" 2>&1
    exit_code=$?

    if [ "${exit_code}" -eq 124 ]; then
      echo "=== [$run_num/$total_runs] TIMED OUT (w_bike=${bike} w_ped=${ped}) after ${MAX_RUN_SECONDS}s -- see ${log_file} ==="
    elif [ "${exit_code}" -ne 0 ]; then
      echo "=== [$run_num/$total_runs] FAILED (exit ${exit_code}) w_bike=${bike} w_ped=${ped} -- see ${log_file} ==="
    else
      echo "=== [$run_num/$total_runs] Finished OK w_bike=${bike} w_ped=${ped} at $(date) ==="
    fi
  done
done

sweep_end=$(date +%s)
echo "All ${total_runs} runs complete. Total sweep time: $(( (sweep_end - sweep_start) / 60 )) minutes."