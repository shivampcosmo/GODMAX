#!/usr/bin/env bash
# Periodically print node CPU/GPU/process telemetry into the Slurm log.

set -u

interval=${MONITOR_INTERVAL:-180}
gpu_sample_interval=${GPU_SAMPLE_INTERVAL:-10}
label=${MONITOR_LABEL:-node}
match_regex=${MONITOR_PROCESS_REGEX:-stage31_pz1_backlight_validation|paste-split|combine-maps|plot-healpix-maps|python|srun}

read_cpu_totals() {
  awk '/^cpu / {
    idle=$5+$6
    total=0
    for (i=2; i<=NF; i++) total += $i
    print total, idle
  }' /proc/stat
}

print_gpu_average() {
  local duration="$1"
  local sample_sleep="$2"
  local elapsed=0
  local nsamp=0
  local util_sum=0
  local mem_sum=0
  local mem_total=0
  local util_now mem_now total_now query

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[monitor:${label}] gpu unavailable: nvidia-smi not found"
    sleep "${duration}"
    return
  fi

  while [ "${elapsed}" -lt "${duration}" ]; do
    query=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    if [ -n "${query}" ]; then
      while IFS=',' read -r util_now mem_now total_now; do
        util_now=$(echo "${util_now}" | tr -d ' ')
        mem_now=$(echo "${mem_now}" | tr -d ' ')
        total_now=$(echo "${total_now}" | tr -d ' ')
        if [[ "${util_now}" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "${mem_now}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
          util_sum=$(awk -v a="${util_sum}" -v b="${util_now}" 'BEGIN {print a+b}')
          mem_sum=$(awk -v a="${mem_sum}" -v b="${mem_now}" 'BEGIN {print a+b}')
          mem_total="${total_now}"
          nsamp=$((nsamp + 1))
        fi
      done <<< "${query}"
    fi
    sleep "${sample_sleep}"
    elapsed=$((elapsed + sample_sleep))
  done

  if [ "${nsamp}" -gt 0 ]; then
    awk -v u="${util_sum}" -v m="${mem_sum}" -v n="${nsamp}" -v mt="${mem_total}" \
      'BEGIN {printf("[monitor:%s] gpu_avg_over_interval util=%.1f%% mem_used=%.0f MiB mem_total=%s MiB samples=%d\n", ENVIRON["MONITOR_LABEL"], u/n, m/n, mt, n)}'
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader 2>/dev/null | sed "s/^/[monitor:${label}] gpu_now /" || true
  else
    echo "[monitor:${label}] gpu unavailable: no nvidia-smi samples"
  fi
}

echo "[monitor:${label}] start host=$(hostname) interval=${interval}s gpu_sample_interval=${gpu_sample_interval}s pid=$$"

while true; do
  ts_start=$(date -Is)
  read -r cpu_total_0 cpu_idle_0 < <(read_cpu_totals)
  print_gpu_average "${interval}" "${gpu_sample_interval}"
  read -r cpu_total_1 cpu_idle_1 < <(read_cpu_totals)
  cpu_busy=$(awk -v t0="${cpu_total_0}" -v i0="${cpu_idle_0}" -v t1="${cpu_total_1}" -v i1="${cpu_idle_1}" \
    'BEGIN {dt=t1-t0; di=i1-i0; if (dt>0) printf("%.1f", 100.0*(dt-di)/dt); else printf("nan")}')
  mem_line=$(free -h | awk '/^Mem:/ {print "mem_used="$3" mem_total="$2" mem_available="$7}')
  load_line=$(awk '{print "loadavg_1m="$1" loadavg_5m="$2" loadavg_15m="$3}' /proc/loadavg)
  echo "[monitor:${label}] summary start=${ts_start} end=$(date -Is) cpu_avg_busy=${cpu_busy}% ${load_line} ${mem_line}"
  ps -u "${USER}" -o pid,ppid,stat,etime,pcpu,pmem,rss,cmd --sort=-pcpu \
    | grep -E "${match_regex}" \
    | grep -v grep \
    | head -20 \
    | sed "s/^/[monitor:${label}] proc /" || true
done
