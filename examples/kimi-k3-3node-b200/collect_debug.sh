#!/bin/bash
# One-shot hang diagnostics for the Kimi-K3 3-node bring-up.
# Run on EACH inference node while the run is stuck:
#   bash examples/kimi-k3-3node-b200/collect_debug.sh
# Results land in running_logs/debug_<host>_<ts>/ on shared scratch.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV="$ROOT_DIR/venv"
OUT="$ROOT_DIR/running_logs/debug_$(hostname -s)_$(date +%H%M%S)"
mkdir -p "$OUT"
echo "Collecting into $OUT ..."

{ date; hostname; uptime; } > "$OUT/00_host.txt" 2>&1

# --- GPU health: does nvidia-smi respond at all, and any XIDs? -------------
{ time timeout 20 nvidia-smi; } > "$OUT/10_nvidia_smi.txt" 2>&1
timeout 20 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks_throttle_reasons.active \
  --format=csv > "$OUT/11_gpu_util.txt" 2>&1
timeout 20 nvidia-smi --query-compute-apps=pid,used_memory --format=csv \
  > "$OUT/12_gpu_procs.txt" 2>&1
# XID / driver errors (may be restricted; try sudo -n silently, then journalctl)
{ dmesg -T 2>/dev/null || sudo -n dmesg -T 2>/dev/null || journalctl -k --no-pager -n 2000 2>/dev/null; } \
  | grep -iE "xid|nvrm|gpu.*(fall|error|timeout)|oom" | tail -50 > "$OUT/13_dmesg_xid.txt"
[[ -s "$OUT/13_dmesg_xid.txt" ]] || echo "(no kernel log access or no matches — if empty AND nvidia-smi above is instant, H1 unlikely)" >> "$OUT/13_dmesg_xid.txt"

# --- Is anything moving? NIC bytes + per-process CPU time, sampled twice ---
IF=$(ip -o -4 addr | awk '/10\.1\.33\./{print $2; exit}')
declare -A CPU0
mapfile -t PIDS < <(pgrep -u "$USER" -f 'VLLM::' | sort -n)
for p in "${PIDS[@]}"; do CPU0[$p]=$(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null || echo 0); done
RX0=$(cat "/sys/class/net/$IF/statistics/rx_bytes"); TX0=$(cat "/sys/class/net/$IF/statistics/tx_bytes")
sleep 15
RX1=$(cat "/sys/class/net/$IF/statistics/rx_bytes"); TX1=$(cat "/sys/class/net/$IF/statistics/tx_bytes")
{
  echo "iface=$IF window=15s"
  echo "rx: $(( (RX1-RX0)/15/1024/1024 )) MB/s   tx: $(( (TX1-TX0)/15/1024/1024 )) MB/s"
  echo; echo "pid cmd cpu_jiffies_delta_15s"
  for p in "${PIDS[@]}"; do
    c1=$(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null || echo 0)
    echo "$p $(tr -d '\0' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-40) $(( c1 - ${CPU0[$p]:-0} ))"
  done
} > "$OUT/20_activity.txt" 2>&1

# --- Worker env sanity: did the iface/EAGER fixes reach the processes? -----
for p in "${PIDS[@]}"; do
  echo "=== pid $p $(tr -d '\0' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-50)"
  tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null \
    | grep -E '^(NCCL|GLOO|TP_SOCKET|CUDA_MODULE|VLLM_(USE_DEEP|FLASHINFER|HAS))' | sort
done > "$OUT/30_worker_env.txt" 2>&1

# --- Stacks: python for all, NATIVE for workers (the decisive C frames) ----
PYSPY="$VENV/bin/py-spy"
for p in "${PIDS[@]}"; do
  name=$(tr -d '\0' < "/proc/$p/cmdline" 2>/dev/null | grep -oE 'Worker_TP[0-9]+|EngineCore|VllmEngine' | head -1)
  timeout 60 "$PYSPY" dump --pid "$p" > "$OUT/40_pyspy_${name:-p}_$p.txt" 2>&1
  if [[ "${name:-}" == Worker_TP* ]]; then
    timeout 120 "$PYSPY" dump --native --pid "$p" > "$OUT/41_native_${name}_$p.txt" 2>&1
  fi
done

echo "Done. Results in: $OUT"
ls -la "$OUT"
