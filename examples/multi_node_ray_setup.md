# Multi-node Ray + vLLM setup

End-to-end recipe for standing up a two-node Ray cluster (for
tensor-parallel + pipeline-parallel vLLM runs) on any cluster.

## What multi-node adds on top of a single-node run

A single-node vLLM launch is just `lm_eval --model vllm --model_args
pretrained=<path>,tensor_parallel_size=<N>,...`. Everything else is
autoconfigured. Multi-node needs six extra pieces, and this doc walks
you through each:

| Extra piece | What it looks like | Why it's needed |
| --- | --- | --- |
| **Network env vars** — `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` | Single NIC that routes cross-node (same name + same `/24` on every node). Optional multi-NIC fast fabric for NCCL only if one truly exists. | Cross-node data plane. Without this, NCCL/Gloo auto-pick a NIC with no route to peer → `Network is unreachable`. |
| **Ray cluster start** — `ray start --head` on one node, `ray start --address=...` on each other | Two commands, one per node | vLLM's default `mp` executor can't cross nodes. Ray schedules workers across the two hosts. |
| **`RAY_ADDRESS=auto`** in the launch env | Single export before `lm_eval` | vLLM's Ray-executor path locates the running cluster via this env var (fallback to `/tmp/ray/session_latest`). |
| **`distributed_executor_backend=ray`** in `--model_args` | One extra key=value in the arg string | Tells vLLM to use Ray instead of its default multi-process executor. |
| **`unset VLLM_PORT`** | Single unset before `lm_eval` | vLLM's default port-scan starts at ~100 (needs root → EACCES). Unsetting `VLLM_PORT` makes vLLM use `bind(("", 0))` — OS picks a free ephemeral port. Never collides, never EACCES. |
| **Env parity across nodes** | Same `torch.__version__` + `torch.cuda.nccl.version()` on every node before `ray start` | Different NCCL versions can't handshake — workers silently hang in `ncclCommInitRank` with no error message. See Step 2c for the preflight. |

If you strip everything else from this doc, that's the delta. The rest
is discovery (finding the right NIC names on your specific cluster) and
troubleshooting for when one of these pieces is misconfigured.


## Assumes only that

- vLLM, Ray, and any deps are importable by the same Python interpreter
  on every node (whichever way you install them — venv, conda, system
  pip, editable checkout — this recipe doesn't care).
- Whichever interpreter you use, it's the **same version and same
  package set** on every node. Ray refuses to attach a worker with a
  different Python version than the head, and NCCL fails if the vLLM
  native extensions don't match.

The example uses one head node + one worker node (2 nodes × 8 GPUs = 16
GPUs total for `TP=8, PP=2`). Extending to N nodes is mechanical.


## 1. Identify the cross-node NIC

Multi-node NCCL/Gloo need one specific NIC that routes between the two
nodes. On each node, list the interfaces:

```bash
ip -o -4 addr | awk '{print $2, $4}' | grep -v -E '^lo|::' | sort -u
```

Find a NIC that appears on **both** nodes with IPs on the **same `/24`
subnet**. That's the cross-node NIC. Everything else — link speed, NIC
count, InfiniBand vs Ethernet — is a distraction.

**Concrete example** (yours will differ):

```
                       HEAD                     WORKER
enp81s0f1np1    10.1.36.104/24          enp81s0f1np1    10.1.36.97/24    <-- SAME /24 on both
ens21np0        172.16.0.104/24         ens21np0        10.255.41.97/32  <-- DIFFERENT subnets — skip
```

Here `enp81s0f1np1` is the cross-node NIC (both on `10.1.36.0/24`).
`ens21np0` looks fast on the head but the worker has it on a `/32` in
a different subnet, so it's not a shared fabric — skip it.

Confirm the NIC actually routes with ping:

```bash
# On head, ping worker over this NIC
ping -c 3 -I enp81s0f1np1 10.1.36.97
```

Success = 3/3 received. That NIC goes into `NET_IF_CTRL` in Step 2a.

That's Step 1. Anything else about network discovery (fast fabric
hunts, InfiniBand tuning, interface-name mismatches) is in the
**N.B.** section at the bottom of the doc.


## 2. Prepare the environment on both nodes

Every env var vLLM's workers need must be exported **before**
`ray start` on **each** node, since raylet inherits the environment and
propagates it to spawned workers.

### 2a. The env-vars block

Save this as `/path/to/shared/ray_env.sh` so both nodes source the same
file:

```bash
# ---- interpreter + PATH ----
# PY = absolute path to the Python interpreter that can import vllm.
# Whatever env manager you use, point PY at THAT env's `python` binary.
# Concrete examples:
#   conda:   PY=${HOME}/miniconda3/envs/<env-name>/bin/python
#   venv:    PY=${HOME}/venvs/<env-name>/bin/python
#   uv:      PY=${HOME}/.venv/bin/python
#   system:  PY=$(command -v python3)
export PY="<path-to-python-that-imports-vllm>"
export PATH="$(dirname "${PY}"):${PATH}"

# ---- HF cache (any shared or scratch path with room) ----
export HF_HOME=/path/to/scratch/${USER}/.cache/
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# ---- network interfaces (fill in from Step 1's discovery) ----
# NET_IF_CTRL: single control NIC that routes cross-node (Ray, TCPStore,
#              Gloo, NCCL). This is REQUIRED — every cluster has one.
# NET_IF_FAST: optional comma-separated list of extra NICs to use ONLY
#              for NCCL, when the cluster actually has a shared fabric
#              (same NIC name, same /24 on both nodes). Leave empty and
#              NCCL uses only NET_IF_CTRL — always safe.
#
# Concrete example — the B200 cluster from Step 1's "concrete example":
# the ens21..ens28 NICs looked fast but were per-node private (each on
# its own /32 subnet on the worker), so they're NOT usable cross-node.
# Only enp81s0f1np1 shares a /24 across the two nodes. So:
#   NET_IF_CTRL="enp81s0f1np1"
#   NET_IF_FAST=""            # no shared fabric — use control NIC for NCCL too
NET_IF_CTRL="<cross-node-nic-from-step-1>"
NET_IF_FAST=""                # or "<fast-iface-1>,<fast-iface-2>,..." if N.B.2 found a real fast fabric

# NCCL uses the fast fabric IF present, else falls back to the control NIC.
export NCCL_SOCKET_IFNAME="${NET_IF_FAST:-${NET_IF_CTRL}}"
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1          # if the cluster has no InfiniBand; unset otherwise
export NCCL_IBEXT_DISABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_P2P_DISABLE=0         # keep intra-node NVLink

# Gloo and tensorpipe always use the control NIC (small CPU-side traffic,
# and Gloo doesn't parse comma lists).
export GLOO_SOCKET_IFNAME="${NET_IF_CTRL}"
export TP_SOCKET_IFNAME="${NET_IF_CTRL}"

# ---- vLLM knobs (set as needed) ----
# vLLM's TCPStore-port picker defaults to a low base (~100 for non-DP
# runs) which needs root. Leaving VLLM_PORT UNSET makes vLLM use
# bind(("", 0)) — the OS picks a free ephemeral port every time. No
# collisions across runs, no EACCES on privileged ports.
unset VLLM_PORT

export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export FLASHINFER_CACHE_DIR=/path/to/scratch/${USER}/flashinfer_cache
mkdir -p "${FLASHINFER_CACHE_DIR}"
```

### 2b. Source it on both nodes

```bash
# On BOTH the head and the worker:
source /path/to/shared/ray_env.sh

# Sanity — same values on both?
echo "PY=${PY}"
echo "NCCL=${NCCL_SOCKET_IFNAME}"
echo "GLOO=${GLOO_SOCKET_IFNAME}"
${PY} --version
```

If any env var differs across the two nodes, Ray workers will inherit
inconsistent settings and NCCL init will hang or error mysteriously.
Diagnose by running `env | sort` on both and diffing.


### 2c. Preflight: verify env consistency across nodes

This is the single most common cause of "everything looks fine but NCCL
init hangs forever" — the head and worker each have their own Python /
conda env, and one has been updated while the other hasn't. Different
`torch` versions → different NCCL versions → silent handshake failure.

**On BOTH nodes**, before starting Ray:

```bash
python -c "
import sys, torch
print('exe   :', sys.executable)
print('torch :', torch.__version__)
print('nccl  :', torch.cuda.nccl.version())
print('cuda  :', torch.version.cuda)
"
```

All four lines must match across nodes. If any differ, either the envs
have drifted, or **your shell's `python` resolves to a different env
than your shell's `ray` / `pip`.** Check for that specifically:

```bash
which python                            # e.g. /home/USER/miniconda3/envs/foo/bin/python
which ray                               # MUST be the same env
which pip
head -1 $(which ray)                    # the ray CLI's shebang python
```

If `which ray` points to a different env, `ray start` will run the
raylet under THAT env's Python, not the one your interactive shell
uses — and its `torch`/`NCCL` versions may not match the head's.

Common causes:

- Multiple envs in PATH (e.g. `/home/USER/.../bin` and `/some/shared/.../bin`),
  and ONE of the two happens to be in front. Force one to win:
  ```bash
  export PATH=$HOME/miniconda3/envs/model_stats/bin:$PATH
  hash -r    # flush shell's cached command lookups
  ```
- A ghost raylet running an old (or deleted) Python binary. Check with:
  ```bash
  for pid in $(pgrep -f 'raylet|ray::'); do
      readlink /proc/${pid}/exe
  done | sort -u
  ```
  Any path ending in `(deleted)` = kill and restart ray fresh in the
  current shell.

**After confirming `python`, `ray`, and `torch` all point at the same
env with the same versions on both nodes, THEN start Ray.** Skipping
this preflight is what causes the multi-hour NCCL-init-hang debugging
sessions.


## 3. Start the Ray cluster

### 3a. Get the head node's IP

```bash
# On the HEAD node:
HEAD_IP=$(ip -o -4 addr show "${NET_IF_CTRL}" | awk '{print $4}' | cut -d/ -f1)
echo "HEAD_IP=${HEAD_IP}"          # e.g. 10.1.36.104 on the example cluster
```

Save this value — you'll need it on the worker.

### 3b. Clean any stale Ray state on both nodes

```bash
# On BOTH nodes — scope pkill to your user so you don't nuke another
# user's Ray cluster if you share the node.
ray stop --force 2>/dev/null || true
pkill -9 -u "${USER}" -f 'ray::|raylet|gcs_server|redis' 2>/dev/null || true
rm -rf /tmp/ray/*
```

Ray stores per-session state in `/tmp/ray`. A prior crashed run leaves
stale actor references that confuse a fresh cluster.

If `ray status` later prints `WARNING ... Found multiple active Ray
instances: {'A:6379', 'B:6380'}`, that's another user's cluster on the
same host — cosmetic warning only, safe to ignore. To silence, set
`RAY_ADDRESS` explicitly (`export RAY_ADDRESS=<head-ip>:6379`).

### 3c. Start the head daemon (on the head node only)

```bash
# On the HEAD:
ray start --head \
    --port=6379 \
    --node-ip-address="${HEAD_IP}" \
    --num-gpus=8
```

Successful output ends with a line like:

```
To connect to this Ray cluster: ...
  ray start --address='<HEAD_IP>:6379'
```

Copy that `--address=…` — you'll paste it into the worker command.

### 3d. Join the worker (on the worker node)

```bash
# On the WORKER — replace HEAD_IP with the value from Step 3a
HEAD_IP=<value copied from head, from Step 3a>    # e.g. 10.1.36.104
MY_IP=$(ip -o -4 addr show "${NET_IF_CTRL}" | awk '{print $4}' | cut -d/ -f1)
echo "MY_IP=${MY_IP}"

ray start \
    --address="${HEAD_IP}:6379" \
    --node-ip-address="${MY_IP}" \
    --num-gpus=8
```

### 3e. Verify from either node

```bash
ray status
```

Expected:

```
Active:
 1 node_<hash-1>
 1 node_<hash-2>

Resources:
 0.0/224.0 CPU
 0.0/16.0 GPU
 0B/5.4TiB memory
```

If only one node appears, the worker's `ray start` failed to reach the
head. Check the connection over the control NIC:

```bash
# From the worker
telnet ${HEAD_IP} 6379    # should connect if head is reachable
```


## 4. Launch the vLLM workload

vLLM's driver (the `lm_eval` process) only runs on **one** node — the
one designated as head. Ray schedules worker actors across both nodes
automatically.

### 4a. Launch (on the head node only)

```bash
# Still on the HEAD, in the same shell that has ray_env.sh sourced:
export RAY_ADDRESS=auto            # attach to the running cluster

bash /path/to/launch_lm_harness.sh 2>&1 | tee launch.log
```

`RAY_ADDRESS=auto` reads `/tmp/ray/session_latest` to find the running
cluster. Alternative: `RAY_ADDRESS="${HEAD_IP}:6379"` for explicit control.

The launch script's `--model_args` must include
`distributed_executor_backend=ray` and set
`tensor_parallel_size × pipeline_parallel_size = total GPUs`.

### 4b. Example: Kimi-K3 with TP=8, PP=2 on 2 nodes × 8 GPUs

A complete launch script for a 16-GPU run (TP fills each node's 8 GPUs,
PP crosses the two nodes). Run this on the head node after Step 3 has
brought the Ray cluster up:

```bash
#!/usr/bin/env bash
# launch_lm_harness_Kimi_K3.sh
set -euo pipefail

# Inherit PY, NCCL_*, GLOO_*, HF_HOME, etc. from Step 2's env file.
source /path/to/shared/ray_env.sh

MODEL_NAME="/path/to/checkpoints/Kimi-K3"

TP_SIZE=8       # tensor parallel — fits in one node's 8 GPUs
PP_SIZE=2       # pipeline parallel — crosses the two nodes

# Ray attaches to the cluster started in Step 3.
export RAY_ADDRESS=auto

# vLLM's `--model_args` is a comma-separated list of key=value pairs.
# distributed_executor_backend=ray is REQUIRED for cross-node execution.
MODEL_ARGS="pretrained=${MODEL_NAME}"
MODEL_ARGS+=",tensor_parallel_size=${TP_SIZE}"
MODEL_ARGS+=",pipeline_parallel_size=${PP_SIZE}"
MODEL_ARGS+=",distributed_executor_backend=ray"
MODEL_ARGS+=",enforce_eager=True"
MODEL_ARGS+=",trust_remote_code=True"
MODEL_ARGS+=",dtype=auto"
MODEL_ARGS+=",gpu_memory_utilization=0.9"

lm_eval --model vllm \
    --model_args "${MODEL_ARGS}" \
    --tasks gsm8k \
    --limit 512
```

Notes on the choices:

- **TP × PP = 8 × 2 = 16** matches the total-GPU count on the cluster.
- **Only PP crosses node boundaries.** TP stays within a single node's
  NVLink domain — cheaper than tensor-splitting across a network.
- **`distributed_executor_backend=ray`** is what makes vLLM use the
  running Ray cluster instead of trying to spawn its own multiproc
  workers (which can't cross nodes).
- **`enforce_eager=True`** disables torch.compile for the run. Skip it
  for peak throughput once you're confident everything else works;
  keep it for debugging or when a plugin (e.g. custom activation hook)
  fights with the compiled graph.
- **`gpu_memory_utilization=0.9`** reserves ~10% headroom for KV
  cache growth + activation buffers. Lower it (0.7-0.8) if the model
  triggers OOM at load time.

### 4c. Signs of a healthy launch

Watch the log for:

1. `Ray Cluster: X CPUs, Y GPUs across 2 nodes` — Ray connection works.
2. `Resolved architecture: <ModelClassName>` — vLLM found the model class.
3. `Initializing an EngineCore ...` followed by worker spawn messages
   with `RayWorkerProc.initialize_worker()`.
4. `Loading weights...` progress bar.
5. Task iteration starts (e.g. gsm8k examples).

### 4d. Signs of trouble

| Symptom | Cause | Fix |
| --- | --- | --- |
| `connect: Network is unreachable` in Gloo trace | Gloo pointed at wrong NIC | Check `GLOO_SOCKET_IFNAME` matches a NIC with a route to peer |
| `no matching interface` NCCL error | NCCL NIC name doesn't exist on both nodes | Verify same-name-on-both-nodes per Step 1 (see N.B.4 for `diff` recipe) |
| Ray shows 1 node instead of 2 | Worker never joined | Check `telnet HEAD_IP 6379`, restart worker's `ray start` |
| `ActorHandleNotFoundError` on shutdown | Stale Ray state from previous crash | Full `ray stop --force`, wipe `/tmp/ray`, restart cluster |
| Workers OOM at model load | Weights don't fit | Lower `gpu_memory_utilization` or increase PP |
| Workers all stuck in `ncclCommInitRank` with 0% GPU util, no NCCL errors, no timeout | Torch/NCCL version mismatch across nodes (silent protocol mismatch) | Run Step 2c preflight; sync envs so `torch.__version__` and `torch.cuda.nccl.version()` are identical on all nodes |
| `Port N is already in use, trying port N+1` loop climbing from ~100 | vLLM defaults its TCPStore start-port to `master_port + 100`; ports < 1024 need root | `unset VLLM_PORT` so vLLM uses `bind(("", 0))` — OS picks a free ephemeral port |
| `address already in use` on a specific `VLLM_PORT` you set | Fixed port collides with prior run / other user | `unset VLLM_PORT` (OS-picked) OR randomize: `export VLLM_PORT=$(( 40000 + RANDOM % 10000 ))` |
| `FileNotFoundError: .../nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so` (or similar package `.so` missing on ONE node) | Version drift between packages in the same wheel family (e.g. `nvidia-cutlass-dsl-libs-cu13==4.5.2` while base is `4.6.0`) | On the node missing the file: `pip install --force-reinstall --no-deps "<pkg>==<version>"` matching the OTHER packages in the family. Then confirm the `.so` is present. |
| `undefined symbol: _ZN3c104impl3cow23materialize_cow_storage...` (or similar torch C++ ABI symbol) at `import vllm._C` or `deep_gemm/_C.*.so` | A precompiled `.so` was built against a different torch version | Rebuild the extension against current torch (`pip install --no-build-isolation --no-deps -e ~/vllm/` after `find vllm -name '*.abi3.so' -delete`). If it's a warning at `import_utils`, may be non-fatal — check if the run continues. |
| Worker's raylet uses a DIFFERENT Python than the head's raylet (per Step 2c preflight) | Multiple conda envs on the same machine; `PATH` picks one for `python`, another for `ray` | `which python && which ray` — if they differ, prepend the desired env's `bin` to `PATH`, `hash -r`, then `pip install ray` INTO that env if missing. Restart raylet in that shell. |


## 5. Teardown

When done, stop Ray cleanly on both nodes so the next session doesn't
inherit stale state:

```bash
# On BOTH nodes:
ray stop --force
```

Optional deeper cleanup after a crashed run (scope to your user):

```bash
rm -rf /tmp/ray/session_*
pkill -9 -u "${USER}" -f 'ray::|raylet|gcs_server' 2>/dev/null || true
```


## 6. Common variations

### N > 2 nodes

Same recipe, just run the `ray start --address=…` command on every
additional worker. `TP × PP = total GPUs / num_nodes` divides across
however many nodes are in the cluster. For N = 4 nodes × 8 GPUs, run
`TP=8, PP=4` or `TP=4, PP=8` (only PP crosses node boundaries; TP is
intra-node).

### Different clusters

The whole procedure works verbatim — only Step 1 (network topology) and
Step 2a (env vars) change. Rerun Step 1's discovery commands on the new
cluster's head node and update `NET_IF_CTRL` / `NET_IF_FAST` in the env
file. Everything else (Ray commands, launch script) is cluster-agnostic.

### If the cluster has InfiniBand

Replace the NCCL block in Step 2a:

```bash
NET_IF_FAST="ib0,ib1,ib2,ib3"    # or whatever ib* interfaces show up
export NCCL_SOCKET_IFNAME="${NET_IF_FAST}"
unset NCCL_IB_DISABLE            # let NCCL use IB natively (much faster)
unset NCCL_IBEXT_DISABLE
unset NCCL_NET_PLUGIN
unset NCCL_NET
```

IB gives 200-400 Gbps per link. If NCCL's IB probe segfaults during
init (a known bug in some NCCL 2.28.x releases), fall back to TCP-Socket
over the IB interface by keeping the disable flags.


## N.B. — extended network discovery

Everything below is optional detail for cases where the one-NIC recipe
in Step 1 isn't enough: hunting for a multi-NIC fast fabric, resolving
interface-name mismatches between nodes, per-NIC ping matrices for
clusters with unusual routing.

### N.B.1 — Identifying the control NIC more carefully

If the "same NIC name on same /24 across both nodes" check in Step 1
turns up multiple candidates, or none obviously stands out, these
commands help disambiguate:

```bash
# Which interface owns the default route?
ip route show default | awk '{print $5}'

# What does hostname's own IP look up to?
hostname -i

# If Ray is already running, ask it:
python -c "import ray; ray.init(address='auto'); print(ray.nodes()[0]['NodeManagerAddress'])"
```

All three should agree on one interface. That's your control NIC.
Named `enp*`, `eth*`, `bond0`, or similar; single interface (not one
of many); on a routable subnet like `10.x.x.x` or `192.168.x.x`.

### N.B.2 — Hunting for a multi-NIC fast fabric

Some clusters have real cross-node fast fabrics (multiple 100-400 Gbps
NICs sharing a subnet across nodes) that dramatically speed up NCCL
data traffic. To identify:

**Link speed** — high-Gbps NICs are fast-fabric candidates:

```bash
for iface in $(ls /sys/class/net/ | grep -v lo); do
    speed=$(cat /sys/class/net/$iface/speed 2>/dev/null || echo 0)
    echo "$iface  ${speed} Mbps"
done | sort -k2 -n
```

100000+ Mbps = candidate. 1000-25000 Mbps = control-plane class.

**Then verify cross-node subnet sharing** — same NIC name + same `/24`
on both nodes + ping success. All three must hold. If any fails, the
NIC is per-node private wiring (typical on many clusters — including
the B200 example above), not a shared fabric. Fall back to using the
control NIC for NCCL too.

If you DO find a shared fast fabric, populate `NET_IF_FAST` in Step 2a
with a comma-separated list of those interface names. NCCL picks
them up as its data plane automatically.

### N.B.3 — Per-NIC reachability matrix

For clusters with unusual routing, check each candidate NIC explicitly:

```bash
# Get the worker's IP on a candidate fast NIC (SSH in and run):
ip -o -4 addr show <fast-iface> | awk '{print $4}' | cut -d/ -f1

# Then from the head:
ping -c 3 -I <fast-iface> <worker-ip-on-that-iface>
```

Repeat for every candidate. Drop any that fail from `NET_IF_FAST`.

### N.B.4 — Interface names mismatched across nodes

NCCL requires the same interface names on all nodes:

```bash
# On head node
ip -o addr | awk '{print $2}' | sort -u > /tmp/nics_head.txt

# On worker node
ip -o addr | awk '{print $2}' | sort -u > /tmp/nics_worker.txt

# Compare
diff /tmp/nics_head.txt /tmp/nics_worker.txt
```

Any NIC that appears on only one side is unusable — NCCL can't handshake
across differently-named interfaces. If the fast NICs have different
names on the two nodes (e.g. `<iface>` on head vs a slightly renamed
variant on worker), use only their intersection, or ask cluster admin
to standardize naming.


## Quick reference card

Print and stick to your monitor:

```
== preflight: on BOTH nodes (Step 2c) ==
which python && which ray                           # must be same env on both
python -c "import torch; print(torch.__version__, torch.cuda.nccl.version())"
# outputs must match across nodes

== on both nodes ==
source /path/to/shared/ray_env.sh
ray stop --force; pkill -9 -u "${USER}" -f 'raylet|ray::'; rm -rf /tmp/ray/*

== on head ==
HEAD_IP=$(ip -o -4 addr show "${NET_IF_CTRL}" | awk '{print $4}' | cut -d/ -f1)
ray start --head --port=6379 --node-ip-address="${HEAD_IP}" --num-gpus=8

== on worker ==
HEAD_IP=<value from head>
MY_IP=$(ip -o -4 addr show "${NET_IF_CTRL}" | awk '{print $4}' | cut -d/ -f1)
ray start --address="${HEAD_IP}:6379" --node-ip-address="${MY_IP}" --num-gpus=8

== on head (verify + launch) ==
ray status                                          # expect 2 nodes / 16 GPUs
python -c "
import ray; ray.init(address='auto')
import ray as r
@r.remote(num_gpus=1)
def check(): import sys, torch; return (sys.executable, torch.__version__, str(torch.cuda.nccl.version()))
for n in r.nodes():
    ip = n['NodeManagerAddress']
    print(ip, r.get(check.options(resources={f'node:{ip}': 0.001}).remote()))
"                                                   # each node MUST print same torch/nccl
export RAY_ADDRESS=auto
bash /path/to/launch_lm_harness.sh 2>&1 | tee launch.log

== teardown ==
ray stop --force        # on both nodes
```
