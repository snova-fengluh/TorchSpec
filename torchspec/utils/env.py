import os

# Environment variables that should be forwarded to all Ray actors.
# NOTE: TORCHINDUCTOR_CACHE_DIR is intentionally excluded — each node should
# use its own node-local default (/tmp/torchinductor_$USER/) to avoid
# cross-node triton kernel cache corruption over NFS.
_TORCHSPEC_ENV_KEYS = [
    "CUDA_LAUNCH_BLOCKING",
    "GLOO_SOCKET_IFNAME",
    "HF_HOME",
    "HF_TOKEN",
    "MC_LOG_LEVEL",
    "MODELOPT_MAX_TOKENS_PER_EXPERT",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SGLANG_DISABLE_CUDNN_CHECK",
    "SGLANG_VLM_CACHE_SIZE_MB",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "TORCHSPEC_LOG_DIR",
    "TORCHSPEC_LOG_LEVEL",
    "TP_SOCKET_IFNAME",
    "CUTE_DSL_CACHE_DIR",
    "TORCHSPEC_FLASH_ATTN_OPT_LEVEL",
]

# Prevent Ray from overriding VISIBLE_DEVICES so actors manage GPU assignment themselves.
# Reference: https://github.com/ray-project/ray/blob/161849364/python/ray/_private/accelerators/
_RAY_NOSET_VISIBLE_DEVICES_KEYS = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def resolve_local_socket_ifnames() -> None:
    """Fix NCCL/GLOO/TP_SOCKET_IFNAME for the node this process runs on.

    ``get_torchspec_env_vars`` forwards the DRIVER's values into every actor's
    runtime_env, overriding the per-node values the raylets were started with.
    NIC names differ across nodes, so the forwarded name may not exist here —
    or worse, name a local interface that has no IPv4 address (e.g. a down
    RoCE rail carrying only a link-local fe80:: address). Gloo then advertises
    an IPv6 address while peers advertise IPv4, and cross-node connect fails
    with ``ss1.ss_family == ss2.ss_family. 10 vs 2``.

    Call this at the start of every Ray actor that joins a distributed group.
    It rewrites each *_SOCKET_IFNAME var to the interface owning this node's
    Ray IP, but ONLY when the current value is unset, names an interface that
    doesn't exist locally, or names one without an IPv4 address. Deliberate
    multi-NIC configs (comma lists, ``^``/``=`` NCCL syntax, or a valid local
    IPv4 interface) are left untouched.
    """
    import socket

    import psutil

    try:
        import ray

        node_ip = ray.util.get_node_ip_address()
    except Exception:
        return

    if_addrs = psutil.net_if_addrs()
    ipv4_ifaces = {
        iface: [a.address for a in addrs if a.family == socket.AF_INET]
        for iface, addrs in if_addrs.items()
    }
    local_iface = next((i for i, v4 in ipv4_ifaces.items() if node_ip in v4), None)
    if local_iface is None:
        return

    for var in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"):
        cur = os.environ.get(var)
        if cur and (
            "," in cur or cur.startswith(("^", "="))  # deliberate NCCL multi-NIC syntax
            or ipv4_ifaces.get(cur)  # names a local interface with an IPv4 addr
        ):
            continue
        if cur != local_iface:
            from torchspec.utils.logging import logger

            logger.info(
                f"{var}={cur!r} is not a local IPv4 interface on this node; "
                f"rewriting to {local_iface!r} (owns node IP {node_ip})"
            )
            os.environ[var] = local_iface


def get_torchspec_env_vars() -> dict[str, str]:
    """Return common environment variables for all Ray actors.

    Includes:
    - TORCHSPEC_* variables (e.g. log level) from the current process
    - RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES = "1" to prevent Ray from
      overriding device visibility

    Intended for use with ``ray.remote(runtime_env={"env_vars": ...})``.
    Call-site env vars merged after this dict take higher priority.
    """
    env = {k: "1" for k in _RAY_NOSET_VISIBLE_DEVICES_KEYS}
    env.update({k: os.environ[k] for k in _TORCHSPEC_ENV_KEYS if k in os.environ})
    return env
