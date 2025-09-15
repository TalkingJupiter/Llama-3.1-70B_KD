#!/usr/bin/env python3
"""
Usage:
  python monitor.py --output logs/runX/telemetry.jsonl --interval 5
"""

import os, sys, time, json, signal, psutil, argparse, datetime

# ------------ NVML init (GPU telemetry) ------------
try:
    import pynvml  # a.k.a. nvidia-ml-py
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

def _to_str(x):
    # NVML sometimes returns bytes, sometimes str
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

def get_gpu_info():
    if not NVML_OK:
        return []
    out = []
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return out
    for i in range(count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = _to_str(pynvml.nvmlDeviceGetName(h))
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            # power in Watts
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power = None
            # total energy (mJ) if supported
            try:
                energy_mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) * 1000.0
            except Exception:
                energy_mJ = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(h)
            except Exception:
                fan = None

            out.append({
                "gpu_index": i,
                "gpu_name": name,
                "power_watts": power,
                "energy_mJ": energy_mJ,
                "memory_used_MB": mem.used / (1024**2),
                "memory_total_MB": mem.total / (1024**2),
                "gpu_utilization_percent": getattr(util, "gpu", None),
                "memory_utilization_percent": getattr(util, "memory", None),
                "temperature_C": temp,
                "fan_speed_percent": fan,
            })
        except Exception:
            # keep going if one device query fails
            continue
    return out

# ------------ CPU telemetry ------------
def get_cpu_info():
    cpu_util = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    info = {
        "cpu_utilization_percent": cpu_util,
        "ram_used_MB": ram.used / (1024**2),
        "ram_total_MB": ram.total / (1024**2),
    }
    # Optional: CPU power via Intel RAPL if available
    rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    try:
        with open(rapl_path, "r") as f:
            energy_uj = int(f.read().strip())
        info["cpu_energy_uj"] = energy_uj  # cumulative since boot
    except Exception:
        # Not available on all nodes / vendors / permissions
        info["cpu_energy_uj"] = None
    return info

# ------------ Main loop ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output JSONL file")
    ap.add_argument("--interval", type=int, default=5, help="Seconds between samples")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stop = False
    def handle_sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    with open(args.output, "a", encoding="utf-8") as f:
        while not stop:
            ts = datetime.datetime.utcnow().isoformat()
            entry = {
                "timestamp": ts,
                "gpus": get_gpu_info(),
                "cpu": get_cpu_info(),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
