#!/usr/bin/env python3
"""
Usage:
  python monitor.py --output logs/runX/telemetry.jsonl --interval 5
"""

import os, sys, time, json, signal, psutil, argparse, datetime

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False


def get_gpu_info():
    if not NVML_OK:
        return []

    gpus = []
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h).decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        try:
            energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) * 1000.0
        except Exception:
            energy = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(h)
        except Exception:
            fan = None

        gpus.append({
            "gpu_index": i,
            "gpu_name": name,
            "power_watts": power,
            "energy_mJ": energy,
            "memory_used_MB": mem.used / (1024**2),
            "memory_total_MB": mem.total / (1024**2),
            "gpu_utilization_percent": util.gpu,
            "memory_utilization_percent": util.memory,
            "temperature_C": temp,
            "fan_speed_percent": fan,
        })
    return gpus


def get_cpu_info():
    cpu_util = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    return {
        "cpu_utilization_percent": cpu_util,
        "ram_used_MB": ram.used / (1024**2),
        "ram_total_MB": ram.total / (1024**2),
    }


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
