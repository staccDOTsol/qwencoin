"""Pick the backend for a merged model from the probe: the fastest one that ran the chip exactly
and whose dtype fits this machine. Prints "DEVICE DTYPE".

    python tools/pick.py logs/backend.json RAM_GB [VRAM_GB [MODEL_GB_IN_BF16]]
    python tools/pick.py --need logs/backend.json MODEL_GB_IN_BF16     the least memory (GB) a backend here would fit in

In memory, on a CPU: the weights once (twice their bf16 size as float32), the largest tensor a
second time while it loads, what a turn allocates beside them, and the nodes. As bfloat16 that is
4 GB beside the weights. As float32 what a turn allocates is twice the size as well. On
linux-aarch64 both dtypes compute the hash exactly and float32 is the faster one: with 4 GB
allowed for either, a 16 GB machine there took the merged 0.8B as float32, 8.7 GB of weights, and
had 15.2 GB in use while it answered one question with every function declared; the step was
ended before the machine stopped answering. 7 GB beside the weights as float32: such a machine
now takes the model as bfloat16, slower and in half the memory.
"""
import json, math, sys


def needs(dev: str, dt: str, size: float) -> float:
    """GB a backend takes for a model of `size` GB in bf16: of the card's memory on cuda, of the machine's else."""
    need = size * (1 if dt == "bfloat16" else 2)
    return need / 0.85 if dev == "cuda" else need * 1.08 + (4 if dt == "bfloat16" else 7)


def exact(runs: list) -> list:
    """(ms a token, device, dtype) of every backend that ran the chip exactly."""
    out = []
    for r in runs:
        if "error" in r:
            continue
        dev, dt = r["backend"].split("/")
        if r["teacher_forced"]["wrong"] == 0 and r["free_running"]["trace_ok"]:
            out.append((r["free_running"]["ms_per_token"], dev, dt))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--need":
        fit = [needs(dev, dt, float(sys.argv[3])) for _, dev, dt in exact(json.load(open(sys.argv[2]))["runs"]) if dev != "cuda"]
        print(math.ceil(min(fit)) if fit else "?")
        sys.exit(0)
    runs = json.load(open(sys.argv[1]))["runs"]
    ram = float(sys.argv[2]) if len(sys.argv) > 2 else 36.0
    vram = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    size = float(sys.argv[4]) if len(sys.argv) > 4 else 15.3  # the merged 4B
    ok = [(ms, dev, dt) for ms, dev, dt in exact(runs) if needs(dev, dt, size) <= (vram if dev == "cuda" else ram)]
    if not ok:
        sys.exit("no backend here runs the chip exactly in a dtype that fits")
    _, dev, dt = min(ok)
    print(dev, dt)
