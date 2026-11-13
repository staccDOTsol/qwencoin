"""Pick the backend for a merged model from the probe: the fastest one that ran the chip exactly
and whose dtype fits this machine. Prints "DEVICE DTYPE".

    python tools/pick.py logs/backend.json RAM_GB [VRAM_GB [MODEL_GB_IN_BF16]]
"""
import json, sys

runs = json.load(open(sys.argv[1]))["runs"]
ram = float(sys.argv[2]) if len(sys.argv) > 2 else 36.0
vram = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
size = float(sys.argv[4]) if len(sys.argv) > 4 else 15.3  # the merged 4B
ok = []
for r in runs:
    if "error" in r:
        continue
    dev, dt = r["backend"].split("/")
    exact = r["teacher_forced"]["wrong"] == 0 and r["free_running"]["trace_ok"]
    need = size * (1 if dt == "bfloat16" else 2)
    # in memory: the weights once, the largest tensor a second time while it loads, the nodes beside it
    fits = need <= 0.85 * vram if dev == "cuda" else need * 1.08 + 4 <= ram
    if exact and fits:
        ok.append((r["free_running"]["ms_per_token"], dev, dt))
if not ok:
    sys.exit("no backend here runs the chip exactly in a dtype that fits")
_, dev, dt = min(ok)
print(dev, dt)
