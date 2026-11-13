"""Assemble the kit for another x86_64 Linux machine (or Windows through WSL2): the code, the
node binaries with the libraries and loader they were built against (so they run on any distro,
whatever its glibc), and the entry scripts.

    python3 tools/kit.py OUT.zip
"""
import os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "tools"))
from pack import WRAP  # noqa: E402  (the launcher: the program on the loader and libraries shipped in lib/)


def libs(binary):
    r = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
    for line in r.splitlines():
        parts = line.split()
        if "=>" in parts and parts[2].startswith("/"):
            yield Path(parts[2])
        elif parts[0].startswith("/"):
            yield Path(parts[0])


files = {}  # archive name -> (bytes, mode)


def add(name, data, mode=0o644):
    files[name] = (data, mode)


add("qwencoin/go.cmd", (ROOT / "win/go.cmd").read_bytes())
add("qwencoin/README.txt", (ROOT / "win/README.txt").read_bytes().replace(b"\n", b"\r\n"))
add("qwencoin/kit/go.sh", (ROOT / "win/go.sh").read_bytes(), 0o755)
for folder, pats in {"hashformer": ["*.py"], "rails": ["*.py"], "tests": ["*.py"], "graph": ["*.py"],
                     "tools": ["*.py", "*.sh", "*.md"], "patches": ["*.patch"], "win": ["*"],
                     "exec": ["qsha.c", "qsha.h", "qsha_circuit.h", "libqsha.so", "qwork.c", "qwork.h", "libqwork.so"],
                     # what the checks printed on the machine the kit was built on
                     "logs": ["qwork-node.log", "unit-qwork.log", "work.log", "tools.log", "untied.log", "remerge.log", "bridge-ir.log",
                              "merge-0.8B.log", "functional-qwork.log", "functional-tests.log", "unit-tests.log"]}.items():
    for pat in pats:
        for f in sorted((ROOT / folder).glob(pat)):
            if f.is_file():
                add(f"qwencoin/kit/{folder}/{f.name}", f.read_bytes(), 0o755 if f.suffix in (".sh", ".so") else 0o644)
for f in sorted((ROOT / "rust").rglob("*")):  # the executor as a Rust crate, and its check: ord's build needs them
    if f.is_file() and "target" not in f.parts:
        add(f"qwencoin/kit/rust/{f.relative_to(ROOT / 'rust').as_posix()}", f.read_bytes())
add("qwencoin/kit/MODELS", b"0.8B 2B 4B 9B 27B\n")
add("qwencoin/kit/vendor/COPYING", (ROOT / "upstream/bitcoin/COPYING").read_bytes())  # the test framework is Bitcoin Core's (MIT)
tf = ROOT / "upstream/bitcoin/test/functional/test_framework"  # the graph builds its transactions with Core's Python
for f in sorted(tf.rglob("*")):
    if f.is_file() and "__pycache__" not in f.parts:
        add(f"qwencoin/kit/vendor/test_framework/{f.relative_to(tf).as_posix()}", f.read_bytes())
for f in ("README.md", "RESULTS.md"):
    add(f"qwencoin/kit/{f}", (ROOT / f).read_bytes())
need = set()
for name in ("qwencoind", "qwencoin-cli", "ord"):
    b = ROOT / "bin" / name
    add(f"qwencoin/kit/libexec/{name}", b.read_bytes(), 0o755)
    add(f"qwencoin/kit/bin/{name}", WRAP.format(name=name).encode(), 0o755)
    need |= set(libs(b))
for lib in sorted(need):
    add(f"qwencoin/kit/lib/{lib.name}", lib.resolve().read_bytes(), 0o755)
pkgs = subprocess.run(["dpkg", "-S"] + [str(l.resolve()) for l in sorted(need)], capture_output=True, text=True).stdout
add("qwencoin/kit/lib/LICENSES.txt", (
    "The files in this folder are the shared libraries and the loader the node binaries were built against,\n"
    "copied unchanged from the build machine (" + subprocess.run(["lsb_release", "-ds"], capture_output=True, text=True).stdout.strip()
    + ") so that the binaries run on any x86_64 Linux.\nThey keep their own licenses: the GNU C Library and libstdc++/libgcc under the LGPL "
    "and the GPL with the GCC Runtime Library Exception,\nlibevent under the 3-clause BSD license, SQLite in the public domain. "
    "Their sources are the distribution's source packages of the same versions:\n\n" + pkgs).encode())
assert "qwencoin/kit/lib/ld-linux-x86-64.so.2" in files, sorted(files)
for name, (data, _) in files.items():
    if (name.endswith((".sh", ".py")) and "/vendor/" not in name) or "/kit/bin/" in name:
        assert b"\r" not in data, name
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, (data, mode) in sorted(files.items()):
        zi = zipfile.ZipInfo(name, date_time=(2026, 11, 12, 0, 0, 0))
        zi.external_attr = (0o100000 | mode) << 16
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, data)
print(f"{out}: {len(files)} files, {out.stat().st_size / 2**20:.1f} MB; libraries: {', '.join(sorted(l.name for l in need))}")
