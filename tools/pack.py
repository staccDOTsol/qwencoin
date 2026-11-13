"""Pack the node binaries of one platform so that they run on any machine of that platform.

    python3 tools/pack.py OUT_DIR [--from BIN_DIR] [--libdirs DIR:DIR...] [--readelf PROG]

OUT_DIR gets bin/ (small launchers), libexec/ (qwencoind, qwencoin-cli, ord as built) and, on
Linux, lib/ (the loader and the shared libraries the programs were built against, so they run
on any distro whatever its glibc). On macOS the programs are packed as they are. exec/ gets
the circuit's executor and the work kernel compiled as libraries for the platform (what the
Python side loads: hashformer/executor.py, hashformer/work.py); --cc names the compiler.

Building where the programs run (the usual case), the libraries are found with ldd. For
programs built for another architecture (aarch64 on an x86-64 machine), --libdirs names the
folders of that architecture's libraries and --readelf its readelf.
"""
import argparse, platform, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("qwencoind", "qwencoin-cli", "ord")
WRAP = '''#!/bin/bash
# the real program is libexec/{name}; on Linux it runs on the loader and libraries shipped in lib/
D="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
for L in "$D"/lib/ld-linux-*.so.*; do [ -x "$L" ] && exec "$L" --library-path "$D/lib" "$D/libexec/{name}" "$@"; done
exec "$D/libexec/{name}" "$@"
'''


def needed_ldd(binary):
    r = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
    for line in r.splitlines():
        parts = line.split()
        if "=>" in parts and len(parts) > 2 and parts[2].startswith("/"):
            yield Path(parts[2])
        elif parts and parts[0].startswith("/"):
            yield Path(parts[0])


def needed_readelf(binary, libdirs, readelf, seen=None):
    """The shared libraries a program of another architecture needs, found in libdirs (and theirs)."""
    seen = {} if seen is None else seen
    d = subprocess.run([readelf, "-d", "-l", str(binary)], capture_output=True, text=True, check=True).stdout
    names = [l.split("[", 1)[1].split("]")[0] for l in d.splitlines() if "(NEEDED)" in l]
    names += [Path(l.split(":", 1)[1].strip(" ]")).name for l in d.splitlines() if "program interpreter" in l]
    for n in names:
        if n in seen:
            continue
        hit = next((Path(x) / n for x in libdirs if (Path(x) / n).exists()), None)
        if hit is None:
            raise SystemExit(f"{n} (needed by {binary}) is in none of {libdirs}")
        seen[n] = hit
        needed_readelf(hit.resolve(), libdirs, readelf, seen)
    return seen


def bundle_macos(out: Path) -> list:
    """Programs built against Homebrew's libraries: those libraries go into lib/ and the programs
    (and the libraries among themselves) are told to load them from there. Everything is signed
    ad hoc again afterwards: changing a load path breaks the signature, and an arm64 Mac runs
    nothing unsigned. SIGN_IDENTITY names a Developer ID to sign with instead."""
    import os
    lib, done = out / "lib", {}
    todo = [out / "libexec" / n for n in NAMES]
    while todo:
        f = todo.pop()
        for line in subprocess.run(["otool", "-L", str(f)], capture_output=True, text=True, check=True).stdout.splitlines()[1:]:
            dep = line.strip().split(" (")[0]
            if not dep or dep.startswith(("/usr/lib/", "/System/", "@")):
                continue
            name = Path(dep).name
            if name not in done:
                lib.mkdir(exist_ok=True)
                shutil.copy(Path(dep).resolve(), lib / name)
                (lib / name).chmod(0o755)
                subprocess.run(["install_name_tool", "-id", f"@loader_path/{name}", str(lib / name)], check=True)
                done[name] = lib / name
                todo.append(lib / name)
            where = f"@executable_path/../lib/{name}" if f.parent.name == "libexec" else f"@loader_path/{name}"
            subprocess.run(["install_name_tool", "-change", dep, where, str(f)], check=True)
    ident = os.environ.get("SIGN_IDENTITY") or "-"
    for f in list(done.values()) + [out / "libexec" / n for n in NAMES]:
        subprocess.run(["codesign", "--force", "--sign", ident] + (["--options", "runtime", "--timestamp"] if ident != "-" else []) + [str(f)], check=True)
    return sorted(done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--from", dest="src", default=str(ROOT / "bin"))
    ap.add_argument("--libdirs", default="")
    ap.add_argument("--readelf", default="readelf")
    ap.add_argument("--cc", default="gcc")
    a = ap.parse_args()
    out, src = Path(a.out), Path(a.src)
    for d in ("bin", "libexec", "lib", "exec"):
        shutil.rmtree(out / d, ignore_errors=True)
    (out / "exec").mkdir(parents=True)
    import os
    if platform.system() == "Darwin":
        os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "13.0")  # as the programs: macOS 13 and later
    # the same commands as hashformer/executor.py and hashformer/work.py use where there is a compiler
    subprocess.run([a.cc, "-O3", "-fPIC", "-shared", "-pthread", "-o", str(out / "exec" / "libqsha.so"), str(ROOT / "exec" / "qsha.c"), "-lm"], check=True)
    subprocess.run([a.cc, "-O3", "-fPIC", "-shared", "-o", str(out / "exec" / "libqwork.so"), str(ROOT / "exec" / "qwork.c")], check=True)
    (out / "bin").mkdir(parents=True)
    (out / "libexec").mkdir()
    libs = {}
    for name in NAMES:
        b = src / name
        shutil.copy(b, out / "libexec" / name)
        (out / "libexec" / name).chmod(0o755)
        (out / "bin" / name).write_text(WRAP.format(name=name))
        (out / "bin" / name).chmod(0o755)
        if platform.system() == "Linux" or a.libdirs:
            if a.libdirs:
                libs.update(needed_readelf(b, a.libdirs.split(":"), a.readelf))
            else:
                libs.update({p.name: p for p in needed_ldd(b)})
    if libs:
        (out / "lib").mkdir()
        for n, p in sorted(libs.items()):
            shutil.copy(p.resolve(), out / "lib" / n)
            (out / "lib" / n).chmod(0o755)
        assert any(n.startswith("ld-linux-") for n in libs), sorted(libs)
        pk = subprocess.run(["dpkg", "-S"] + [str(p.resolve()) for p in libs.values()], capture_output=True, text=True).stdout
        (out / "lib" / "LICENSES.txt").write_text(
            "The files in this folder are the shared libraries and the loader the node binaries were built against,\n"
            "copied unchanged from the build machine so that the binaries run on any Linux of this architecture.\n"
            "They keep their own licenses: the GNU C Library and libstdc++/libgcc under the LGPL and the GPL with the\n"
            "GCC Runtime Library Exception, libevent under the 3-clause BSD license, SQLite in the public domain.\n"
            "Their sources are the distribution's source packages of the same versions:\n\n" + pk)
    mac = bundle_macos(out) if platform.system() == "Darwin" and not a.libdirs else []
    print(f"{out}: {', '.join(NAMES)}" + (f"; libraries: {', '.join(sorted(libs) or mac)}" if libs or mac else "; no libraries packed"))
