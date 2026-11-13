// Compiles the executor (exec/qsha.c and its generated circuit header). tools/build.sh copies
// them into ./c; from a checkout of the whole tree they are found in ../../exec.
use std::path::PathBuf;

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let dir = [here.join("c"), here.join("../../exec")]
        .into_iter()
        .find(|d| d.join("qsha.c").exists())
        .expect("qsha.c not found: run tools/build.sh ord");
    for f in ["qsha.c", "qsha.h", "qsha_circuit.h"] {
        println!("cargo:rerun-if-changed={}", dir.join(f).display());
    }
    cc::Build::new().file(dir.join("qsha.c")).include(&dir).opt_level(3).warnings(false).compile("qsha");
}
