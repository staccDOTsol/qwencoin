//! SHA-256 as a run of the compiled model. No hashing happens in Rust or in native C here:
//! `qsha.c` evaluates the weights of the hashformer circuit, token by token.
#![no_std]

use core::ffi::c_long;

extern "C" {
    /// One whole message from the SHA token: the model supplies the IV, the padding and the length.
    pub fn qsha_sha256(msg: *const u8, len: usize, out: *mut u8) -> c_long;
    /// The initial value, as the model prints it.
    pub fn qsha_iv(state: *mut u32);
    /// One compression of a full block; `nblk` counts the blocks hashed so far.
    pub fn qsha_compress(state: *mut u32, nblk: *mut u64, block: *const u8) -> c_long;
    /// The tail of a message whose first `nblk` blocks are in `state`: the model pads and prints.
    pub fn qsha_finish(state: *const u32, nblk: u64, tail: *const u8, n: usize, out: *mut u8) -> c_long;
    /// Model tokens run so far by this process.
    pub fn qsha_tokens() -> u64;
}

/// The model's IV.
pub fn iv() -> [u32; 8] {
    let mut s = [0u32; 8];
    unsafe { qsha_iv(s.as_mut_ptr()) };
    s
}

/// Compress one block into `state`. `done` is the number of blocks already hashed.
pub fn compress(state: &mut [u32; 8], done: u64, block: &[u8; 64]) {
    let mut n = done;
    let ran = unsafe { qsha_compress(state.as_mut_ptr(), &mut n, block.as_ptr()) };
    assert!(ran > 0 && n == done + 1, "the model did not run");
}

/// Finish a message: `done` full blocks are in `state`, `tail` (under 64 bytes) remains.
pub fn finish(state: &[u32; 8], done: u64, tail: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let ran = unsafe { qsha_finish(state.as_ptr(), done, tail.as_ptr(), tail.len(), out.as_mut_ptr()) };
    assert!(ran > 0, "the model did not run");
    out
}

/// A whole message of under 64 bytes, from the model's own start.
pub fn sha256_short(msg: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let ran = unsafe { qsha_sha256(msg.as_ptr(), msg.len(), out.as_mut_ptr()) };
    assert!(ran > 0, "the model did not run");
    out
}
