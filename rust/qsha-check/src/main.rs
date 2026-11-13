//! The patched crates against things that do not use the model: the `sha2` crate for digests,
//! the BIP340 test vector for signing. Every bitcoin_hashes / secp256k1 hash below is a run of
//! the compiled weights (there is no other implementation left in those crates).
use bitcoin_hashes::{hmac, sha256, sha256d, sha256t_hash_newtype, Hash, HashEngine};
use sha2::{Digest, Sha256};

sha256t_hash_newtype! {
    struct TapLeafTag = hash_str("TapLeaf");
    #[hash_newtype(forward)]
    struct TapLeaf(_);
}

fn native(m: &[u8]) -> [u8; 32] { Sha256::digest(m).into() }

fn main() {
    let (mut seed, mut n, mut tokens) = (0x9e3779b97f4a7c15u64, 0u32, 0u64);
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    // every length around the block and padding boundaries, then random lengths, random chunking
    for len in (0..=200usize).chain((0..300).map(|i| 201 + (i * 37) % 2800)) {
        let msg: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
        assert_eq!(sha256::Hash::hash(&msg).to_byte_array(), native(&msg), "one shot, len {len}");
        let mut e = sha256::HashEngine::default();
        let mut i = 0;
        while i < len { let k = 1 + (rnd() as usize) % 97; let j = (i + k).min(len); e.input(&msg[i..j]); i = j; }
        assert_eq!(sha256::Hash::from_engine(e).to_byte_array(), native(&msg), "chunked, len {len}");
        assert_eq!(sha256d::Hash::hash(&msg).to_byte_array(), native(&native(&msg)), "sha256d, len {len}");
        n += 3;
    }
    // tagged hash: the midstate upstream bakes in at compile time, here made by the model
    let t = native(b"TapLeaf");
    let data = b"\xc0\x02\x51\xac";
    let want = native(&[&t[..], &t[..], &data[..]].concat());
    assert_eq!(TapLeaf::hash(data).to_byte_array(), want, "tagged hash");
    // HMAC-SHA256, RFC 4231 test case 2
    let mut h = hmac::HmacEngine::<sha256::Hash>::new(b"Jefe");
    h.input(b"what do ya want for nothing?");
    assert_eq!(hmac::Hmac::<sha256::Hash>::from_engine(h).to_string(),
               "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    // Bitcoin's genesis header
    let hdr: Vec<u8> = (0..80).map(|i| u8::from_str_radix(&HDR[2 * i..2 * i + 2], 16).unwrap()).collect();
    assert_eq!(sha256d::Hash::hash(&hdr).to_string(), "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f");
    n += 3;
    // BIP340 test vector 0: the nonce, aux and challenge hashes run inside libsecp256k1
    let secp = secp256k1::global::SECP256K1;
    let mut sk = [0u8; 32]; sk[31] = 3;
    let kp = secp256k1::Keypair::from_seckey_slice(secp, &sk).unwrap();
    let msg = secp256k1::Message::from_digest([0u8; 32]);
    let sig = secp.sign_schnorr_with_aux_rand(&msg, &kp, &[0u8; 32]);
    assert_eq!(sig.to_string().to_uppercase(), "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215\
25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0");
    secp.verify_schnorr(&sig, &msg, &kp.x_only_public_key().0).unwrap();
    // ECDSA with RFC 6979 nonces (HMAC-SHA256 inside libsecp256k1), checked by verification
    let m2 = secp256k1::Message::from_digest(native(b"qwencoin"));
    let es = secp.sign_ecdsa(&m2, &kp.secret_key());
    secp.verify_ecdsa(&m2, &es, &kp.public_key()).unwrap();
    tokens += unsafe { qsha_tokens() };
    println!("OK: {n} digests equal the sha2 crate's, tagged hash, HMAC, genesis header, BIP340 vector 0, ECDSA; \
{tokens} model tokens");
}

extern "C" { fn qsha_tokens() -> u64; }

const HDR: &str = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e\
67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c";
