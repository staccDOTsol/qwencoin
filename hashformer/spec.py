"""Trace format of the SHA-256 machine.

The machine is an autoregressive token stream. The host feeds message bytes; everything else
(the IV, the padding, the bit length, all 64 rounds, the chaining adds, the digest) is produced
by the model itself, one token per forward pass. State lives in Gated DeltaNet rows and reaches
them only through tokens, so the stream is a complete execution log of the hash.

Tokens
    B(x)   x in 0..255     a message byte (host), or a pad / length byte (model)
    V(p)   p in 0..65535   a 16-bit half word inside a record
    F(p)   p in 0..65535   the 16-bit half word that ends a record (state commits here)
    END    host: end of message. It *is* the 0x80 pad byte and occupies its slot in the block
    SHA    host: start a message; the model then emits the IV itself
    RES    host: resume from a midstate; the host supplies the chaining value and block count
    BLK    model: block boundary (resets the round and byte counters)
    R0     model: rounds 0..15 begin        R16    model: rounds 16..63 begin
    RC     model: chaining adds begin       RN     model: block counter record begins
    IN     model: "give me input" (the host answers with a byte, END, or a RES record token)
    HALT   model: the last 16 value tokens before this are the digest

Stream of one message (model tokens in brackets)
    SHA [cv x8] [n] [BLK]  bytes.. END [pad bytes] [R0 rounds R16 rounds RC cv x8] [HALT]
                           (64 bytes per block; a non-final block ends [RN n BLK] instead)

Records
    round t < 16    V V V F        E[t] hi, lo; A[t] hi, lo
    round t >= 16   V V V V V F    W[t] hi, lo; E[t] hi, lo; A[t] hi, lo
    cv word i       V F            hi, lo of H[i] (the IV, or old H[i] + working variable)
    n               V V V F        64-bit count of blocks done, most significant chunk first
"""

B_BASE = 0
CTRL_BASE = 256
CONTROL_NAMES = ["END", "SHA", "RES", "BLK", "IN", "HALT", "R0", "R16", "RC", "RN"]
CONTROLS = {n: CTRL_BASE + i for i, n in enumerate(CONTROL_NAMES)}
V_BASE = 512
F_BASE = V_BASE + 65536
VOCAB = F_BASE + 65536  # 131,584 chip tokens
PAD = VOCAB  # a zero row: not a chip token

END, SHA, RES, BLK, IN, HALT, R0, R16, RC, RN = (CONTROLS[n] for n in CONTROL_NAMES)

# Text tokens of a host language model, as the merged circuit sees them (textspec.py, bridge.py).
# In the circuit's own id space text token q is TEXT0 + q; in a merged checkpoint it is q and
# chip token t is V_host + t.
TEXT0 = 1 << 18
# output codes (family C) of the text tokens the circuit prints
PRINT_NAMES = ["NL", "TRO", "TRC", "THO", "THC", "NLNL"]  # "\n" <tool_response> </tool_response> <think> </think> "\n\n"
PRINT_CODE = {n: len(CONTROL_NAMES) + i for i, n in enumerate(PRINT_NAMES)}
HEX_CODE = 16  # hex digit x prints with code HEX_CODE + x

FAMILIES = {"B": (B_BASE, 256), "V": (V_BASE, 65536), "F": (F_BASE, 65536)}
FAM_ID = {"B": 0, "V": 1, "F": 2, "C": 3}  # output-code family ids (C: a control)
NFAM = 2
NPAY = 16


def B(x: int) -> int:
    return B_BASE + (x & 0xFF)


def V(p: int) -> int:
    return V_BASE + (p & 0xFFFF)


def F(p: int) -> int:
    return F_BASE + (p & 0xFFFF)


def family(t: int) -> tuple[str, int]:
    """(family, payload) of a chip token; controls are family "C" with their index as payload."""
    if 0 <= t < 256:
        return "B", t
    if CTRL_BASE <= t < CTRL_BASE + len(CONTROL_NAMES):
        return "C", t - CTRL_BASE
    if V_BASE <= t < F_BASE:
        return "V", t - V_BASE
    if F_BASE <= t < VOCAB:
        return "F", t - F_BASE
    raise ValueError(f"not a chip token: {t}")


def token(fam_id: int, payload: int) -> int | None:
    """The token with this output code, or None if no token has it."""
    if fam_id == 0:
        return B_BASE + payload if payload < 256 else None
    if fam_id == 1:
        return V_BASE + payload
    if fam_id == 2:
        return F_BASE + payload
    return CTRL_BASE + payload if payload < len(CONTROL_NAMES) else None


def code_bits(t: int) -> list[int]:
    """Output code of a chip token: family id (NFAM bits) then payload (NPAY bits), LSB first."""
    fam, p = family(t)
    fid = FAM_ID[fam]
    return [fid >> i & 1 for i in range(NFAM)] + [p >> i & 1 for i in range(NPAY)]


def name(t: int) -> str:
    if t == PAD:
        return "PAD"
    fam, p = family(t)
    if fam == "C":
        return CONTROL_NAMES[p]
    return f"{fam}({p:02x})" if fam == "B" else f"{fam}({p:04x})"


# SHA-256 constants (FIPS 180-4). The compiler bakes these into the weights.
IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
      0x5BE0CD19]
K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4,
    0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE,
    0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F,
    0x4A7484AA, 0x5CB0A9DC, 0x76F988DA, 0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC,
    0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
    0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070, 0x19A4C116,
    0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7,
    0xC67178F2,
]
