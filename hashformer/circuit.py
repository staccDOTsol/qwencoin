"""SHA-256 as a circuit of Qwen3.5 sublayers (spec.py has the trace, ref.py the reference).

Every position computes the token that follows it. State changes only through tokens: a head
writes a row from token payloads at lags 0..3, so the stream is the hash's execution log.

  L1  attn  counters T (records in the stage), J (V tokens in the record), NB (bytes in the
            block): rows that halve per step; the stage register and the `ended` flag
      mlp   thermometers of the counters, gated by stage; clears the token's own output code
  L2  attn  the working-variable history A[t], E[t] (rows t + 4; four read ports, A and E as
            the two value heads of each key head), the schedule W[t] (four read ports), the
            block counter N, the `last` flag
      mlp   Sigma0/1, sigma0/1, Ch, Maj on the port outputs, K[t], the N chunk; which token
            kind comes next
  L3  attn  the bit length LEN, captured at END, read out one byte at a time
      mlp   adder stage 1: per nibble column, the carry out of the sum of up to 7 operands
  L4  mlp   adder stage 2: generate / propagate / digit bits of each column
  L5  mlp   adder stage 3: carry lookahead, the half word to print, the output code (the exit)

One 32-bit adder serves every sum (W[t], E[t], A[t], the chaining adds, the block count, the
length bytes): the read ports are gated by what the position has to print, so only that sum's
operands reach the adder. The IV and K[t] are constants of the weights.

With ``text`` (a TextSpec: the tokens of the language model the chip is merged into) the circuit
also reads that model's text and answers its tool call, with no host in between (the bridge):

  L1  attn  FNOK / ARMOK: the call so far is  <tool_call> .. sha .. hex  and only legal tokens
            since; four counters of the hex tokens since the parameter name (by the bits of
            their nibble count); PREV, the last nibble of the previous hex token; PT, the text
            tokens since the latest chip token
      mlp   thermometers of those counters: the nibble count and its parity
  L2  mlp   the bytes a hex token completes (up to 4, a nibble may wait for the next token)
  L3  attn  MEM0..3: the payload, one row per byte, written while the model writes its call
      mlp   fire: this token is </tool_call> of a well-formed call (the circuit emits SHA)
  L4  attn  BR: this hash was started by a call; then the bytes come from MEM instead of IN
  L5  attn  PRN: the digest is complete; the answer is printed as text
      mlp   <tool_response>, the 64 hex digits (the chaining value read back through the
            adder), </tool_response> and an empty <think> block; then the language model speaks

A position carries an output code only where the circuit owns the next token: after every chip
token, at a firing </tool_call>, and while it prints. Everywhere else the code is exactly 0.
"""

import numpy as np

from . import spec as S
from .gadgets import Circ, eq, from_bits
from .ir import Chan, Expr, K, OUT_W, Target, as_expr, esum
from .textspec import KMAX, MAX_NIBBLES, N_PRINT, PRINT_HEAD, PRINT_TAIL, SLOTS

CODE_AMP = 128.0  # a token's own code in its (tied) embedding row: CODE_AMP * (2 * bit - 1)
KAPPA = 2.0**-20  # the predicted code the exit leaves: KAPPA * (2 * bit - 1)
NCODE = S.NFAM + S.NPAY
LAG1 = (0.0, 1.0, 0.0, 0.0)


def lag(n: int) -> tuple:
    return tuple(1.0 if i == n else 0.0 for i in range(4))


def nib(x: int, k: int) -> int:
    return x >> 4 * k & 15


class Sha(Circ):
    def __init__(self, target: Target | None = None, text=None) -> None:
        super().__init__(target)
        self.text = text
        f = self.feature
        ctl = {n: f(f"is{n}", lambda t, v=v: t == v) for n, v in S.CONTROLS.items()
               if n in ("END", "SHA", "RES", "BLK", "R0", "R16", "RC", "RN")}
        self.ctl = ctl
        self.isB = f("isB", lambda t: t < 256)
        self.isV = f("isV", lambda t: (t >= S.V_BASE) & (t < S.F_BASE))
        self.isF = f("isF", lambda t: (t >= S.F_BASE) & (t < S.VOCAB))
        # 16-bit payload of V / F tokens; byte of B tokens (END reads as the 0x80 pad byte)
        self.pay = [f(f"pay{i}", lambda t, i=i: np.where(t >= S.V_BASE, (t - S.V_BASE) >> i & 1, 0))
                    for i in range(16)]
        self.bb = [f(f"bb{i}", lambda t, i=i: np.where(t < 256, t >> i & 1,
                                                    np.where(t == S.END, 0x80 >> i & 1, 0)))
                   for i in range(8)]
        self.code = [f(f"code{i}", lambda t, i=i: CODE_AMP * (2 * code_table()[t, i] - 1))
                     for i in range(NCODE)]
        self.keep: list[int] = []  # dims the output head reads (the code)
        if text is not None:
            self.n_text = text.n
            tf = lambda name, table: f(name, text=np.asarray(table, dtype=np.float64))  # noqa: E731
            self.isText = tf("isText", np.ones(text.n))
            self.isTCO, self.isTCC = tf("isTCO", text.is_tco), tf("isTCC", text.is_tcc)
            self.isFN, self.isPARAM = tf("isFN", text.is_fn), tf("isPARAM", text.is_param)
            # a token that cannot stand between the parameter name and </tool_call> (any chip token too)
            self.isBRK = f("isBRK", lambda t: np.ones(len(t)), text=text.is_brk.astype(np.float64))
            self.nbit = [tf(f"n.bit{b}", text.nn >> b & 1) for b in range(4)]
            self.nge = [None] + [tf(f"n>={v}", text.nn >= v) for v in range(1, KMAX + 1)]
            self.rn = [[tf(f"rn{j}[{b}]", text.nib[:, j] >> b & 1) for b in range(4)] for j in range(KMAX)]
            for i, e in enumerate(self.code):  # the printed tokens' own codes (family C)
                table = np.zeros(text.n)
                for q, p in text.code.items():
                    bit = 1 if i < S.NFAM else p >> (i - S.NFAM) & 1
                    table[q] = CODE_AMP * (2 * bit - 1)
                self.text_features[next(iter(e.terms))] = table


_CODE = None


def code_table() -> np.ndarray:
    """(VOCAB, NCODE) output-code bits of every chip token."""
    global _CODE
    if _CODE is None:
        t = np.arange(S.VOCAB)
        fam = np.where(t < 256, 0, np.where(t < S.V_BASE, 3, np.where(t < S.F_BASE, 1, 2)))
        payload = np.where(t < 256, t, np.where(t < S.V_BASE, t - S.CTRL_BASE,
                                              np.where(t < S.F_BASE, t - S.V_BASE, t - S.F_BASE)))
        bits = [fam >> i & 1 for i in range(S.NFAM)] + [payload >> i & 1 for i in range(S.NPAY)]
        _CODE = np.stack(bits, 1)
    return _CODE


def build(target: Target | None = None, text=None) -> Sha:
    c = Sha(target, text)
    bridge = text is not None
    one, pay, bb, ctl = c.one, c.pay, c.bb, c.ctl
    isV, isF = c.isV, c.isF
    isEND, isSHA, isRES, isBLK = ctl["END"], ctl["SHA"], ctl["RES"], ctl["BLK"]
    is_byte = c.isB + isEND

    # ------------------------------------------------------------ L1 attn: counters, stage
    t_reset = esum(ctl[n] for n in ("SHA", "RES", "BLK", "R0", "R16", "RC", "RN"))
    j_reset = one - isV
    nb_reset = isSHA + isRES + isBLK
    xT = c.counter("T", t_reset, isF)
    xJ = c.counter("J", j_reset, isV)
    xNB = c.counter("NB", nb_reset, is_byte)
    stage_bits = {2 + i: ctl[n] for i, n in enumerate(("SHA", "RES", "BLK", "R0", "R16", "RC", "RN"))}
    # on a text token no stage is read: the machine predicts nothing there
    stage, (ended,) = c.table("stage", {0: one}, {0: one}, [
        {"data": stage_bits, "ref": one, "beta": t_reset, "gate": one - c.isText if bridge else one},
        {"data": {2: isEND}, "ref": one, "beta": isEND + isSHA + isRES},
    ])
    INIT, RESM, BYTES, EARLY, LATE, CVS, NS = stage
    if bridge:
        isT, chipt = c.isText, one - c.isText
        ((fnok,),) = c.table("FNOK", {0: one}, {0: one}, [
            {"data": {2: c.isFN}, "ref": one, "beta": c.isFN + c.isTCO}])
        ((armok,),) = c.table("ARMOK", {0: one}, {0: one}, [
            {"data": {2: c.isPARAM}, "ref": one, "beta": c.isPARAM + c.isBRK}])
        xcb = [c.counter(f"CB{b}", c.isPARAM, c.nbit[b]) for b in range(4)]
        xPT = c.counter("PT", chipt, isT)
        ((*prev,),) = c.table("PREV", {0: c.nge[1]}, {0: one}, [
            {"data": {2 + b: Chan(c.rn[0][b], LAG1) for b in range(4)}, "ref": one}], key_taps=LAG1)

    # ------------------------------------------------------------ L1 mlp: thermometers
    gE = c.thermo(xT, t_reset, range(16), "T.E", [EARLY])
    gL = c.thermo(xT, t_reset, range(48), "T.L", [LATE])
    gC = c.thermo(xT, t_reset, range(8), "T.C", [CVS])
    gI = c.thermo(xT, t_reset, range(9), "T.I", [INIT])
    gR = c.thermo(xT, t_reset, range(9), "T.R", [RESM])
    gN = c.thermo(xT, t_reset, [0], "T.N", [NS])
    cvlike = CVS + INIT + RESM
    fC = c.thermo(xT, t_reset, [0, 4, 8], "T.fC", [isF, cvlike])  # F tokens of cv / n records
    jE = c.thermo(xJ, j_reset, range(3), "J.E", [EARLY])
    jL = c.thermo(xJ, j_reset, range(5), "J.L", [LATE])
    jC = c.thermo(xJ, j_reset, [0], "J.C", [CVS])
    jI = c.thermo(xJ, j_reset, range(3), "J.I", [INIT])
    jN = c.thermo(xJ, j_reset, range(3), "J.N", [NS])
    nb = c.thermo(xNB, nb_reset, range(64), "NB")
    word_ends = [v for r in range(16) for v in (4 * r + 3, 4 * r + 4) if v < 64]
    nbB = c.thermo(xNB, nb_reset, word_ends, "NB.byte", [is_byte])
    end_late = c.thermo(xNB, nb_reset, [56], "NB.end", [isEND])[56]  # END at index >= 56
    if bridge:
        # nibbles since the parameter name: cb[b] counts the hex tokens with bit b in their length
        cap = (MAX_NIBBLES, MAX_NIBBLES // 2, MAX_NIBBLES // 4, MAX_NIBBLES // 8)
        thb = [c.thermo(xcb[b], c.isPARAM, range(cap[b] + 1), f"CB{b}") for b in range(4)]
        cnt = esum((1 << b) * thb[b][m] for b in range(4) for m in range(cap[b]))  # exact up to 64 per counter
        ovf = esum(thb[b][cap[b]] for b in range(4))  # a counter ran past its range
        par = esum(thb[0][2 * j] - thb[0][2 * j + 1] for j in range(MAX_NIBBLES // 2))  # count is odd
        thP = c.thermo(xPT, chipt, range(N_PRINT), "PT")  # [text tokens since the chip token > m]
        eqPW = [thP[2 + 8 * w] - thP[10 + 8 * w] for w in range(8)]  # the position prints a digit of word w
        pA, pE = thP[2] - thP[34], thP[34] - thP[66]
    f_round = c.and_([isF, EARLY + LATE], "F.round")  # an F token that ends a round
    f_ns = c.and_([isF, NS], "F.ns")
    blk_ended = c.and_([isBLK, ended], "BLK.ended")
    for i, cd in enumerate(c.code):  # the token's own code goes; the exit writes the next one
        c.ffn([(-1, cd, one)], f"code{i}.clear", into=cd)

    eqE = [eq(gE, EARLY, v) for v in range(17)]  # [EARLY & T == v]
    eqL = [eq(gL, LATE, v) for v in range(49)]
    eqC = [eq(gC, CVS, v) for v in range(9)]
    eqI = [eq(gI, INIT, v) for v in range(10)]
    eqR = [eq(gR, RESM, v) for v in range(10)]
    eqCV = [eqC[i] + eqI[i] + eqR[i] for i in range(8)]  # a cv record's word index
    eqNB = [None] + [eq(nb, one, v) for v in range(1, 65)]  # [NB == v], v >= 1

    def eq_byte(v):  # [byte token & NB == v] at a word's last byte
        return nbB[v - 1] - nbB[v] if v < 64 else nbB[v - 1]

    def jeq(th, flag, v):
        return eq(th, flag, v)

    # what the position has to print (0/1 forms; they may also be set where nothing is printed)
    cW = LATE - jL[1]  # LATE, J <= 1: a half of W[t]
    cE = (EARLY - jE[1]) + (jL[1] - jL[3])  # a half of E[t]
    cA = jE[1] + jL[3]  # a half of A[t]
    cEA = EARLY + jL[1]
    cCa = CVS - gC[3]  # a chaining add over the A rows
    cCe = gC[3] - gC[7]  # ... over the E rows

    def idx(lst, i):
        return lst[i] if 0 <= i < len(lst) else Expr()

    # ------------------------------------------------------------ L2 attn: A / E history
    # row r holds A[r - 4] (value head 0) and E[r - 4] (value head 1); rows 0..3 are the chaining
    # value. The key comes from t-1: a record's last token has already stepped T.
    ae_keys = {}
    for r in range(68):
        e = idx(eqE[:16], r - 4) + idx(eqL[:48], r - 20)
        if r <= 3:
            e = e + eqCV[3 - r] + eqCV[7 - r]
        ae_keys[r] = e
    word = {i: (pay[i] if i < 16 else Chan(pay[i - 16], LAG1)) for i in range(32)}

    def word_data(sets):
        return {base + i: word[i] for base in sets for i in range(32)}

    wA = f_round + fC[0] - fC[4]
    wE = (jE[1] - jE[2]) + (jL[3] - jL[4]) + (fC[4] - fC[8])
    ports = []
    for p in (1, 2, 3):
        q = {r: idx(eqE[:16], r - 4 + p) + idx(eqL[:48], r - 20 + p) for r in range(68)}
        q = {r: e for r, e in q.items() if e.terms}
        a_, e_ = c.table(f"AE{p}", ae_keys, q, [
            {"data": word_data([2]), "ref": one, "beta": wA, "gate": cA,
             "outs": [(2 + i, OUT_W, f"A{p}[{i}]") for i in range(32)]},
            {"data": word_data([2]), "ref": one, "beta": wE, "gate": cEA,
             "outs": [(2 + i, OUT_W, f"E{p}[{i}]") for i in range(32)]},
        ], key_taps=LAG1)
        ports.append((a_, e_))
    (A1, E1), (A2, E2), (A3, E3) = ports
    # port 4: A[t-4] and E[t-4] in a round; in a chaining add, two rows at once (their sum)
    q4 = {}
    for r in range(68):
        e = idx(eqE[:16], r) + idx(eqL[:48], r - 16)
        if r <= 3:
            e = e + eqC[3 - r] + eqC[7 - r]
        if r >= 64:
            e = e + idx(eqC[:8], 67 - r) + idx(eqC[:8], 71 - r)
        if bridge and r <= 3:  # the digest, read back word by word while it is printed
            e = e + eqPW[3 - r] + eqPW[7 - r]
        if e.terms:
            q4[r] = e

    def two_sets(nm):
        return [(2 + i, OUT_W, f"{nm}[{i}]") for i in range(32)] + \
               [(34 + i, 2 * OUT_W, f"{nm}[{i}]") for i in range(32)]

    def set_gates(g1, g2):
        return {**{2 + i: g1 for i in range(32)}, **{34 + i: g2 for i in range(32)}}

    D, H = c.table("AE4", ae_keys, q4, [
        {"data": word_data([2, 34]), "ref": one, "beta": wA,
         "gates": set_gates(cE + pA if bridge else cE, cCa), "outs": two_sets("D")},
        {"data": word_data([2, 34]), "ref": one, "beta": wE,
         "gates": set_gates(cEA + pE if bridge else cEA, cCe), "outs": two_sets("H")},
    ], key_taps=LAG1)

    # ------------------------------------------------------------ L2 attn: schedule W
    # row t holds W[t]: columns 2.. when it came as two half words, 34.. as four bytes
    w_keys = {r: (eq_byte(4 * (r + 1)) if r < 16 else eqL[r - 16]) for r in range(64)}
    w_data = {2 + i: word[i] for i in range(32)}
    w_data.update({34 + i: Chan(bb[i % 8], lag(i // 8)) for i in range(32)})
    w_beta = is_byte + (jL[1] - jL[2])

    def w_port(nm, q, gate, weight=OUT_W):
        (out,) = c.table(nm, w_keys, {r: e for r, e in q.items() if e.terms}, [
            {"data": w_data, "ref": one, "beta": w_beta,
             "gates": {col: gate for col in range(2, 66)},
             "outs": [(base + i, weight, f"{nm}[{i}]") for i in range(32) for base in (2, 34)]}])
        return out

    Wt = w_port("Wt", {r: idx(eqE[:16], r) + idx(eqL[:48], r - 16) for r in range(64)}, cEA)
    W2 = w_port("W2", {r: idx(eqL[:48], r - 14) for r in range(64)}, cW)
    W15 = w_port("W15", {r: idx(eqL[:48], r - 1) for r in range(64)}, cW)
    W716 = w_port("W7+16", {r: idx(eqL[:48], r - 9) + idx(eqL[:48], r) for r in range(64)}, cW,
                  2 * OUT_W)

    # ------------------------------------------------------------ L2 attn: N, last
    (N,) = c.table("N", {0: one}, {0: one}, [
        {"data": {2 + i: Chan(pay[i % 16], lag(i // 16)) for i in range(64)}, "ref": one,
         "beta": fC[8] + f_ns}])
    ((last,),) = c.table("last", {0: one}, {0: one}, [
        {"data": {2: (isEND - end_late) + blk_ended}, "ref": one,
         "beta": isEND + isBLK + isSHA + isRES}])

    # ------------------------------------------------------------ L2 mlp: round functions
    selN = [jeq(jN, NS, 3 - ch) for ch in range(4)]  # the n record prints chunk ch
    klate = [esum(nib(S.K[T + 16], k) * eqL[T] for T in range(48)) for k in range(8)]
    T0 = []
    for k in range(8):
        terms = []
        for b in range(4):
            i, w = 4 * k + b, 1 << b
            terms += c.t_xor([E1[(i + 6) % 32], E1[(i + 11) % 32], E1[(i + 25) % 32]], w)
            terms += [(w, E2[i] - E3[i], E1[i])]  # Ch = g + e (f - g); g is added as a row
            terms += c.t_xor([A1[(i + 2) % 32], A1[(i + 13) % 32], A1[(i + 22) % 32]], w)
            s = A1[i] + A2[i] + A3[i]
            terms += [(w, one, s - 1), (-w, one, s - 2)]  # Maj
            terms += c.t_xor([W2[(i + 17) % 32], W2[(i + 19) % 32]]
                             + ([W2[i + 10]] if i + 10 < 32 else []), w)
            terms += c.t_xor([W15[(i + 7) % 32], W15[(i + 18) % 32]]
                             + ([W15[i + 3]] if i + 3 < 32 else []), w)
        terms += [(1, klate[k], jL[1])]  # K[t] in rounds 16..63, not while W[t] is printed
        if k < 4:
            terms += [(1, from_bits(N[16 * ch + 4 * k:16 * ch + 4 * k + 4]), selN[ch])
                      for ch in range(4)]
        if k == 0:  # the carry into chunk ch of N + 1: every lower bit is set
            for ch in (1, 2, 3):
                terms += c.t_and([selN[ch]] + N[:16 * ch])
        T0.append(c.ffn(terms, f"T0[{k}]"))

    # ------------------------------------------------------------ L2 mlp: what comes next
    nE, nL, nC = EARLY - gE[15], LATE - gL[47], CVS - gC[7]  # the stage still has records
    nI = INIT - gI[7]  # INIT, T < 8: the IV
    iN = gI[7] - gI[8]  # INIT, T == 8: the n record
    nN = NS - gN[0]
    evenE = (EARLY - jE[0]) + (jE[1] - jE[2])
    oddE = (jE[0] - jE[1]) + jE[2]
    evenL = (LATE - jL[0]) + (jL[1] - jL[2]) + (jL[3] - jL[4])
    oddL = (jL[0] - jL[1]) + (jL[2] - jL[3]) + jL[4]
    more = one - nb[63]  # NB < 64
    kB = c.and_([BYTES, ended, more], "k.B")
    kV = c.or_and([[nE, EARLY - jE[2]], [nL, LATE - jL[4]], [nC, CVS - jC[0]],
                   [nI, INIT - jI[0]], [iN, INIT - jI[2]], [nN, NS - jN[2]]], "k.V")
    kF = c.or_and([[nE, jE[2]], [nL, jL[4]], [nC, jC[0]], [nI, jI[0] - jI[1]], [iN, jI[2]],
                   [nN, jN[2]]], "k.F")
    kINb = c.and_([BYTES, one - ended, more], "k.IN")  # the block wants a message byte
    kIN = kINb + (RESM - gR[8])
    kBLK = gI[8] + gR[8] + gN[0]
    kR0 = c.and_([BYTES, nb[63]], "k.R0")
    kRN = c.and_([gC[7], one - last], "k.RN")
    kHALT = c.and_([gC[7], last], "k.HALT")
    if bridge:
        # ---------------------------------------------------------- the bridge: capture
        big = c.ffn([(1, one, cnt - MAX_NIBBLES), (-1, one, cnt - MAX_NIBBLES - 1)], "cnt>max")
        th2 = {m: c.ffn([(1, one, cnt - m), (-1, one, cnt - m - 1)], f"cnt>{m}")
               for m in range(1, MAX_NIBBLES + 2 * SLOTS + 2, 2)}
        nq = one - par

        def neq(v):  # [the token has exactly v nibbles]
            return c.nge[v] - c.nge[v + 1] if v < KMAX else c.nge[v]

        # slot i is the i-th byte from the end that this token completes. Its low nibble is
        # nibble 2i (count even) or 2i+1 (count odd) from the token's end, its high nibble the
        # next one, or the nibble left waiting by the previous hex token.
        mbits, mvalid = [Expr() for _ in range(8)], Expr()
        mem_q = {0: one - nb[0], **{a: eqNB[a] for a in range(1, MAX_NIBBLES // 2)}}
        for i in range(SLOTS):
            lo = [c.or_and([[nq, c.rn[2 * i][b]], [par, c.rn[2 * i + 1][b]]], f"cap{i}.lo{b}") for b in range(4)]
            hi = []
            for b in range(4):
                cases = [[nq, c.rn[2 * i + 1][b]], [nq, neq(2 * i + 1), prev[b]], [par, neq(2 * i + 2), prev[b]]]
                if 2 * i + 2 < KMAX:
                    cases.append([par, c.rn[2 * i + 2][b]])
                hi.append(c.or_and(cases, f"cap{i}.hi{b}"))
            valid = c.or_and([[nq, c.nge[2 * i + 1]], [par, c.nge[2 * i + 2]]], f"cap{i}.valid")
            keys = {a: th2[2 * (a + i) + 1] - th2[2 * (a + i) + 3] for a in range(MAX_NIBBLES // 2)}
            data = {**{2 + b: lo[b] for b in range(4)}, **{6 + b: hi[b] for b in range(4)}, 10: one}
            (outs,) = c.table(f"MEM{i}", keys, mem_q, [
                {"data": data, "ref": one, "beta": valid, "a": 238 * c.isPARAM, "gate": kINb}])
            mbits = [mbits[b] + outs[b] for b in range(8)]
            mvalid = mvalid + outs[8]
        # ---------------------------------------------------------- the bridge: control
        fire = c.and_([c.isTCC, fnok, armok, nq, one - big, one - ovf], "fire")
        ((br,),) = c.table("BR", {0: one}, {0: one}, [
            {"data": {2: Chan(fire, LAG1)}, "ref": one, "beta": isSHA + isRES}])
        kMB = c.and_([kINb, br, mvalid], "k.MB")  # the next message byte, from MEM
        kENDb = c.and_([kINb, br, one - mvalid], "k.END")  # the message is over
        kIN = c.and_([kINb, one - br], "k.INhost") + (RESM - gR[8])
        kPR0 = c.and_([kHALT, br], "k.PR0")  # the digest is complete: print instead of HALT
        kHALT = c.and_([kHALT, one - br], "k.HALThost")
        ((printing,),) = c.table("PRN", {0: one}, {0: one}, [{"data": {2: kPR0}, "ref": one, "beta": chipt}])
    kinds = {"IN": kIN, "BLK": kBLK, "R0": kR0, "R16": gE[15], "RC": gL[47], "RN": kRN,
             "HALT": kHALT}
    kcode = {n: S.CONTROLS[n] - S.CTRL_BASE for n in kinds}
    if bridge:
        kinds.update({"END": kENDb, "SHA": fire, "PR0": kPR0})
        kcode.update({"END": S.END - S.CTRL_BASE, "SHA": S.SHA - S.CTRL_BASE, "PR0": S.PRINT_CODE[PRINT_HEAD[0]]})
    sHi = c.or_and([[nE, evenE], [nL, evenL], [nC, CVS - jC[0]], [nI, INIT - jI[0]]], "s.hi")
    sLo = c.or_and([[nE, oddE], [nL, oddL], [nC, jC[0]], [nI, jI[0] - jI[1]]], "s.lo") + nN + kB

    # ------------------------------------------------------------ L3 attn: LEN
    nbm1 = [esum(eqNB[v] for v in range(1, 65) if (v - 1) >> b & 1) for b in range(6)]
    len_bits = {2 + i: (nbm1[i - 3] if i < 9 else N[i - 9]) for i in range(3, 64)}
    ((*lenb,),) = c.table("LEN", {0: one}, {0: one}, [
        {"data": len_bits, "ref": one, "beta": isEND,
         "gates": {2 + 8 * j + b: last + eqNB[63 - j] - 1 for j in range(8) for b in range(8)
                   if 2 + 8 * j + b in len_bits},
         "outs": [(2 + 8 * j + b, OUT_W, f"lenb[{b}]") for b in range(8) for j in range(8)
                  if 2 + 8 * j + b in len_bits]}])
    assert len(lenb) == 8

    # ------------------------------------------------------------ L3 mlp: adder stage 1
    kearly = [esum(nib(S.K[t], k) * eqE[t] for t in range(16)) for k in range(8)]
    iv = [esum(nib(S.IV[i], k) * eqI[i] for i in range(8)) for k in range(8)]
    S_ = []
    for k in range(8):
        rows = [D, H, Wt, W716, E3]
        s = T0[k] + esum(from_bits(r[4 * k:4 * k + 4]) for r in rows) + kearly[k] + iv[k]
        if k < 2:
            s = s + from_bits(lenb[4 * k:4 * k + 4])
        if k == 0:
            s = s + selN[0]  # the + 1 of N + 1
        S_.append(s)
    v = [c.ffn([t for m in range(1, 7) for t in ((1, one, s - 16 * m + 1), (-1, one, s - 16 * m))],
               f"carry[{k}]") for k, s in enumerate(S_)]

    # ------------------------------------------------------------ L4 mlp: adder stage 2
    G, P, Dg = [], [], []
    for k in range(8):
        s = S_[k] - 16 * v[k] + (v[k - 1] if k else 0)  # digit + carry in: 0..21
        G.append(c.ffn(c.t_lut(s, 0, 21, lambda z: int(z >= 16)), f"gen[{k}]"))
        P.append(c.ffn(c.t_lut(s, 0, 21, lambda z: int(z == 15)), f"prop[{k}]"))
        Dg.append([c.ffn(c.t_lut(s, 0, 21, lambda z, b=b: z % 16 >> b & 1), f"digit[{k}][{b}]")
                   for b in range(4)])

    # ------------------------------------------------------------ L5 mlp: stage 3 + exit
    def result_bit(k, b, sel, coef):
        """coef * sel * (bit b of nibble k of the sum): the digit bit, flipped when a carry
        enters the nibble and the digit's lower bits are all set. sel: a literal or a list."""
        d, sel = Dg[k], sel if isinstance(sel, list) else [sel]
        terms = c.t_and(sel + [d[b]], coef)
        for j in range(k):
            src = sel + [G[j]] + P[j + 1:k] + d[:b]
            terms += c.t_and(src + [one - d[b]], coef) + c.t_and(src + [d[b]], -coef)
        return terms

    kctl = esum(kinds.values())
    fam = [kV + kctl, kF + kctl]
    owned = chipt + fire if bridge else one  # where the machine's own code is written
    if bridge:
        # while it prints: the text token at count k (1..73) is followed by PRINT[k]
        seq = PRINT_HEAD + [None] * 64 + PRINT_TAIL
        rng = thP[0] - thP[N_PRINT - 1]
        pbit = {}  # code bit -> [count at which the printed token has it]
        for k in range(1, N_PRINT):
            p = S.HEX_CODE if seq[k] is None else S.PRINT_CODE[seq[k]]
            for b in range(S.NPAY):
                if p >> b & 1:
                    pbit[b] = pbit.get(b, Expr()) + (thP[k - 1] - thP[k])
        seln = [esum(thP[2 + 8 * w + 7 - n] - thP[3 + 8 * w + 7 - n] for w in range(8)) for n in range(8)]
    for i in range(NCODE):
        if i < S.NFAM:
            terms = [(1, 2 * KAPPA * fam[i] - KAPPA * owned, one)]
            if bridge:
                terms += c.t_and([printing, rng], KAPPA)
        else:
            b = i - S.NFAM
            ctl_bit = esum(e for n, e in kinds.items() if kcode[n] >> b & 1)
            terms = [(1, 2 * KAPPA * ctl_bit - KAPPA * owned, one)]
            terms += result_bit(b // 4, b % 4, sLo, 2 * KAPPA)
            terms += result_bit(4 + b // 4, b % 4, sHi, 2 * KAPPA)
            if bridge:
                terms += c.t_and([printing, rng], -KAPPA)
                if b in pbit:
                    terms += c.t_and([printing, pbit[b]], 2 * KAPPA)
                if b < 4:  # a hex digit of the digest: nibble n of the word the adder passes through
                    for n in range(8):
                        terms += result_bit(n, b, [printing, seln[n]], 2 * KAPPA)
                if b < 8:  # a message byte from MEM
                    terms += c.t_and([kMB, mbits[b]], 2 * KAPPA)
        c.ffn(terms, f"exit.code{i}", into=c.code[i])
    c.keep = [next(iter(e.terms)) for e in c.code]
    return c


def predictions(c: Sha, m: np.ndarray) -> list:
    """The token each position predicts, read from the code dims the exit leaves. None: the
    code is 0 (a merged circuit leaves the position to the language model) or names no token.
    A printed text token q comes back as TEXT0 + q."""
    code = m[:, c.keep]
    bits = code > 0
    fam = bits[:, 0] * 1 + bits[:, 1] * 2
    payload = (bits[:, S.NFAM:] * (1 << np.arange(S.NPAY))).sum(1)
    out = []
    for own, f, p in zip((code != 0).any(1), fam, payload):
        t = S.token(int(f), int(p)) if own else None
        if own and t is None and c.text is not None and int(f) == 3 and int(p) in c.text.token_of_code:
            t = S.TEXT0 + c.text.token_of_code[int(p)]
        out.append(t)
    return out
