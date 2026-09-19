"""Fastest measured CUDA-graph backward; 256-target checkpoints."""

from __future__ import annotations
import math
import torch
from torch.autograd.function import once_differentiable
from ._common import triton, tl

from . import _forward as fw
from ._common import allocate_state as _allocate_state, check_long

if triton is not None:
    from ._common import (
        _prefix_combine,
        _probability_contract,
        _context_dot,
        _score_adjoint_contract,
        _sum_combine,
    )
    from ._forward import _training_reduce

    @triton.jit(do_not_specialize=["START", "END"])
    def _backward_outer(
        QF,
        KF,
        QC,
        KC,
        VC,
        GY,
        Y32,
        LC,
        ZK,
        ZV,
        M,
        L,
        C,
        DB,
        DBO,
        CO,
        CK,
        CV,
        GA,
        PE,
        GELL,
        START,
        END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
        ADAPTIVE: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        s = tile * BS + tl.arange(0, BS)
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        state = bh * T + s
        qf = tl.load(QF + sp, sm, 0)
        kf = tl.load(KF + tp, tm, 0)
        scores = tl.where(j[None, :] <= END, tl.dot(qf, tl.trans(kf)) * SCALE, -float("inf"))
        m0 = tl.load(M + state, s < T, -float("inf"))
        l0 = tl.load(L + state, s < T, 0)
        final_m = tl.maximum(m0, tl.max(scores, 1))
        old_mass = tl.where(l0 > 0, l0 * tl.exp(m0 - final_m), 0.0)
        e = tl.where(j[None, :] <= END, tl.exp(scores - final_m[:, None]), 0.0)
        end_den = old_mass + tl.sum(e, 1)
        pe = e / end_den[:, None]
        # Scratch matrices are source-major for GA/PE, target-major otherwise.
        tiles: tl.constexpr = triton.cdiv(T, BS)
        scalar = (bh * tiles + tile) * BT * BS
        si = scalar + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ti = scalar + r[:, None] * BS + tl.arange(0, BS)[None, :]
        ci = (bh * tiles + tile) * BT * BT + r[:, None] * BT + r[None, :]
        tl.store(PE + si, pe)
        if tile * BS <= END:
            qc = tl.load(QC + tp, tm, 0)
            gy = tl.load(GY + tp, tm, 0)
            kc = tl.load(KC + tp, tm, 0)
            vc = tl.load(VC + tp, tm, 0)
            qg = tl.dot(qc, tl.trans(kc)) * SCALE
            hv = tl.dot(gy, tl.trans(vc))
            lower = r[:, None] >= r[None, :]
            common = False
            if ADAPTIVE:
                first = tl.reshape(tl.gather(scores, tl.full((BS, 1), 0, tl.int32), 1), (BS,))
                common = tl.max(tl.where(s < T, final_m - tl.maximum(m0, first), 0.0), 0) <= 32.0
            pm = tl.full((BS, BT), 0.0, tl.float32)
            il = tl.full((BS, BT), 0.0, tl.float32)
            if common:
                den = old_mass[:, None] + tl.cumsum(e, 1)
                il = 1.0 / den
                oldp = tl.trans(old_mass[:, None] * il)
                newk = _context_dot(tl.where(lower, qg, 0.0), tl.trans(e), FAST) * tl.trans(il)
                newv = _context_dot(tl.where(lower, hv, 0.0), tl.trans(e), FAST) * tl.trans(il)
            else:
                leaf = tl.broadcast_to((j[None, :] <= END).to(tl.float32), (BS, BT))
                lm, ll = tl.associative_scan((scores, leaf), 1, _prefix_combine)
                pm = tl.maximum(lm, m0[:, None])
                den = ll * tl.exp(lm - pm) + tl.where(
                    l0[:, None] > 0, l0[:, None] * tl.exp(m0[:, None] - pm), 0.0
                )
                il = 1.0 / den
                oldp = tl.trans(
                    tl.where(l0[:, None] > 0, l0[:, None] * tl.exp(m0[:, None] - pm) * il, 0.0)
                )
                newk = _probability_contract(scores, pm, il, qg, BS, BT, 4, False)
                newv = _probability_contract(scores, pm, il, hv, BS, BT, 4, False)
            zk = tl.load(ZK + state[:, None] * D + d[None, :], sm, 0)
            qold = _context_dot(qc, tl.trans(zk), FAST) * SCALE
            kcontext = newk + qold * oldp
            ksource = tl.load(KC + sp, sm, 0)
            logits = kcontext + tl.dot(qc, tl.trans(ksource)) * SCALE
            zv = tl.load(ZV + state[:, None] * D + d[None, :], sm, 0)
            hold = _context_dot(gy, tl.trans(zv), FAST)
            vcontext = newv + hold * oldp
            vsource = tl.load(VC + sp, sm, 0)
            hsource = tl.dot(gy, tl.trans(vsource))
            lc = tl.load(LC + bh * T + j, j <= END, 0)
            active = (j[:, None] <= END) & (s[None, :] < T) & (s[None, :] <= j[:, None])
            c = tl.where(active, tl.exp(logits - lc[:, None]), 0.0)
            y = tl.load(Y32 + tp, tm, 0)
            delta = tl.sum(gy.to(tl.float32) * y, 1)
            db = c * (hsource + vcontext - delta[:, None])
            mean = db * kcontext + c * vcontext
            go = tl.sum(tl.trans(oldp * (db * qold + c * hold - mean)), 1)
            if common:
                rk = db * tl.trans(il)
                rv = c * tl.trans(il)
                ck = tl.where(lower, _context_dot(rk, e, FAST), 0.0)
                cv = tl.where(lower, _context_dot(rv, e, FAST), 0.0)
                ga = _context_dot(tl.trans(rk), tl.where(lower, qg, 0.0), FAST)
                ga += _context_dot(tl.trans(rv), tl.where(lower, hv, 0.0), FAST)
                ga -= tl.associative_scan(
                    tl.trans(mean * tl.trans(il)), 1, _sum_combine, reverse=True
                )
                ga *= e
            else:
                ck = _probability_contract(scores, pm, il, db, BS, BT, 4, True)
                cv = _probability_contract(scores, pm, il, c, BS, BT, 4, True)
                ga = _score_adjoint_contract(scores, pm, il, db, c, qg, hv, mean, BS, BT, 4)
            tl.store(C + ti, c)
            tl.store(DB + ti, db)
            tl.store(DBO + ti, db * oldp)
            tl.store(CO + ti, c * oldp)
            tl.store(CK + ci, ck)
            tl.store(CV + ci, cv)
            tl.store(GA + si, ga)
            tl.store(GELL + state, go, s < T)
        else:
            tl.store(C + ti, 0.0)
            tl.store(DB + ti, 0.0)
            tl.store(DBO + ti, 0.0)
            tl.store(CO + ti, 0.0)
            tl.store(CK + ci, 0.0)
            tl.store(CV + ci, 0.0)
            tl.store(GA + si, 0.0)
            tl.store(GELL + state, 0.0, s < T)

    @triton.jit(do_not_specialize=["START", "END"])
    def _backward_future_scores(
        KC,
        VC,
        ZK0,
        ZV0,
        ZK1,
        ZV1,
        GZK,
        GZV,
        FGA,
        FDOT,
        START,
        END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
        FAST: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2)
        b = bh // H
        h = bh % H
        s = tile * BS + tl.arange(0, BS)
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        state = bh * T + s
        tiles: tl.constexpr = triton.cdiv(T, BS)
        if kind == 0:
            gz = GZK
            old = ZK0
            new = ZK1
            values = KC
        else:
            gz = GZV
            old = ZV0
            new = ZV1
            values = VC
        g = tl.load(gz + state[:, None] * D + d[None, :], sm, 0)
        z1 = tl.load(new + state[:, None] * D + d[None, :], sm, 0)
        dot1 = tl.sum(g * z1, 1)
        z0 = tl.load(old + state[:, None] * D + d[None, :], sm, 0)
        dot0 = tl.sum(g * z0, 1)
        v = tl.load(values + ((b * T + j[:, None]) * H + h) * D + d[None, :], tm, 0)
        score = _context_dot(g, tl.trans(v), FAST)
        plane: tl.constexpr = BH * tiles * BS * BT
        ix = (bh * tiles + tile) * BS * BT + tl.arange(0, BS)[:, None] * BT + r[None, :]
        tl.store(FGA + kind * plane + ix, score)
        tl.store(FDOT + (kind * 2) * BH * T + state, dot1, s < T)
        tl.store(FDOT + (kind * 2 + 1) * BH * T + state, dot0, s < T)

    @triton.jit(do_not_specialize=["START", "END"])
    def _backward_future_reduce(
        M0,
        L0,
        M1,
        L1,
        GELL,
        GELLO,
        PE,
        GA,
        FGA,
        FDOT,
        START,
        END,
        T: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        s = tile * BS + tl.arange(0, BS)
        r = tl.arange(0, BT)
        state = bh * T + s
        tiles: tl.constexpr = triton.cdiv(T, BS)
        ix = (bh * tiles + tile) * BS * BT + tl.arange(0, BS)[:, None] * BT + r[None, :]
        plane: tl.constexpr = BH * tiles * BS * BT
        k1 = tl.load(FDOT + state, s < T, 0)
        k0 = tl.load(FDOT + BH * T + state, s < T, 0)
        v1 = tl.load(FDOT + 2 * BH * T + state, s < T, 0)
        v0 = tl.load(FDOT + 3 * BH * T + state, s < T, 0)
        bias = tl.load(GELL + state, s < T, 0) - k1 - v1
        score = tl.load(FGA + ix) + tl.load(FGA + plane + ix)
        ga = tl.load(GA + ix) + tl.load(PE + ix) * (score + bias[:, None])
        tl.store(GA + ix, ga)
        m0 = tl.load(M0 + state, s < T, -float("inf"))
        l0 = tl.load(L0 + state, s < T, 0)
        m1 = tl.load(M1 + state, s < T, 0)
        l1 = tl.load(L1 + state, s < T, 1)
        oldp = tl.where(l0 > 0, l0 * tl.exp(m0 - m1) / l1, 0.0)
        outer = tl.load(GELLO + state, s < T, 0)
        tl.store(GELL + state, outer + oldp * (bias + k0 + v0), s < T)

    @triton.jit(do_not_specialize=["START", "END"])
    def _backward_future_values(
        PE,
        GZK,
        GZV,
        FK,
        FV,
        START,
        END,
        T: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        FAST: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2)
        s = tile * BS + tl.arange(0, BS)
        r = tl.arange(0, BT)
        d = tl.arange(0, BD)
        tiles: tl.constexpr = triton.cdiv(T, BS)
        ix = (bh * tiles + tile) * BT * BS + tl.arange(0, BS)[:, None] * BT + r[None, :]
        p = tl.trans(tl.load(PE + ix))
        if kind == 0:
            gz = GZK
            dest = FK
        else:
            gz = GZV
            dest = FV
        g = tl.load(
            gz + (bh * T + s[:, None]) * D + d[None, :], (s[:, None] < T) & (d[None, :] < D), 0
        )
        value = _context_dot(p, g, FAST)
        out = ((bh * tiles + tile) * BT + r[:, None]) * D + d[None, :]
        tl.store(dest + out, value, (START + r[:, None] <= END) & (d[None, :] < D))

    @triton.jit(do_not_specialize=["START", "END"])
    def _backward_source(
        QF,
        KF,
        QC,
        GY,
        GA,
        C,
        DB,
        DBO,
        CO,
        ZK0,
        ZV0,
        M0,
        L0,
        M1,
        L1,
        GZK,
        GZV,
        GQF,
        GKC,
        GVC,
        GELL,
        START,
        END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
        SKIP: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2)
        b = bh // H
        h = bh % H
        s = tile * BS + tl.arange(0, BS)
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        tiles: tl.constexpr = triton.cdiv(T, BS)
        state = bh * T + s
        sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        si = (bh * tiles + tile) * BT * BS + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ti = (bh * tiles + tile) * BT * BS + r[:, None] * BS + tl.arange(0, BS)[None, :]
        if kind == 0:
            a = tl.load(GA + si)
            k = tl.load(KF + tp, tm, 0)
            end_ptr = ((b * T + END) * H + h) * D + d
            prev_ptr = ((b * T + START - 1) * H + h) * D + d
            anchor = tl.load(KF + end_ptr, d < D, 0).to(tl.float32)
            previous = tl.load(KF + prev_ptr, (START > 0) & (d < D), 0).to(tl.float32)
            # GQF stores accumulated_dq + gell*the current boundary's KF anchor.
            # sum(GA_current) = gell_next - gell_previous. Centering uses this
            # analytic identity instead of cancellation among large KF scores.
            centered = k.to(tl.float32) - anchor[None, :]
            ge = tl.load(GELL + state, s < T, 0)
            update = (
                _context_dot(a, centered, FAST) + ge[:, None] * (previous - anchor)[None, :]
            ) * SCALE
            old = tl.load(GQF + sp, sm, 0)
            tl.store(GQF + sp, old + update, sm)
        elif kind == 1:
            if not SKIP or tile * BS <= END:
                a = tl.trans(tl.load(DB + ti))
                q = tl.load(QC + tp, tm, 0)
                update = _context_dot(a, q, FAST) * SCALE
                old = tl.load(GKC + sp, sm, 0)
                tl.store(GKC + sp, old + update, sm)
        elif kind == 2:
            if not SKIP or tile * BS <= END:
                a = tl.trans(tl.load(C + ti))
                g = tl.load(GY + tp, tm, 0)
                update = _context_dot(a, g, FAST)
                old = tl.load(GVC + sp, sm, 0)
                tl.store(GVC + sp, old + update, sm)
        else:
            m0 = tl.load(M0 + state, s < T, -float("inf"))
            l0 = tl.load(L0 + state, s < T, 0)
            m1 = tl.load(M1 + state, s < T, 0)
            l1 = tl.load(L1 + state, s < T, 1)
            p = tl.where(l0 > 0, l0 * tl.exp(m0 - m1) / l1, 0.0)
            if kind == 3:
                a = tl.trans(tl.load(DBO + ti))
                q = tl.load(QC + tp, tm, 0)
                update = _context_dot(a, q, FAST) * SCALE
                old = tl.load(GZK + state[:, None] * D + d[None, :], sm, 0)
                tl.store(GZK + state[:, None] * D + d[None, :], update + p[:, None] * old, sm)
            else:
                a = tl.trans(tl.load(CO + ti))
                g = tl.load(GY + tp, tm, 0)
                update = _context_dot(a, g, FAST)
                old = tl.load(GZV + state[:, None] * D + d[None, :], sm, 0)
                tl.store(GZV + state[:, None] * D + d[None, :], update + p[:, None] * old, sm)

    @triton.jit(do_not_specialize=["START", "END"])
    def _backward_target(
        QF,
        QC,
        KC,
        GY,
        ZK0,
        GA,
        DB,
        DBO,
        CK,
        CV,
        FK,
        FV,
        PART,
        START,
        END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
        SKIP: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2)
        b = bh // H
        h = bh % H
        s = tile * BS + tl.arange(0, BS)
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        state = bh * T + s
        sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        tiles: tl.constexpr = triton.cdiv(T, BS)
        si = (bh * tiles + tile) * BT * BS + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ti = (bh * tiles + tile) * BT * BS + r[:, None] * BS + tl.arange(0, BS)[None, :]
        ci = (bh * tiles + tile) * BT * BT + r[:, None] * BT + r[None, :]
        pi = (bh * tiles + tile) * BT * D + r[:, None] * D + d[None, :]
        if kind == 0:
            a = tl.trans(tl.load(GA + si))
            q = tl.load(QF + sp, sm, 0)
            update = _context_dot(a, q, FAST) * SCALE
        elif kind == 1:
            if not SKIP or tile * BS <= END:
                a = tl.load(DB + ti)
                k = tl.load(KC + sp, sm, 0)
                update = _context_dot(a, k, FAST)
                a = tl.load(DBO + ti)
                z = tl.load(ZK0 + state[:, None] * D + d[None, :], sm, 0)
                update += _context_dot(a, z, FAST)
                ck = tl.load(CK + ci)
                kc = tl.load(KC + tp, tm, 0)
                update += _context_dot(ck, kc, FAST)
                update *= SCALE
            else:
                update = tl.full((BT, BD), 0.0, tl.float32)
        elif kind == 2:
            ck = tl.trans(tl.load(CK + ci))
            q = tl.load(QC + tp, tm, 0)
            update = _context_dot(ck, q, FAST) * SCALE + tl.load(FK + pi, tm, 0)
        else:
            cv = tl.trans(tl.load(CV + ci))
            g = tl.load(GY + tp, tm, 0)
            update = _context_dot(cv, g, FAST) + tl.load(FV + pi, tm, 0)
        plane: tl.constexpr = BH * tiles * BT * D
        tl.store(PART + kind * plane + pi, update, tm)

    @triton.jit(do_not_specialize=["START"])
    def _backward_reduce(
        PART,
        GKF,
        GQC,
        GKC,
        GVC,
        START,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
        BP: tl.constexpr,
    ):
        t = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2)
        b = bh // H
        h = bh % H
        d = tl.arange(0, BD)
        tiles: tl.constexpr = triton.cdiv(T, BS)
        s = tl.arange(0, BP)
        idx = ((bh * tiles + s[:, None]) * BT + t) * D + d[None, :]
        plane: tl.constexpr = BH * tiles * BT * D
        p = tl.load(PART + kind * plane + idx, (s[:, None] < tiles) & (d[None, :] < D), 0)
        value = tl.sum(p, 0)
        out = ((b * T + START + t) * H + h) * D + d
        if kind == 0:
            dest = GKF
        elif kind == 1:
            dest = GQC
        elif kind == 2:
            dest = GKC
        else:
            dest = GVC
        old = tl.load(dest + out, d < D, 0)
        tl.store(dest + out, old + value, d < D)

    @triton.jit
    def _snapshot(
        ZK, ZV, M, L, OK, OV, OM, OL, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N * D
        tl.store(OK + i, tl.load(ZK + i, mask, 0), mask)
        tl.store(OV + i, tl.load(ZV + i, mask, 0), mask)
        valid = i < N
        tl.store(OM + i, tl.load(M + i, valid, 0), valid)
        tl.store(OL + i, tl.load(L + i, valid, 0), valid)

    @triton.jit(do_not_specialize=["START", "END"])
    def _replay(
        QF,
        KF,
        KC,
        VC,
        ZK,
        ZV,
        M,
        L,
        OK,
        OV,
        OM,
        OL,
        START,
        END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        s = tile * BS + tl.arange(0, BS)
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        state = bh * T + s
        q = tl.load(QF + sp, sm, 0)
        k = tl.load(KF + tp, tm, 0)
        scores = tl.where(j[None, :] <= END, tl.dot(q, tl.trans(k)) * SCALE, -float("inf"))
        m = tl.load(M + state, s < T, -float("inf"))
        l = tl.load(L + state, s < T, 0)
        maximum = tl.maximum(m, tl.max(scores, 1))
        old = tl.where(l > 0, l * tl.exp(m - maximum), 0.0)
        e = tl.where(j[None, :] <= END, tl.exp(scores - maximum[:, None]), 0.0)
        den = old + tl.sum(e, 1)
        zk = tl.load(ZK + state[:, None] * D + d[None, :], sm, 0)
        kc = tl.load(KC + tp, tm, 0)
        new = (old[:, None] * zk + _context_dot(e, kc, FAST)) / den[:, None]
        tl.store(OK + state[:, None] * D + d[None, :], new, sm)
        zv = tl.load(ZV + state[:, None] * D + d[None, :], sm, 0)
        vc = tl.load(VC + tp, tm, 0)
        new = (old[:, None] * zv + _context_dot(e, vc, FAST)) / den[:, None]
        tl.store(OV + state[:, None] * D + d[None, :], new, sm)
        tl.store(OM + state, maximum, s < T)
        tl.store(OL + state, den, s < T)


def _forward_run(inputs, checkpoint=256, fast=True, adaptive=True):
    # Custom Function forward runs without grad enabled.
    b, t, h, d = check_long(inputs)
    if checkpoint < 16 or checkpoint % 16:
        raise ValueError("checkpoint must be a multiple of 16")
    qf, kf, qc, kc, vc = inputs
    bh = b * h
    bs, bt = 32, 16
    bd = max(32, triton.next_power_of_2(d))
    tiles = triton.cdiv(t, bs)
    chunks = triton.cdiv(t, checkpoint)
    checks = _allocate_state(b, h, t, d, qf.device, chunks + 1)
    state = _allocate_state(b, h, t, d, qf.device, 1)
    state[0].zero_()
    state[1].zero_()
    state[2].fill_(-float("inf"))
    state[3].zero_()
    for src, dst in zip(state, checks):
        dst[0].copy_(src[0])
    zk, zv, m, l = (x[0] for x in state)
    pa = torch.empty((bh, tiles, bt, d), device=qf.device, dtype=torch.float32)
    pm = torch.empty((bh, tiles, bt), device=qf.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    y = torch.empty_like(qf)
    y32 = torch.empty_like(qf, dtype=torch.float32)
    lc = torch.empty((bh, t), device=qf.device, dtype=torch.float32)
    for start in range(0, t, bt):
        end = min(start + bt, t) - 1
        k = fw._two_geometry_factorized[(tiles, h, b)](
            qf,
            kf,
            vc,
            kc,
            qc,
            zk,
            zv,
            m,
            l,
            pa,
            pm,
            pl,
            start,
            end,
            t,
            h,
            d,
            bd,
            bs,
            bs,
            bt,
            bt,
            tiles,
            1 / math.sqrt(d),
            4,
            False,
            fast,
            adaptive,
            num_warps=4,
            num_stages=3,
        )
        k = _training_reduce[(end - start + 1, bh)](
            pa,
            pm,
            pl,
            y,
            y32,
            lc,
            start,
            end,
            t,
            h,
            d,
            bd,
            bt,
            tiles,
            triton.next_power_of_2(tiles),
            num_warps=4,
        )
        if (end + 1) % checkpoint == 0 or end == t - 1:
            index = triton.cdiv(end + 1, checkpoint)
            k = _snapshot[(triton.cdiv(bh * t * d, 512),)](
                zk, zv, m, l, *(x[index] for x in checks), bh * t, d, 512
            )
    return y, checks, y32, lc


def _backward_run(
    inputs, checks, y32, lc, gy, checkpoint=256, fast=True, adaptive=True, skip_inactive=True
):
    qf, kf, qc, kc, vc = inputs
    b, t, h, d = qf.shape
    bh = b * h
    bs, bt = 32, 16
    bd = max(32, triton.next_power_of_2(d))
    tiles = triton.cdiv(t, bs)
    levels = checkpoint // bt + 1
    hist = _allocate_state(b, h, t, d, qf.device, levels)
    gzk = torch.zeros((bh, t, d), device=qf.device, dtype=torch.float32)
    gzv = torch.zeros_like(gzk)
    ge = torch.zeros((bh, t), device=qf.device, dtype=torch.float32)
    geo = torch.empty_like(ge)
    grads = [torch.zeros_like(qf, dtype=torch.float32) for _ in range(5)]
    gqf, gkf, gqc, gkc, gvc = grads
    c, db, dbo, co, ga, pe = [
        torch.empty((bh, tiles, bt, bs), device=qf.device, dtype=torch.float32) for _ in range(6)
    ]
    ck = torch.empty((bh, tiles, bt, bt), device=qf.device, dtype=torch.float32)
    cv = torch.empty_like(ck)
    fk = torch.empty((bh, tiles, bt, d), device=qf.device, dtype=torch.float32)
    fv = torch.empty_like(fk)
    fga = torch.empty((2, bh, tiles, bs, bt), device=qf.device, dtype=torch.float32)
    fdot = torch.empty((4, bh, t), device=qf.device, dtype=torch.float32)
    part = torch.empty((4, bh, tiles, bt, d), device=qf.device, dtype=torch.float32)
    gy = gy.contiguous()
    for chunk in reversed(range(triton.cdiv(t, checkpoint))):
        start_chunk = chunk * checkpoint
        end_chunk = min(start_chunk + checkpoint, t)
        for dst, src in zip(hist, checks):
            dst[0].copy_(src[chunk])
        n = triton.cdiv(end_chunk - start_chunk, bt)
        for i in range(n):
            start = start_chunk + i * bt
            end = min(start + bt, end_chunk) - 1
            k = _replay[(tiles, bh)](
                qf,
                kf,
                kc,
                vc,
                *(x[i] for x in hist),
                *(x[i + 1] for x in hist),
                start,
                end,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                1 / math.sqrt(d),
                fast,
                num_warps=4,
                num_stages=3,
            )
        for i in reversed(range(n)):
            start = start_chunk + i * bt
            end = min(start + bt, end_chunk) - 1
            old = tuple(x[i] for x in hist)
            new = tuple(x[i + 1] for x in hist)
            k = _backward_outer[(tiles, bh)](
                qf,
                kf,
                qc,
                kc,
                vc,
                gy,
                y32,
                lc,
                *old,
                c,
                db,
                dbo,
                co,
                ck,
                cv,
                ga,
                pe,
                geo,
                start,
                end,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                1 / math.sqrt(d),
                fast,
                adaptive,
                num_warps=4,
                num_stages=3,
            )
            _backward_future_scores[(tiles, bh, 2)](
                kc,
                vc,
                old[0],
                old[1],
                new[0],
                new[1],
                gzk,
                gzv,
                fga,
                fdot,
                start,
                end,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                bh,
                fast,
                num_warps=4,
                num_stages=3,
            )
            _backward_future_reduce[(tiles, bh)](
                *old[2:],
                *new[2:],
                ge,
                geo,
                pe,
                ga,
                fga,
                fdot,
                start,
                end,
                t,
                bs,
                bt,
                bh,
                num_warps=4,
                num_stages=3,
            )
            _backward_future_values[(tiles, bh, 2)](
                pe, gzk, gzv, fk, fv, start, end, t, d, bd, bs, bt, fast, num_warps=4, num_stages=3
            )
            k = _backward_source[(tiles, bh, 5)](
                qf,
                kf,
                qc,
                gy,
                ga,
                c,
                db,
                dbo,
                co,
                *old,
                *new[2:],
                gzk,
                gzv,
                gqf,
                gkc,
                gvc,
                ge,
                start,
                end,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                1 / math.sqrt(d),
                fast,
                skip_inactive,
                num_warps=4,
                num_stages=3,
            )
            k = _backward_target[(tiles, bh, 4)](
                qf,
                qc,
                kc,
                gy,
                old[0],
                ga,
                db,
                dbo,
                ck,
                cv,
                fk,
                fv,
                part,
                start,
                end,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                bh,
                1 / math.sqrt(d),
                fast,
                skip_inactive,
                num_warps=4,
                num_stages=3,
            )
            k = _backward_reduce[(end - start + 1, bh, 4)](
                part,
                gkf,
                gqc,
                gkc,
                gvc,
                start,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                bh,
                triton.next_power_of_2(tiles),
                num_warps=4,
            )
    return tuple(grads)


class _TwoGeometryLong(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qf, kf, qc, kc, vc, checkpoint, fast, adaptive, skip):
        inputs = (qf, kf, qc, kc, vc)
        y, checks, y32, lc = _forward_run(inputs, checkpoint, fast, adaptive)
        ctx.save_for_backward(*inputs, *checks, y32, lc)
        ctx.options = checkpoint, fast, adaptive, skip
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, gy):
        saved = ctx.saved_tensors
        grads = _backward_run(saved[:5], saved[5:9], saved[9], saved[10], gy, *ctx.options)
        return tuple(g.to(x.dtype) for g, x in zip(grads, saved[:5])) + (None, None, None, None)


def run(inputs):
    check_long(inputs)
    return _TwoGeometryLong.apply(*inputs, 256, True, True, True)
