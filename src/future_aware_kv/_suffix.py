"""Fastest measured eager backward; bounded source-major prefix/suffix replay."""

from __future__ import annotations
import math
import torch
from torch.autograd.function import once_differentiable
from ._common import triton, tl

from ._common import check_long, allocate_state
from . import _forward as fw

if triton is not None:
    from ._common import (
        _prefix_combine,
        _probability_contract,
        _context_dot,
        _score_adjoint_contract,
        _sum_combine,
    )
    from ._forward import _training_reduce

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _prefix_state_from_coefficients(
        KC,
        VC,
        ZK,
        ZV,
        L,
        E,
        OLD,
        START,
        END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        FAST: tl.constexpr,
        SOURCE_START,
        W: tl.constexpr,
        KIND: tl.constexpr = -1,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2) if KIND < 0 else KIND
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        state = bh * W + ls
        si = (bh * tiles + tile) * BS * BT + tl.arange(0, BS)[:, None] * BT + r[None, :]
        e = tl.load(E + si)
        old = tl.load(OLD + state, s < T, 0)
        den = tl.load(L + state, s < T, 1)
        if kind == 0:
            values = KC
            zptr = ZK
        else:
            values = VC
            zptr = ZV
        tm = (j[:, None] <= END) & (d[None, :] < D)
        sm = (s[:, None] < T) & (d[None, :] < D)
        v = tl.load(values + ((b * T + j[:, None]) * H + h) * D + d[None, :], tm, 0)
        z = tl.load(zptr + state[:, None] * D + d[None, :], sm, 0)
        value = (old[:, None] * z + _context_dot(e, v, FAST)) / den[:, None]
        tl.store(zptr + state[:, None] * D + d[None, :], value, sm)

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _prefix_prepare(
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
        SCALARS,
        NORMS,
        INDEX,
        WB,
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
        SOURCE_START,
        W: tl.constexpr,
        STATE_E,
        STATE_OLD,
        EXPORT_STATE: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        state = bh * W + ls
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
        tiles: tl.constexpr = triton.cdiv(W, BS)
        scalar = (bh * tiles + tile) * BT * BS
        si = scalar + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ti = scalar + r[:, None] * BS + tl.arange(0, BS)[None, :]
        ci = (bh * tiles + tile) * BT * BT + r[:, None] * BT + r[None, :]
        if EXPORT_STATE:
            # Export unnormalized coefficients to preserve replay's FP32
            # arithmetic order rather than quantizing E/den before its dot.
            tl.store(STATE_E + si, e)
            tl.store(STATE_OLD + state, old_mass, s < T)
            tl.store(M + state, final_m, s < T)
            tl.store(L + state, end_den, s < T)
        window_plane = WB * tl.num_programs(1) * tiles * BT * BS
        wi = (
            ((INDEX * tl.num_programs(1) + bh) * tiles + tile) * BT * BS
            + r[:, None] * BS
            + tl.arange(0, BS)[None, :]
        )
        ni = (INDEX * tl.num_programs(1) + bh) * W + ls
        tl.store(NORMS + ni, m0, s < T)
        tl.store(NORMS + WB * tl.num_programs(1) * W + ni, l0, s < T)
        if SOURCE_START + tile * BS <= END:
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
            if common:
                ck = tl.where(lower, _context_dot(db * tl.trans(il), e, FAST), 0.0)
            else:
                ck = _probability_contract(scores, pm, il, db, BS, BT, 4, True)
            tl.store(C + ti, c)
            tl.store(DB + ti, db)
            tl.store(DBO + ti, db * oldp)
            tl.store(CO + ti, c * oldp)
            tl.store(CK + ci, ck)
            tl.store(SCALARS + wi, c)
            tl.store(SCALARS + window_plane + wi, db)
            tl.store(SCALARS + 2 * window_plane + wi, mean)
        else:
            tl.store(C + ti, 0.0)
            tl.store(DB + ti, 0.0)
            tl.store(DBO + ti, 0.0)
            tl.store(CO + ti, 0.0)
            tl.store(CK + ci, 0.0)
            tl.store(SCALARS + wi, 0.0)
            tl.store(SCALARS + window_plane + wi, 0.0)
            tl.store(SCALARS + 2 * window_plane + wi, 0.0)

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _prefix_qc(
        QC,
        KC,
        ZK,
        DB,
        DBO,
        CK,
        PART,
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
        SOURCE_START,
        W: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        ti = (bh * tiles + tile) * BT * BS + r[:, None] * BS + tl.arange(0, BS)[None, :]
        ci = (bh * tiles + tile) * BT * BT + r[:, None] * BT + r[None, :]
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        if SOURCE_START + tile * BS <= END:
            k = tl.load(KC + ((b * T + s[:, None]) * H + h) * D + d[None, :], sm, 0)
            a = tl.load(DB + ti)
            update = _context_dot(a, k, FAST)
            z = tl.load(ZK + (bh * W + ls[:, None]) * D + d[None, :], sm, 0)
            a = tl.load(DBO + ti)
            update += _context_dot(a, z, FAST)
            ck = tl.load(CK + ci)
            k = tl.load(KC + ((b * T + j[:, None]) * H + h) * D + d[None, :], tm, 0)
            update += _context_dot(ck, k, FAST)
            update *= SCALE
        else:
            update = tl.full((BT, BD), 0.0, tl.float32)
        out = ((bh * tiles + tile) * BT + r[:, None]) * D + d[None, :]
        tl.store(PART + out, update, tm)

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _suffix_scalar(
        QF,
        KF,
        SCALARS,
        NORMS,
        C,
        DB,
        DBO,
        CO,
        CK,
        CV,
        GA,
        PE,
        OLDP,
        GO,
        RESET,
        START,
        END,
        INDEX,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        WB: tl.constexpr,
        BH: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
        ADAPTIVE: tl.constexpr,
        SOURCE_START,
        W: tl.constexpr,
        SCORE_SCRATCH,
        CACHE_QK: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        state = bh * W + ls
        sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        q = tl.load(QF + sp, sm, 0)
        k = tl.load(KF + tp, tm, 0)
        scores = tl.where(j[None, :] <= END, tl.dot(q, tl.trans(k)) * SCALE, -float("inf"))
        ni = (INDEX * BH + bh) * W + ls
        m0 = tl.load(NORMS + ni, s < T, -float("inf"))
        l0 = tl.load(NORMS + WB * BH * W + ni, s < T, 0)
        maximum = tl.maximum(m0, tl.max(scores, 1))
        old = tl.where(l0 > 0, l0 * tl.exp(m0 - maximum), 0.0)
        e = tl.where(j[None, :] <= END, tl.exp(scores - maximum[:, None]), 0.0)
        end_den = old + tl.sum(e, 1)
        pe = e / end_den[:, None]
        tl.store(OLDP + state, old / end_den, s < T)
        si = (bh * tiles + tile) * BT * BS + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ti = (bh * tiles + tile) * BT * BS + r[:, None] * BS + tl.arange(0, BS)[None, :]
        ci = (bh * tiles + tile) * BT * BT + r[:, None] * BT + r[None, :]
        wi = (
            ((INDEX * BH + bh) * tiles + tile) * BT * BS
            + r[:, None] * BS
            + tl.arange(0, BS)[None, :]
        )
        wp: tl.constexpr = WB * BH * tiles * BT * BS
        c = tl.load(SCALARS + wi)
        db = tl.load(SCALARS + wp + wi)
        mu = tl.load(SCALARS + 2 * wp + wi)
        if CACHE_QK:
            tl.store(SCORE_SCRATCH + si, scores)
        tl.store(C + ti, c)
        tl.store(DB + ti, db)
        tl.store(PE + si, pe)
        lower = r[:, None] >= r[None, :]
        common = False
        if ADAPTIVE:
            first = tl.reshape(tl.gather(scores, tl.full((BS, 1), 0, tl.int32), 1), (BS,))
            common = tl.max(tl.where(s < T, maximum - tl.maximum(m0, first), 0.0), 0) <= 32.0
        pm = tl.full((BS, BT), 0.0, tl.float32)
        if common:
            den = old[:, None] + tl.cumsum(e, 1)
            il = 1.0 / den
            oldp = tl.trans(old[:, None] * il)
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
        ix = tl.broadcast_to(tl.maximum(r - 1, 0)[None, :], (BS, BT))
        prev_l = tl.gather(den, ix, 1)
        if common:
            rho = tl.where(r[None, :] > 0, prev_l, old[:, None]) * il
        else:
            prev_m = tl.gather(pm, ix, 1)
            prev_l = tl.where(r[None, :] > 0, prev_l, l0[:, None])
            prev_m = tl.where(r[None, :] > 0, prev_m, m0[:, None])
            rho = tl.where(prev_l > 0, prev_l * tl.exp(prev_m - pm) * il, 0.0)
        tl.store(RESET + si, (rho == 0) & (j[None, :] <= END))
        tl.store(DBO + ti, db * oldp)
        tl.store(CO + ti, c * oldp)
        tl.store(GO + state, tl.sum(tl.trans(oldp * mu), 1), s < T)

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _suffix_contractions(
        QF,
        KF,
        QC,
        KC,
        VC,
        GY,
        SCALARS,
        NORMS,
        CK,
        CV,
        GA,
        START,
        END,
        INDEX,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        WB: tl.constexpr,
        BH: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
        ADAPTIVE: tl.constexpr,
        SOURCE_START,
        W: tl.constexpr,
        SCORE_SCRATCH,
        CACHE_QK: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        si = (bh * tiles + tile) * BT * BS + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ci = (bh * tiles + tile) * BT * BT + r[:, None] * BT + r[None, :]
        if SOURCE_START + tile * BS <= END:
            sm = (s[:, None] < T) & (d[None, :] < D)
            tm = (j[:, None] <= END) & (d[None, :] < D)
            sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
            tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
            if CACHE_QK:
                scores = tl.load(SCORE_SCRATCH + si)
            else:
                q = tl.load(QF + sp, sm, 0)
                k = tl.load(KF + tp, tm, 0)
                scores = tl.where(j[None, :] <= END, tl.dot(q, tl.trans(k)) * SCALE, -float("inf"))
            ni = (INDEX * BH + bh) * W + ls
            m0 = tl.load(NORMS + ni, s < T, -float("inf"))
            l0 = tl.load(NORMS + WB * BH * W + ni, s < T, 0)
            maximum = tl.maximum(m0, tl.max(scores, 1))
            old = tl.where(l0 > 0, l0 * tl.exp(m0 - maximum), 0.0)
            e = tl.where(j[None, :] <= END, tl.exp(scores - maximum[:, None]), 0.0)
            common = False
            if ADAPTIVE:
                first = tl.reshape(tl.gather(scores, tl.full((BS, 1), 0, tl.int32), 1), (BS,))
                common = tl.max(tl.where(s < T, maximum - tl.maximum(m0, first), 0.0), 0) <= 32.0
            wi = (
                ((INDEX * BH + bh) * tiles + tile) * BT * BS
                + r[:, None] * BS
                + tl.arange(0, BS)[None, :]
            )
            wp: tl.constexpr = WB * BH * tiles * BT * BS
            c = tl.load(SCALARS + wi)
            db = tl.load(SCALARS + wp + wi)
            mu = tl.load(SCALARS + 2 * wp + wi)
            qc = tl.load(QC + tp, tm, 0)
            kc = tl.load(KC + tp, tm, 0)
            qg = tl.dot(qc, tl.trans(kc)) * SCALE
            gy = tl.load(GY + tp, tm, 0)
            vc = tl.load(VC + tp, tm, 0)
            hv = tl.dot(gy, tl.trans(vc))
            lower = r[:, None] >= r[None, :]
            if common:
                il = 1.0 / (old[:, None] + tl.cumsum(e, 1))
                rk = db * tl.trans(il)
                rv = c * tl.trans(il)
                ck = tl.where(lower, _context_dot(rk, e, FAST), 0.0)
                cv = tl.where(lower, _context_dot(rv, e, FAST), 0.0)
                ga = _context_dot(tl.trans(rk), tl.where(lower, qg, 0.0), FAST)
                ga += _context_dot(tl.trans(rv), tl.where(lower, hv, 0.0), FAST)
                ga -= tl.associative_scan(
                    tl.trans(mu * tl.trans(il)), 1, _sum_combine, reverse=True
                )
                ga *= e
            else:
                leaf = tl.broadcast_to((j[None, :] <= END).to(tl.float32), (BS, BT))
                lm, ll = tl.associative_scan((scores, leaf), 1, _prefix_combine)
                pm = tl.maximum(lm, m0[:, None])
                den = ll * tl.exp(lm - pm) + tl.where(
                    l0[:, None] > 0, l0[:, None] * tl.exp(m0[:, None] - pm), 0.0
                )
                il = 1.0 / den
                ck = _probability_contract(scores, pm, il, db, BS, BT, 4, True)
                cv = _probability_contract(scores, pm, il, c, BS, BT, 4, True)
                ga = _score_adjoint_contract(scores, pm, il, db, c, qg, hv, mu, BS, BT, 4)
            tl.store(CK + ci, ck)
            tl.store(CV + ci, cv)
            tl.store(GA + si, ga)
        else:
            tl.store(CK + ci, 0.0)
            tl.store(CV + ci, 0.0)
            tl.store(GA + si, 0.0)

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _suffix_future_scores(
        KC,
        VC,
        RK,
        RV,
        FGA,
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
        SOURCE_START,
        W: tl.constexpr,
        KIND: tl.constexpr = -1,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2) if KIND < 0 else KIND
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        if kind == 0:
            gz = RK
            values = KC
        else:
            gz = RV
            values = VC
        g = tl.load(
            gz + (bh * W + ls[:, None]) * D + d[None, :], (s[:, None] < T) & (d[None, :] < D), 0
        )
        v = tl.load(
            values + ((b * T + j[:, None]) * H + h) * D + d[None, :],
            (j[:, None] <= END) & (d[None, :] < D),
            0,
        )
        score = _context_dot(g, tl.trans(v), FAST)
        plane: tl.constexpr = BH * tiles * BS * BT
        ix = (bh * tiles + tile) * BS * BT + tl.arange(0, BS)[:, None] * BT + r[None, :]
        tl.store(FGA + kind * plane + ix, score)

    @triton.jit
    def _min_combine(a, b):
        return tl.minimum(a, b)

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _suffix_source(
        KF,
        QC,
        GY,
        GA,
        C,
        DB,
        DBO,
        CO,
        OLDP,
        RK,
        RV,
        RGA,
        RESET,
        GQF,
        GKC,
        GVC,
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
        SOURCE_START,
        W: tl.constexpr,
        PE,
        FGA,
        RMU,
        GO,
        RK_NEXT,
        RV_NEXT,
        FUSE_FUTURE: tl.constexpr,
        TARGET_OWNED: tl.constexpr,
        KIND: tl.constexpr = -1,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2) if KIND < 0 else KIND
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        state = bh * W + ls
        sp = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        si = (bh * tiles + tile) * BT * BS + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ti = (bh * tiles + tile) * BT * BS + r[:, None] * BS + tl.arange(0, BS)[None, :]
        if kind == 0:
            a = tl.load(GA + si)
            if FUSE_FUTURE:
                plane: tl.constexpr = tl.num_programs(1) * tiles * BS * BT
                mu = tl.load(RMU + state, s < T, 0)
                score = tl.load(FGA + si) + tl.load(FGA + plane + si) - mu[:, None]
                a += tl.load(PE + si) * score
                p = tl.load(OLDP + state, s < T, 0)
                go = tl.load(GO + state, s < T, 0)
                tl.store(RMU + state, go + p * mu, s < T)
            tail = tl.load(RGA + state, s < T, 0)
            raw = tl.associative_scan(a, 1, _sum_combine, reverse=True) + tail[:, None]
            reset = tl.load(RESET + si)
            # rho_j=0 makes keys <j irrelevant to every later prefix.
            # Hence suffix sum(dA)_j=0 exactly. Propagate that conservation
            # correction toward earlier keys without subtracting large KF.
            nearest = tl.associative_scan(
                tl.where(reset, r[None, :], BT), 1, _min_combine, reverse=True
            )
            correction = tl.gather(raw, tl.minimum(nearest, BT - 1), 1)
            cumulative = raw - tl.where(nearest < BT, correction, 0.0)
            nxt = tl.gather(
                cumulative, tl.broadcast_to(tl.minimum(r + 1, BT - 1)[None, :], (BS, BT)), 1
            )
            nxt = tl.where(r[None, :] < BT - 1, nxt, tail[:, None])
            repaired = cumulative - nxt
            any_reset = tl.sum(reset.to(tl.int32), 1) > 0
            tl.store(GA + si, tl.where(any_reset[:, None], repaired, a))
            first = tl.reshape(tl.gather(cumulative, tl.full((BS, 1), 0, tl.int32), 1), (BS,))
            k = tl.load(KF + tp, tm, 0).to(tl.float32)
            previous = tl.load(
                KF + ((b * T + tl.maximum(j - 1, 0)[:, None]) * H + h) * D + d[None, :], tm, 0
            ).to(tl.float32)
            delta = tl.where((j[:, None] > 0) & (j[:, None] <= END), k - previous, 0.0)
            # Summation by parts: sum dA_j KF_j = sum suffix(dA)_j ΔKF_j.
            # ΔKF_0=0 uses exact shift invariance; large common offsets cancel.
            update = _context_dot(cumulative, delta, False) * SCALE
            tl.store(GQF + sp, tl.load(GQF + sp, sm, 0) + update, sm)
            tl.store(RGA + state, first, s < T)
        elif kind == 1 or kind == 2:
            if SOURCE_START + tile * BS <= END:
                if kind == 1:
                    a = tl.trans(tl.load(DB + ti))
                    q = tl.load(QC + tp, tm, 0)
                    update = _context_dot(a, q, FAST) * SCALE
                    dest = GKC
                else:
                    a = tl.trans(tl.load(C + ti))
                    q = tl.load(GY + tp, tm, 0)
                    update = _context_dot(a, q, FAST)
                    dest = GVC
                tl.store(dest + sp, tl.load(dest + sp, sm, 0) + update, sm)
        else:
            if kind == 3:
                a = tl.trans(tl.load(DBO + ti))
                q = tl.load(QC + tp, tm, 0)
                update = _context_dot(a, q, FAST) * SCALE
                old_dest = RK
                if TARGET_OWNED:
                    dest = RK_NEXT
                else:
                    dest = RK
            else:
                a = tl.trans(tl.load(CO + ti))
                q = tl.load(GY + tp, tm, 0)
                update = _context_dot(a, q, FAST)
                old_dest = RV
                if TARGET_OWNED:
                    dest = RV_NEXT
                else:
                    dest = RV
            p = tl.load(OLDP + state, s < T, 0)
            ptr = (bh * W + ls[:, None]) * D + d[None, :]
            tl.store(dest + ptr, update + p[:, None] * tl.load(old_dest + ptr, sm, 0), sm)

    @triton.jit(do_not_specialize=["START", "END", "SOURCE_START"])
    def _suffix_target(
        QF,
        QC,
        GY,
        GA,
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
        SOURCE_START,
        W: tl.constexpr,
        KIND: tl.constexpr = -1,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2) if KIND < 0 else KIND
        b = bh // H
        h = bh % H
        ls = tile * BS + tl.arange(0, BS)
        s = SOURCE_START + ls
        r = tl.arange(0, BT)
        j = START + r
        d = tl.arange(0, BD)
        sm = (s[:, None] < T) & (d[None, :] < D)
        tm = (j[:, None] <= END) & (d[None, :] < D)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        si = (bh * tiles + tile) * BT * BS + tl.arange(0, BS)[:, None] * BT + r[None, :]
        ci = (bh * tiles + tile) * BT * BT + r[:, None] * BT + r[None, :]
        pi = ((bh * tiles + tile) * BT + r[:, None]) * D + d[None, :]
        tp = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        if kind == 0:
            a = tl.trans(tl.load(GA + si))
            q = tl.load(QF + ((b * T + s[:, None]) * H + h) * D + d[None, :], sm, 0)
            update = _context_dot(a, q, FAST) * SCALE
        elif kind == 1:
            ck = tl.trans(tl.load(CK + ci))
            q = tl.load(QC + tp, tm, 0)
            update = _context_dot(ck, q, FAST) * SCALE + tl.load(FK + pi, tm, 0)
        else:
            cv = tl.trans(tl.load(CV + ci))
            g = tl.load(GY + tp, tm, 0)
            update = _context_dot(cv, g, FAST) + tl.load(FV + pi, tm, 0)
        plane: tl.constexpr = BH * tiles * BT * D
        tl.store(PART + kind * plane + pi, update, tm)

    @triton.jit(do_not_specialize=["START", "SOURCE_START"])
    def _window_reduce(
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
        PLANES: tl.constexpr,
        SOURCE_START,
        W: tl.constexpr,
        NB: tl.constexpr = 1,
    ):
        t = tl.program_id(0)
        bh = tl.program_id(1)
        kind = tl.program_id(2)
        b = bh // H
        h = bh % H
        d = tl.arange(0, BD)
        s = tl.arange(0, BP)
        tiles: tl.constexpr = triton.cdiv(W, BS)
        plane: tl.constexpr = BH * tiles * BT * D
        slot = t // BT
        r = t % BT
        idx = ((bh * tiles + s[:, None]) * BT + r) * D + d[None, :]
        p = tl.load(
            PART + slot * PLANES * plane + kind * plane + idx,
            (s[:, None] < tiles) & (d[None, :] < D),
            0,
        )
        value = tl.sum(p, 0)
        out = ((b * T + START + t) * H + h) * D + d
        if PLANES == 1:
            dest = GQC
        elif kind == 0:
            dest = GKF
        elif kind == 1:
            dest = GKC
        else:
            dest = GVC
        tl.store(dest + out, tl.load(dest + out, d < D, 0) + value, d < D)

    @triton.jit(do_not_specialize=["START", "END"])
    def _suffix_future_values_inline(
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
        KIND: tl.constexpr,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        kind = KIND
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

    @triton.jit(do_not_specialize=["INDEX", "SOURCE_START"])
    def _prefix_superblock(
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
        SCALARS,
        NORMS,
        PE,
        OLDP,
        QPART,
        INDEX,
        WB: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
        ADAPTIVE: tl.constexpr,
        SOURCE_START,
        W: tl.constexpr,
        NB: tl.constexpr,
    ):
        # Source-owned bounded persistence. Each CTA owns its source state and
        # writes NB dQC partial blocks. No inter-CTA exchange within the loop.
        tiles: tl.constexpr = triton.cdiv(W, BS)
        plane: tl.constexpr = BH * tiles * BT * D
        for slot in range(NB):
            index = INDEX + slot
            if index < WB:
                start = index * BT
                end = tl.minimum(start + BT, T) - 1
                _prefix_prepare(
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
                    SCALARS,
                    NORMS,
                    index,
                    WB,
                    start,
                    end,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    SCALE,
                    FAST,
                    ADAPTIVE,
                    SOURCE_START,
                    W,
                    PE,
                    OLDP,
                    True,
                )
                tl.debug_barrier()
                _prefix_qc(
                    QC,
                    KC,
                    ZK,
                    DB,
                    DBO,
                    CK,
                    QPART + slot * plane,
                    start,
                    end,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    SCALE,
                    FAST,
                    SOURCE_START,
                    W,
                )
                _prefix_state_from_coefficients(
                    KC,
                    VC,
                    ZK,
                    ZV,
                    L,
                    PE,
                    OLDP,
                    start,
                    end,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    FAST,
                    SOURCE_START,
                    W,
                    0,
                )
                _prefix_state_from_coefficients(
                    KC,
                    VC,
                    ZK,
                    ZV,
                    L,
                    PE,
                    OLDP,
                    start,
                    end,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    FAST,
                    SOURCE_START,
                    W,
                    1,
                )
                tl.debug_barrier()

    @triton.jit(do_not_specialize=["INDEX", "SOURCE_START"])
    def _suffix_superblock(
        QF,
        KF,
        QC,
        KC,
        VC,
        GY,
        SCALARS,
        NORMS,
        C,
        DB,
        DBO,
        CO,
        CK,
        CV,
        GA,
        PE,
        OLDP,
        GO,
        RESET,
        FGA,
        FK,
        FV,
        RK,
        RV,
        RMU,
        RGA,
        GQF,
        GKC,
        GVC,
        PART,
        INDEX,
        WB: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BT: tl.constexpr,
        BH: tl.constexpr,
        SCALE: tl.constexpr,
        FAST: tl.constexpr,
        ADAPTIVE: tl.constexpr,
        SOURCE_START,
        W: tl.constexpr,
        NB: tl.constexpr,
    ):
        # Descending targets, same source CTA throughout this bounded group.
        # All scalar scratch and suffix state are CTA-private source slices.
        # Target partials are reduced in a separate launch after the group.
        tiles: tl.constexpr = triton.cdiv(W, BS)
        plane: tl.constexpr = BH * tiles * BT * D
        for step in range(NB):
            slot = NB - 1 - step
            index = INDEX + slot
            if index < WB:
                start = index * BT
                end = tl.minimum(start + BT, T) - 1
                _suffix_scalar(
                    QF,
                    KF,
                    SCALARS,
                    NORMS,
                    C,
                    DB,
                    DBO,
                    CO,
                    CK,
                    CV,
                    GA,
                    PE,
                    OLDP,
                    GO,
                    RESET,
                    start,
                    end,
                    index,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    WB,
                    BH,
                    SCALE,
                    FAST,
                    ADAPTIVE,
                    SOURCE_START,
                    W,
                    FGA,
                    True,
                )
                tl.debug_barrier()
                _suffix_contractions(
                    QF,
                    KF,
                    QC,
                    KC,
                    VC,
                    GY,
                    SCALARS,
                    NORMS,
                    CK,
                    CV,
                    GA,
                    start,
                    end,
                    index,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    WB,
                    BH,
                    SCALE,
                    FAST,
                    ADAPTIVE,
                    SOURCE_START,
                    W,
                    FGA,
                    True,
                )
                _suffix_future_scores(
                    KC,
                    VC,
                    RK,
                    RV,
                    FGA,
                    start,
                    end,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    BH,
                    FAST,
                    SOURCE_START,
                    W,
                    0,
                )
                _suffix_future_scores(
                    KC,
                    VC,
                    RK,
                    RV,
                    FGA,
                    start,
                    end,
                    T,
                    H,
                    D,
                    BD,
                    BS,
                    BT,
                    BH,
                    FAST,
                    SOURCE_START,
                    W,
                    1,
                )
                _suffix_future_values_inline(
                    PE, RK, RV, FK, FV, start, end, W, D, BD, BS, BT, FAST, 0
                )
                _suffix_future_values_inline(
                    PE, RK, RV, FK, FV, start, end, W, D, BD, BS, BT, FAST, 1
                )
                tl.debug_barrier()
                # Kind 0 repairs GA; kinds 3/4 advance R only after future
                # values have consumed R_after. All direct source writes are
                # disjoint across CTAs and target reduction occurs afterward.
                for kind in tl.static_range(5):
                    _suffix_source(
                        KF,
                        QC,
                        GY,
                        GA,
                        C,
                        DB,
                        DBO,
                        CO,
                        OLDP,
                        RK,
                        RV,
                        RGA,
                        RESET,
                        GQF,
                        GKC,
                        GVC,
                        start,
                        end,
                        T,
                        H,
                        D,
                        BD,
                        BS,
                        BT,
                        SCALE,
                        FAST,
                        SOURCE_START,
                        W,
                        PE,
                        FGA,
                        RMU,
                        GO,
                        RK,
                        RV,
                        True,
                        False,
                        kind,
                    )
                tl.debug_barrier()
                for kind in tl.static_range(3):
                    _suffix_target(
                        QF,
                        QC,
                        GY,
                        GA,
                        CK,
                        CV,
                        FK,
                        FV,
                        PART + slot * 3 * plane,
                        start,
                        end,
                        T,
                        H,
                        D,
                        BD,
                        BS,
                        BT,
                        BH,
                        SCALE,
                        FAST,
                        SOURCE_START,
                        W,
                        kind,
                    )
                tl.debug_barrier()


def backward_run(inputs, y32, lc, gy):
    """One forward and one reverse target traversal per disjoint source window."""
    qf, kf, qc, kc, vc = inputs
    b, t, h, d = qf.shape
    bh = b * h
    bs, bt, w, nb = 32, 16, 512, 4
    bd = max(32, triton.next_power_of_2(d))
    tiles = triton.cdiv(w, bs)
    wb = triton.cdiv(t, bt)
    scale = d**-0.5
    zk, zv, m, l = (x[0] for x in allocate_state(b, h, w, d, qf.device))
    rk = torch.empty((bh, w, d), device=qf.device, dtype=torch.float32)
    rv = torch.empty_like(rk)
    rmu = torch.empty((bh, w), device=qf.device, dtype=torch.float32)
    rga = torch.empty_like(rmu)
    grads = [torch.zeros_like(qf, dtype=torch.float32) for _ in range(5)]
    gqf, gkf, gqc, gkc, gvc = grads
    # Bounded W×T histories only: never T×T histories or D-wide checkpoints.
    scalars = torch.empty((3, wb, bh, tiles, bt, bs), device=qf.device, dtype=torch.float32)
    norms = torch.empty((2, wb, bh, w), device=qf.device, dtype=torch.float32)
    c, db, dbo, co, ga, pe = [
        torch.empty((bh, tiles, bt, bs), device=qf.device, dtype=torch.float32) for _ in range(6)
    ]
    ck = torch.empty((bh, tiles, bt, bt), device=qf.device, dtype=torch.float32)
    cv = torch.empty_like(ck)
    fk = torch.empty((bh, tiles, bt, d), device=qf.device, dtype=torch.float32)
    fv = torch.empty_like(fk)
    fga = torch.empty((2, bh, tiles, bs, bt), device=qf.device, dtype=torch.float32)
    oldp = torch.empty_like(rmu)
    go = torch.empty_like(rmu)
    reset = torch.empty((bh, tiles, bs, bt), device=qf.device, dtype=torch.bool)
    part = torch.empty((nb, 3, bh, tiles, bt, d), device=qf.device, dtype=torch.float32)
    qpart = torch.empty((nb, bh, tiles, bt, d), device=qf.device, dtype=torch.float32)
    gy = gy.contiguous()
    # SOURCE-WINDOW-MAJOR: pair work is O(1), without T/W catch-up replays.
    for source_start in range(0, t, w):
        zk.zero_()
        zv.zero_()
        m.fill_(-float("inf"))
        l.zero_()
        rk.zero_()
        rv.zero_()
        rmu.zero_()
        rga.zero_()
        for i in range(0, wb, nb):
            start = i * bt
            end = min(start + nb * bt, t) - 1
            _prefix_superblock[(tiles, bh)](
                qf,
                kf,
                qc,
                kc,
                vc,
                gy,
                y32,
                lc,
                zk,
                zv,
                m,
                l,
                c,
                db,
                dbo,
                co,
                ck,
                scalars,
                norms,
                pe,
                oldp,
                qpart,
                i,
                wb,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                bh,
                scale,
                True,
                True,
                source_start,
                w,
                nb,
                num_warps=4,
                num_stages=3,
            )
            _window_reduce[(end - start + 1, bh, 1)](
                qpart,
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
                1,
                source_start,
                w,
                nb,
                num_warps=4,
            )
        for i in reversed(range(0, wb, nb)):
            start = i * bt
            end = min(start + nb * bt, t) - 1
            _suffix_superblock[(tiles, bh)](
                qf,
                kf,
                qc,
                kc,
                vc,
                gy,
                scalars,
                norms,
                c,
                db,
                dbo,
                co,
                ck,
                cv,
                ga,
                pe,
                oldp,
                go,
                reset,
                fga,
                fk,
                fv,
                rk,
                rv,
                rmu,
                rga,
                gqf,
                gkc,
                gvc,
                part,
                i,
                wb,
                t,
                h,
                d,
                bd,
                bs,
                bt,
                bh,
                scale,
                True,
                True,
                source_start,
                w,
                nb,
                num_warps=4,
                num_stages=3,
            )
            _window_reduce[(end - start + 1, bh, 3)](
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
                3,
                source_start,
                w,
                nb,
                num_warps=4,
            )
    return tuple(grads)


class _BoundedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        y, y32, lc = fw.run(inputs)
        ctx.save_for_backward(*inputs, y32, lc)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, gy):
        xs = ctx.saved_tensors
        gradients = backward_run(xs[:5], xs[5], xs[6], gy)
        return tuple(g.to(x.dtype) for g, x in zip(gradients, xs[:5]))


def run(inputs):
    check_long(inputs)
    return _BoundedAttention.apply(*inputs)
