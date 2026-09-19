"""Guarded stable 32-source × 16-target fused forward."""

from __future__ import annotations
import math
import torch
from torch.autograd.function import once_differentiable
from ._common import triton, tl

from ._common import check_long, allocate_state

if triton is not None:
    from ._common import _prefix_combine, _probability_contract, _context_dot

    @triton.jit
    def _finish_two_contexts(
        scores,
        kc,
        vc,
        m0,
        l0,
        STATE_K,
        STATE_V,
        STATE_M,
        STATE_L,
        state,
        s,
        d,
        source_mask,
        valid_s,
        valid_j,
        BLOCK_START,
        D: tl.constexpr,
        BM: tl.constexpr,
        BD: tl.constexpr,
        FAST_DOT: tl.constexpr,
    ):
        maximum = tl.maximum(m0, tl.max(scores, 1))
        old_mass = tl.where(l0 > 0, l0 * tl.exp(m0 - maximum), 0.0)
        e = tl.where(valid_j[None, :], tl.exp(scores - maximum[:, None]), 0.0)
        den = old_mass + tl.sum(e, 1)
        if BLOCK_START == 0:
            zk = tl.full((BM, BD), 0.0, tl.float32)
        else:
            zk = tl.load(STATE_K + state[:, None] * D + d[None, :], source_mask, 0)
        zk_new = (old_mass[:, None] * zk + _context_dot(e, kc, FAST_DOT)) / den[:, None]
        tl.store(STATE_K + state[:, None] * D + d[None, :], zk_new, source_mask)
        if BLOCK_START == 0:
            zv = tl.full((BM, BD), 0.0, tl.float32)
        else:
            zv = tl.load(STATE_V + state[:, None] * D + d[None, :], source_mask, 0)
        zv_new = (old_mass[:, None] * zv + _context_dot(e, vc, FAST_DOT)) / den[:, None]
        tl.store(STATE_V + state[:, None] * D + d[None, :], zv_new, source_mask)
        tl.store(STATE_M + state, maximum, valid_s)
        tl.store(STATE_L + state, den, valid_s)

    @triton.jit(do_not_specialize=["BLOCK_START", "BLOCK_END"])
    def _two_geometry_factorized(
        Q,
        K,
        VC,
        KC,
        QC,
        STATE_K,
        STATE_V,
        STATE_M,
        STATE_L,
        PART_A,
        PART_M,
        PART_L,
        BLOCK_START,
        BLOCK_END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BS: tl.constexpr,
        BM: tl.constexpr,
        BT: tl.constexpr,
        BN: tl.constexpr,
        SOURCE_TILES: tl.constexpr,
        SCALE: tl.constexpr,
        GROUP: tl.constexpr,
        SPLIT_STATE: tl.constexpr,
        FAST_DOT: tl.constexpr,
        ADAPTIVE: tl.constexpr,
    ):
        source_tile = tl.program_id(0)
        h = tl.program_id(1)
        b = tl.program_id(2)
        s = source_tile * BS + tl.arange(0, BM)
        r = tl.arange(0, BN)
        j = BLOCK_START + r
        d = tl.arange(0, BD)
        valid_s = (s < (source_tile + 1) * BS) & (s < T)
        valid_j = j <= BLOCK_END
        source_ptr = ((b * T + s[:, None]) * H + h) * D + d[None, :]
        target_ptr = ((b * T + j[:, None]) * H + h) * D + d[None, :]
        source_mask = valid_s[:, None] & (d[None, :] < D)
        target_mask = valid_j[:, None] & (d[None, :] < D)
        qf = tl.load(Q + source_ptr, source_mask, 0)
        u_bf16 = tl.load(KC + source_ptr, source_mask, 0)
        kf = tl.load(K + target_ptr, target_mask, 0)
        vf_bf16 = tl.load(VC + target_ptr, target_mask, 0)
        qc_bf16 = tl.load(QC + target_ptr, target_mask, 0)
        kc_target = tl.load(KC + target_ptr, target_mask, 0)
        state = ((b * H + h) * T + s).to(tl.int64)
        if BLOCK_START == 0:
            m0 = tl.full((BM,), -float("inf"), tl.float32)
            l0_previous = tl.full((BM,), 0.0, tl.float32)
        else:
            m0 = tl.load(STATE_M + state, valid_s, -float("inf"))
            l0_previous = tl.load(STATE_L + state, valid_s, 0.0)

        scores = tl.dot(qf, tl.trans(kf)) * SCALE
        scores = tl.where(valid_j[None, :], scores, -float("inf"))
        if source_tile * BS <= BLOCK_END:
            q_current = tl.dot(qc_bf16, tl.trans(kc_target)) * SCALE
            use_common = False
            common_m = tl.maximum(m0, tl.max(scores, 1))
            e_common = tl.full((BM, BN), 0.0, tl.float32)
            prefix_m = tl.full((BM, BN), 0.0, tl.float32)
            lower = r[:, None] >= r[None, :]
            if ADAPTIVE:
                first_score = tl.reshape(tl.gather(scores, tl.full((BM, 1), 0, tl.int32), 1), (BM,))
                first_m = tl.maximum(m0, first_score)
                # E and R cannot overflow/underflow at this conservative bound.
                # Every prefix has at least exp(-32) mass in common units.
                gap = tl.where(valid_s, common_m - first_m, 0.0)
                use_common = tl.max(gap, 0) <= 32.0
            if use_common:
                old_mass_common = tl.where(
                    l0_previous > 0, l0_previous * tl.exp(m0 - common_m), 0.0
                )
                e_common = tl.where(valid_j[None, :], tl.exp(scores - common_m[:, None]), 0.0)
                den_common = old_mass_common[:, None] + tl.cumsum(e_common, 1)
                prefix_inverse_l = 1.0 / den_common
                lower = r[:, None] >= r[None, :]
                new_score = _context_dot(
                    tl.where(lower, q_current, 0.0), tl.trans(e_common), FAST_DOT
                )
                new_score *= tl.trans(prefix_inverse_l)
                old_probability = tl.trans(old_mass_common[:, None] * prefix_inverse_l)
            else:
                # Exact prefix-local scaling is the fallback, including gap=128.
                leaf_l = tl.broadcast_to(valid_j[None, :].to(tl.float32), (BM, BN))
                local_m, local_l = tl.associative_scan((scores, leaf_l), 1, _prefix_combine)
                prefix_m = tl.maximum(local_m, m0[:, None])
                prefix_l = local_l * tl.exp(local_m - prefix_m)
                prefix_l += tl.where(
                    l0_previous[:, None] > 0,
                    l0_previous[:, None] * tl.exp(m0[:, None] - prefix_m),
                    0.0,
                )
                prefix_inverse_l = 1.0 / prefix_l
                new_score = _probability_contract(
                    scores, prefix_m, prefix_inverse_l, q_current, BM, BN, GROUP, False
                )
                old_probability = tl.trans(
                    tl.where(
                        l0_previous[:, None] > 0,
                        l0_previous[:, None] * tl.exp(m0[:, None] - prefix_m) * prefix_inverse_l,
                        0.0,
                    )
                )
            if BLOCK_START == 0:
                zk0 = tl.full((BM, BD), 0.0, tl.float32)
            else:
                zk0 = tl.load(STATE_K + state[:, None] * D + d[None, :], source_mask, 0)
            q_old = _context_dot(qc_bf16, tl.trans(zk0), FAST_DOT) * SCALE
            logits = new_score + q_old * old_probability
            logits += tl.dot(qc_bf16, tl.trans(u_bf16)) * SCALE
            active = valid_j[:, None] & valid_s[None, :] & (s[None, :] <= j[:, None])
            masked_logits = tl.where(active, logits, -float("inf"))
            has_source = tl.sum(active.to(tl.int32), 1) > 0
            partial_max = tl.where(has_source, tl.max(masked_logits, 1), 0.0)
            w = tl.where(active, tl.exp(masked_logits - partial_max[:, None]), 0.0)
            partial_den = tl.sum(w, 1)
            # No key context survives into the value contractions.
            vc_source = tl.load(VC + source_ptr, source_mask, 0)
            partial_num = _context_dot(w, vc_source, FAST_DOT)
            if BLOCK_START == 0:
                zv0 = tl.full((BM, BD), 0.0, tl.float32)
            else:
                zv0 = tl.load(STATE_V + state[:, None] * D + d[None, :], source_mask, 0)
            partial_num += _context_dot(w * old_probability, zv0, FAST_DOT)
            if use_common:
                cur_coeff = _context_dot(w * tl.trans(prefix_inverse_l), e_common, FAST_DOT)
                cur_coeff = tl.where(lower, cur_coeff, 0.0)
            else:
                cur_coeff = _probability_contract(
                    scores, prefix_m, prefix_inverse_l, w, BM, BN, GROUP, True
                )
            partial_num += _context_dot(cur_coeff, vf_bf16, FAST_DOT)
            partial = (((b * H + h) * SOURCE_TILES + source_tile) * BT + r).to(tl.int64)
            tl.store(PART_M + partial, tl.where(has_source, partial_max, -float("inf")), valid_j)
            tl.store(PART_L + partial, partial_den, valid_j)
            tl.store(PART_A + partial[:, None] * D + d[None, :], partial_num, target_mask)
        else:
            # Future-source inner state must still advance. Only outer
            # contractions are skipped; write neutral partials for reduction.
            partial = (((b * H + h) * SOURCE_TILES + source_tile) * BT + r).to(tl.int64)
            tl.store(PART_M + partial, -float("inf"), valid_j)
            tl.store(PART_L + partial, 0.0, valid_j)
            tl.store(PART_A + partial[:, None] * D + d[None, :], 0.0, target_mask)
        if not SPLIT_STATE:
            _finish_two_contexts(
                scores,
                kc_target,
                vf_bf16,
                m0,
                l0_previous,
                STATE_K,
                STATE_V,
                STATE_M,
                STATE_L,
                state,
                s,
                d,
                source_mask,
                valid_s,
                valid_j,
                BLOCK_START,
                D,
                BM,
                BD,
                FAST_DOT,
            )

    @triton.jit(do_not_specialize=["START", "END"])
    def _training_reduce(
        PA,
        PM,
        PL,
        Y,
        Y32,
        LC,
        START,
        END,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
        TILES: tl.constexpr,
        BP: tl.constexpr,
    ):
        r = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        d = tl.arange(0, BD)
        tile = tl.arange(0, BP)
        p = ((bh * TILES + tile) * BT + r).to(tl.int64)
        m = tl.load(PM + p, tile < TILES, -float("inf"))
        l = tl.load(PL + p, tile < TILES, 0)
        a = tl.load(PA + p[:, None] * D + d[None, :], (tile[:, None] < TILES) & (d[None, :] < D), 0)
        maximum = tl.max(m, 0)
        scale = tl.where(l > 0, tl.exp(m - maximum), 0.0)
        denominator = tl.sum(scale * l, 0)
        value = tl.sum(a * scale[:, None], 0) / denominator
        out = ((b * T + START + r) * H + h) * D + d
        tl.store(Y + out, value, d < D)
        tl.store(Y32 + out, value, d < D)
        tl.store(LC + bh * T + START + r, maximum + tl.log(denominator))


if triton is not None:

    @triton.jit(do_not_specialize=["BLOCK_START"])
    def _preaged_outer_reduce(
        PART_A,
        PART_M,
        PART_L,
        Y,
        BLOCK_START,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
        SOURCE_TILES: tl.constexpr,
        BP: tl.constexpr,
    ):
        local_t = tl.program_id(0)
        h = tl.program_id(1)
        b = tl.program_id(2)
        target = BLOCK_START + local_t
        tile = tl.arange(0, BP)
        d = tl.arange(0, BD)
        partial = (((b * H + h) * SOURCE_TILES + tile) * BT + local_t).to(tl.int64)
        maximums = tl.load(PART_M + partial, tile < SOURCE_TILES, -float("inf"))
        denominators = tl.load(PART_L + partial, tile < SOURCE_TILES, 0.0)
        numerators = tl.load(
            PART_A + partial[:, None] * D + d[None, :],
            (tile[:, None] < SOURCE_TILES) & (d[None, :] < D),
            0,
        )
        maximum = tl.max(maximums, 0)
        scales = tl.where(denominators > 0, tl.exp(maximums - maximum), 0.0)
        denominator = tl.sum(scales * denominators, 0)
        numerator = tl.sum(scales[:, None] * numerators, 0)
        out = ((b * T + target) * H + h) * D + d
        tl.store(Y + out, numerator / denominator, d < D)


def run(inputs, fast=True, adaptive=True, training=True):
    b, t, h, d = check_long(inputs)
    bh = b * h
    bs, bt = 32, 16
    bd = max(32, triton.next_power_of_2(d))
    tiles = triton.cdiv(t, bs)
    qf, kf, qc, kc, vc = inputs
    state = allocate_state(b, h, t, d, qf.device, 1)
    if training:
        state[0].zero_()
        state[1].zero_()
        state[2].fill_(-float("inf"))
        state[3].zero_()
    zk, zv, m, l = (x[0] for x in state)
    pa = torch.empty((bh, tiles, bt, d), device=qf.device, dtype=torch.float32)
    pm = torch.empty((bh, tiles, bt), device=qf.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    y = torch.empty_like(qf)
    y32 = torch.empty_like(qf, dtype=torch.float32) if training else None
    lc = torch.empty((bh, t), device=qf.device, dtype=torch.float32) if training else None
    for start in range(0, t, bt):
        end = min(start + bt, t) - 1
        _two_geometry_factorized[(tiles, h, b)](
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
        if training:
            _training_reduce[(end - start + 1, bh)](
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
        else:
            _preaged_outer_reduce[(end - start + 1, h, b)](
                pa,
                pm,
                pl,
                y,
                start,
                t,
                h,
                d,
                bd,
                bt,
                tiles,
                triton.next_power_of_2(tiles),
                num_warps=4,
                num_stages=3,
            )
    return y, y32, lc
