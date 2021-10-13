import numpy as np
import math

def ccsd_t2_20(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir)

    g_vvoo = g_so[nocc:, nocc:, :nocc, :nocc]
    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dp[:nocc, :nocc]
    d_vv = -dp[nocc:, nocc:]
    d_ov = -dp[:nocc, nocc:]
    d_vo = -dp[nocc:, :nocc]

    res_t2_20 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_20 += 1.0 * np.einsum('baji->abij', g_vvoo, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t1_10, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_20, optimize=True)
    #res_t2_20 += 1.0 * np.einsum('I,baji->abijI', G, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bi,aj->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ai,bj->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bj,ai->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('aj,bi->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kj,baki->abij', d_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('bc,acji->abij', d_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('ac,bcji->abij', d_vv, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,cj,bk,al->abij', g_oovv, t1_10, t1_10, t1_10, t1_10, optimize=True)


    t2_20 += np.einsum('abij,iajb -> abij', res_t2_20, e_denom, optimize=True)

    return t2_20

def ccsd_t2_21(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    eps_vir_p_w = eps[nocc:] + w
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_w)

    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dp[:nocc, :nocc]
    d_vv = -dp[nocc:, nocc:]
    d_ov = -dp[:nocc, nocc:]
    d_vo = -dp[nocc:, :nocc]

    res_t2_21 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t2_11, optimize = True)
    res_t2_21 += 1.0 * w * np.einsum('baji->abij', t2_21, optimize = True)
    #res_t2_21 += 1.0 * G * np.einsum('J,baji->abij', t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bi,aj->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ai,bj->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bj,ai->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('aj,bi->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_22, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bcji,ak->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,acji,bk->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,baki,cj->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bakj,ci->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klji,ak,bl->abij', g_oooo, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,ak,cj->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,bk,cj->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,ak,ci->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,bk,ci->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bacd,dj,ci->abij', g_vvvv, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,bakj,cl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bckj,al->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,ackj,bl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klic,balk,cj->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,baki,cl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bcki,al->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,acki,bl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kljc,balk,ci->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,adji,ck->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kbcd,dcji,ak->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,bdji,ck->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('kacd,dcji,bk->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,adki,cj->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,bdki,cj->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,adkj,ci->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,bdkj,ci->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,bakj,dcli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bdkj,acli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,adkj,bcli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,dckj,bali->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.25 * np.einsum('klcd,balk,dcji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bdlk,acji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,adlk,bcji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ki,bj,ak->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,aj,bk->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kj,bi,ak->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,ai,bk->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bc,ai,cj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ac,bi,cj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bi,ackj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ai,bckj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 2.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bj,acki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,aj,bcki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -2.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 2.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -2.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,cj,ak,bl->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bk,al,cj->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,ci,ak,bl->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bk,al,ci->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,ak,cj->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,bk,cj->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,dj,ak,ci->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,dj,bk,ci->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,bakj,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,bckj,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,ackj,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,di,balk,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,baki,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,bcki,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,acki,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bk,adji,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bk,dcji,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,ak,bdji,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dk,bcji,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,ak,dcji,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dk,acji,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bk,adli,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,ak,bdli,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dk,bali,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,dj,balk,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bk,adlj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,ak,bdlj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dk,balj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,cj,bi,ak->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,ai,bk->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bk,ai,cj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ak,bi,cj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)

    t2_21 += np.einsum('abij,iajb -> abij', res_t2_21, e_denom, optimize=True)

    return t2_21

def ccsd_t2_22(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    eps_vir_p_2w = eps[nocc:] + 2 * w
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_2w)

    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dp[:nocc, :nocc]
    d_vv = -dp[nocc:, nocc:]
    d_ov = -dp[:nocc, nocc:]

    res_t2_22 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t2_12, optimize = True)
    res_t2_22 += 1.0 * w * np.einsum('baji->abij', t2_22, optimize = True)
    res_t2_22 += 1.0 * w * np.einsum('baji->abij', t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klji,ak,bl->abij', g_oooo, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,ak,cj->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,bk,cj->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,ak,ci->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,bk,ci->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bacd,dj,ci->abij', g_vvvv, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kj,baki->abij', d_oo, t2_22, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('bc,acji->abij', d_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('ac,bcji->abij', d_vv, t2_22, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,bakj,cl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bckj,al->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,ackj,bl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klic,balk,cj->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,baki,cl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bcki,al->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,acki,bl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kljc,balk,ci->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,adji,ck->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kbcd,dcji,ak->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bdji,ck->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('kacd,dcji,bk->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,adki,cj->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,bdki,cj->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,adkj,ci->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bdkj,ci->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,bakj,dcli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bdkj,acli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,adkj,bcli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dckj,bali->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.25 * np.einsum('klcd,balk,dcji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bdlk,acji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,adlk,bcji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,ci,bakj->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,cj,baki->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,bk,acji->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ak,bcji->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klji,bk,al->abij', g_oooo, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t2_11, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,bj,ak->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,aj,bk->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,bi,ak->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,ai,bk->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,ai,cj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,bi,cj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,aj,ci->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bj,ci->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,cj,balk->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,ck,balj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ci,balk->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bi,ackj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ai,bckj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bj,acki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,aj,bcki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,baji,ck->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,cj,ak,bl->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bk,al,cj->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ci,ak,bl->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bk,al,ci->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,ak,cj->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,bk,cj->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,dj,ak,ci->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,dj,bk,ci->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,bakj,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,bckj,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,ackj,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,di,balk,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,baki,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,bcki,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,acki,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,adji,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bk,dcji,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bdji,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dk,bcji,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,ak,dcji,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dk,acji,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bk,adli,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,ak,bdli,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dk,bali,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dj,balk,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,adlj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bdlj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dk,balj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,bk,cj,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klic,ak,cj,bl->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,bk,ci,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kljc,ak,ci,bl->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbcd,dj,ci,ak->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kacd,dj,ci,bk->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,ak,di,cj->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,bk,di,cj->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,cj,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,cj,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,cj,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,cj,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,bi,ak->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,ai,bk->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,ai,cj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,bi,cj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,aj,ci->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bj,ci->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 2.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,ci,balk->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bk,di,aclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,ak,di,bclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dk,ci,balj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bk,dj,acli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,ak,dj,bcli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dk,cj,bali->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bl,dcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dk,bl,acji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dk,al,bcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdji,ak,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adji,bk,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dcji,bk,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,baki,dj,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdki,cj,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adki,cj,bl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bakj,di,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bdkj,ci,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,adkj,ci,bl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,balk,di,cj->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,bi,cj,ak->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ai,cj,bk->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)

    t2_22 += np.einsum('abij,iajb -> abij', res_t2_22, e_denom, optimize=True)

    return t2_22


def ccsd_t1_10(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    f_vo = f_so[nocc:, :nocc]

    d_oo = -dp[:nocc, :nocc]
    d_vv = -dp[nocc:, nocc:]
    d_ov = -dp[:nocc, nocc:]
    d_vo = -dp[nocc:, :nocc]

    res_t1_10 = np.zeros((nvir, nocc))

    res_t1_10 += 1.0 * np.einsum('ai->ai', f_vo, optimize=True)
    res_t1_10 += -1.0 * np.einsum('ji,aj->ai', f_oo, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('ab,bi->ai', f_vv, t1_10, optimize=True)
    res_t1_10 += 1.0 * t1_01 * np.einsum('ai->ai', d_vo, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_20, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t1_10, optimize=True)
    res_t1_10 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_20, optimize=True)
    #res_t1_10 += 1.0 * np.einsum('ai->aiI', G, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize=True)
    res_t1_10 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t1_10, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize=True)
    res_t1_10 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t1_10, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t1_10, optimize=True)

    t1_10 += np.einsum('ai,ia -> ai', res_t1_10, e_denom, optimize=True)

    return t1_10

def ccsd_t1_01(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    d_ov = -dp[:nocc, nocc:]

    res_t1_01 = 0
    G = 0
    #res_t1_01 += 1.0 * G
    res_t1_01 += 1.0 * w * t1_01
    res_t1_01 += 1.0 * G * t2_02
    res_t1_01 += 1.0 * np.einsum('ia,ai->', d_ov, t1_10, optimize=True)
    res_t1_01 += 1.0 * np.einsum('ia,ai->', f_ov, t2_11, optimize=True)
    res_t1_01 += 1.0 * np.einsum('ia,ai->', d_ov, t2_12, optimize=True)
    res_t1_01 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t1_10, optimize=True)
    res_t1_01 += 0.25 * np.einsum('ijab,baji->', g_oovv, t2_21, optimize=True)
    res_t1_01 += 1.0 * t1_01 * np.einsum('ia,ai->', d_ov, t2_11, optimize=True)
    res_t1_01 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t1_10, t2_11, optimize=True)

    if w == 0:
        t1_01 = 0
    else:
        t1_01 += -res_t1_01 / w

    return t1_01

def ccsd_t2_02(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    d_ov = -dp[:nocc, nocc:]

    res_t2_02 = 0

    res_t2_02 += 1.0 * w * t2_02
    res_t2_02 += 1.0 * w * t2_02
    res_t2_02 += 1.0 * np.einsum('ia,ai->', f_ov, t2_12)
    res_t2_02 += 1.0 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 1.0 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 0.25 * np.einsum('ijab,baji->', g_oovv, t2_22)
    res_t2_02 += 1.0 * t1_01 * np.einsum('ia,ai->', d_ov, t2_12)
    res_t2_02 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t1_10, t2_12)
    res_t2_02 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t2_11, t2_11)

    if w == 0:
        t2_02 = 0
    else:
        t2_02 += -res_t2_02 / (2 * w)

    return t2_02

def ccsd_t2_11(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:] + w
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dp[:nocc, :nocc]
    d_vv = -dp[nocc:, nocc:]
    d_ov = -dp[:nocc, nocc:]
    d_vo = -dp[nocc:, :nocc]

    res_t2_11 = np.zeros((nvir, nocc))

    res_t2_11 += 1.0 * np.einsum('ai->ai', d_vo, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize = True)
    res_t2_11 += 1.0 * t2_02 * np.einsum('ai->ai', d_vo, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', f_oo, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', f_vv, t2_11, optimize = True)
    res_t2_11 += 1.0 * w * np.einsum('ai->ai', t2_11, optimize = True)
    #res_t2_11 += 1.0 * G * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_21, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_12, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_12, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize = True)
    res_t2_11 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_21, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_21, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_22, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,aj,bi->ai', f_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_11 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkib,bj,ak->ai', g_ooov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jabc,cj,bi->ai', g_ovvv, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,acji,bk->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkbc,cbji,ak->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkbc,ackj,bi->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jb,ai,bj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_11 += -2.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkbc,ci,bj,ak->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkbc,aj,ck,bi->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)

    t2_11 += np.einsum('ai,ia -> ai', res_t2_11, e_denom, optimize=True)

    return t2_11

def ccsd_t2_12(f_so, g_so, dp, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:] + 2 * w
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dp[:nocc, :nocc]
    d_vv = -dp[nocc:, nocc:]
    d_ov = -dp[:nocc, nocc:]

    res_t2_12 = np.zeros((nvir, nocc))

    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', f_oo, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', f_vv, t2_12, optimize = True)
    res_t2_12 += 1.0 * w * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_12 += 1.0 * w * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_22, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_22, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_22, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', f_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t2_12, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t2_12, optimize = True)
    res_t2_12 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkib,bj,ak->ai', g_ooov, t1_10, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jabc,cj,bi->ai', g_ovvv, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_22, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,acji,bk->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkbc,cbji,ak->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkbc,ackj,bi->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += -2.0 * np.einsum('jb,bi,aj->ai', f_ov, t2_11, t2_11, optimize = True)
    res_t2_12 += -2.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t2_11, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,ai,bj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,ci,bj,ak->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,aj,ck,bi->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -2.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,aj,ci,bk->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,cj,bi,ak->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)

    t2_12 += np.einsum('ai,ia -> ai', res_t2_12, e_denom, optimize=True)

    return t2_12
