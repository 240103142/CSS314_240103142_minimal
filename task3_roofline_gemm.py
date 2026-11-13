"""Phase 3 - empirical roofline (3A) + progressive cache-blocked GEMM (3B)."""
import multiprocessing as mp
from array import array
from multiprocessing import shared_memory, util

from harness import run_benchmark, emit_verify_tag
from lab_common import STUDENT_ID, save_results, scale, hw, p_phys, attach_shm, log

BUF_BYTES = scale(64 * 10**6, 8 * 10**6)        # two 64 MB buffers = 128 MB total
FLOPS_PER_WORKER = scale(60_000_000, 3_000_000)
N = scale(256, 64)
BS_LIST = [8, 16, 32, 64, 96, 128]

_g = {}


def _release_worker_views():
    """Release memoryview exports before interpreter shutdown (avoids BufferError on exit)."""
    for k in ("A", "B", "C", "mva", "mvb"):
        v = _g.pop(k, None)
        if isinstance(v, memoryview) and k in ("A", "B", "C"):
            v.release()
    for k in [k for k in _g if k.startswith("shm")]:
        _g.pop(k).close()


# ----------------------------------------------------------------- Part 3A
def init_bw(name_a, name_b):
    _g["shmA"] = attach_shm(name_a)
    _g["shmB"] = attach_shm(name_b)
    _g["mva"] = _g["shmA"].buf
    _g["mvb"] = _g["shmB"].buf
    util.Finalize(None, _release_worker_views, exitpriority=10)


def copy_slice(rng):
    s, e = rng
    _g["mva"][s:e] = _g["mvb"][s:e]          # native memcpy: 1 read + 1 write per byte
    return e - s


def fma_worker(iters):
    a, b = 0.999999, 1e-6
    x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 0.1
    x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 0.2
    for _ in range(iters):                    # 16 FMAs x 2 FLOPs = 32 FLOPs / iteration
        x0 = x0 * a + b; x1 = x1 * a + b; x2 = x2 * a + b; x3 = x3 * a + b
        x4 = x4 * a + b; x5 = x5 * a + b; x6 = x6 * a + b; x7 = x7 * a + b
        x8 = x8 * a + b; x9 = x9 * a + b; x10 = x10 * a + b; x11 = x11 * a + b
        x12 = x12 * a + b; x13 = x13 * a + b; x14 = x14 * a + b; x15 = x15 * a + b
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15


def roofline(out, P):
    A = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    B = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    try:
        B.buf[:BUF_BYTES] = bytes(range(256)) * (BUF_BYTES // 256) + bytes(BUF_BYTES % 256)
        ranges = [(t * BUF_BYTES // P, (t + 1) * BUF_BYTES // P) for t in range(P)]
        with mp.Pool(P, initializer=init_bw, initargs=(A.name, B.name)) as pool:
            pool.map(copy_slice, ranges)      # touch pages once
            st = run_benchmark(lambda: pool.map(copy_slice, ranges, chunksize=1),
                               scale(2, 1), scale(7, 3))
        assert bytes(A.buf[:4096]) == bytes(B.buf[:4096])
    finally:
        for s in (A, B):
            s.close()
            s.unlink()
    bw = 2 * BUF_BYTES / ((st.median_ms / 1e3) * 1e9)
    log(out, f"=== Phase 3A: roofline (P_phys = {P}) ===")
    log(out, f"Copy {BUF_BYTES/1e6:.0f} MB (x2 buffers): median {st.median_ms:.3f} ms, "
             f"IQR {st.iqr_ms:.3f} ms  => BW_sustained = {bw:.3f} GB/s")

    iters = FLOPS_PER_WORKER // 32
    with mp.Pool(P) as pool:
        pool.map(fma_worker, [1000] * P)
        st2 = run_benchmark(lambda: pool.map(fma_worker, [iters] * P, chunksize=1), scale(1, 0), scale(5, 3))
    gflops = P * iters * 32 / ((st2.median_ms / 1e3) * 1e9)
    log(out, f"FMA loop {P} x {iters*32:.3g} FLOPs: median {st2.median_ms:.2f} ms  "
             f"=> P_sustained = {gflops:.4f} GFLOP/s")
    ridge = gflops / bw
    log(out, f"Ridge point I* = P/BW = {ridge:.6f} FLOP/Byte")
    return {"buf_bytes": BUF_BYTES, "bw_ms": st.median_ms, "bw_iqr": st.iqr_ms, "BW": bw,
            "fma_ms": st2.median_ms, "P_sust": gflops, "ridge": ridge}


# ----------------------------------------------------------------- Part 3B
def init_gemm(na, nb, nc, n):
    for k, nm in (("A", na), ("B", nb), ("C", nc)):
        shm = attach_shm(nm)
        _g["shm" + k] = shm
        _g[k] = shm.buf.cast("d")
    _g["N"] = n
    util.Finalize(None, _release_worker_views, exitpriority=10)


def v1_naive(r0, r1):
    A, B, C, n = _g["A"], _g["B"], _g["C"], _g["N"]
    for i in range(r0, r1):
        io = i * n
        for j in range(n):
            dot = 0.0
            for k in range(n):
                dot += A[io + k] * B[k * n + j]       # B walked with stride N
            C[io + j] = dot


def v2_ikj(r0, r1):
    A, B, C, n = _g["A"], _g["B"], _g["C"], _g["N"]
    for i in range(r0, r1):
        io = i * n
        crow = [0.0] * n
        for k in range(n):
            aik = A[io + k]                           # hoisted a_ik
            brow = B[k * n:(k + 1) * n]               # contiguous row slice (zero-copy view)
            for j in range(n):
                crow[j] += aik * brow[j]              # unit stride
        C[io:io + n] = array("d", crow)


def v3_blocked(r0, r1, bs):
    A, B, C, n = _g["A"], _g["B"], _g["C"], _g["N"]
    cloc = [[0.0] * n for _ in range(r1 - r0)]
    for ii in range(r0, r1, bs):
        imax = min(ii + bs, r1)
        for jj in range(0, n, bs):
            jmax = min(jj + bs, n)
            for kk in range(0, n, bs):
                kmax = min(kk + bs, n)
                for i in range(ii, imax):              # micro-kernel order i-k-j
                    crow = cloc[i - r0]
                    io = i * n
                    for k in range(kk, kmax):
                        aik = A[io + k]                # hoisted out of j-loop
                        brow = B[k * n + jj:k * n + jmax]
                        j = jj
                        for b in brow:
                            crow[j] += aik * b
                            j += 1
    for i in range(r0, r1):
        C[i * n:(i + 1) * n] = array("d", cloc[i - r0])


def task(args):
    variant, r0, r1, bs = args
    if variant == 1:
        v1_naive(r0, r1)
    elif variant == 2:
        v2_ikj(r0, r1)
    else:
        v3_blocked(r0, r1, bs)
    return r1 - r0


def gemm(out, P, roof):
    h = hw()
    nbytes = N * N * 8
    shms = [shared_memory.SharedMemory(create=True, size=nbytes) for _ in range(3)]
    A, B, C = (s.buf.cast("d") for s in shms)
    for idx in range(N * N):
        A[idx] = ((idx * 7) % 13) / 13.0 - 0.5
        B[idx] = ((idx * 11) % 17) / 17.0 - 0.5
    rows = [(t * N // P, (t + 1) * N // P) for t in range(P)]
    flops = 2 * N ** 3
    results = []
    try:
        with mp.Pool(P, initializer=init_gemm, initargs=(shms[0].name, shms[1].name, shms[2].name, N)) as pool:
            def bench(variant, bs=0):
                jobs = [(variant, r0, r1, bs) for r0, r1 in rows]
                st = run_benchmark(lambda: pool.map(task, jobs, chunksize=1), scale(1, 0), scale(5, 1))
                return st, flops / ((st.median_ms / 1e3) * 1e9)

            log(out, f"=== Phase 3B: GEMM N = {N}, P_phys = {P} ===")
            st, g = bench(1)
            ref1 = list(C)
            results.append({"name": "Variant 1 (naive i-j-k)", "bs": None, "ms": st.median_ms, "iqr": st.iqr_ms, "gflops": g})
            log(out, f"V1 naive ijk   : {st.median_ms:10.2f} ms  {g:.5f} GFLOP/s")
            st, g = bench(2)
            ref2 = list(C)
            err12 = max(abs(x - y) for x, y in zip(ref1, ref2))
            results.append({"name": "Variant 2 (i-k-j)", "bs": None, "ms": st.median_ms, "iqr": st.iqr_ms, "gflops": g})
            log(out, f"V2 ikj         : {st.median_ms:10.2f} ms  {g:.5f} GFLOP/s   max|V1-V2|={err12:.2e}")
            for bs in BS_LIST:
                for i in range(N * N):
                    C[i] = 0.0
                st, g = bench(3, bs)
                err = max(abs(x - y) for x, y in zip(C, ref2))
                ok = err < 1e-9
                results.append({"name": "Variant 3 (blocked)", "bs": bs, "ms": st.median_ms, "iqr": st.iqr_ms,
                                "gflops": g, "max_err": err, "ok": ok})
                log(out, f"V3 BS={bs:3d}     : {st.median_ms:10.2f} ms  {g:.5f} GFLOP/s   "
                         f"max|V3-V2|={err:.2e} {'OK' if ok else 'FAIL'}")
    finally:
        A.release(); B.release(); C.release()
        for s in shms:
            s.close()
            s.unlink()

    v3 = [r for r in results if r["bs"]]
    best = max(v3, key=lambda r: r["gflops"])
    bs_star = best["bs"]
    l1, l2 = h.get("l1d_bytes", 0), h.get("l2_bytes", 0)
    for r in v3:
        r["W_tile"] = 3 * r["bs"] ** 2 * 8
        r["fits"] = "L1d" if l1 and r["W_tile"] <= l1 else ("L2" if l2 and r["W_tile"] <= l2 else "> L2")

    BW, Ps = roof["BW"], roof["P_sust"]
    mapping = [
        {"name": "Variant 1", "I": 0.125, "gflops": results[0]["gflops"]},
        {"name": "Variant 2", "I": 0.125, "gflops": results[1]["gflops"]},
        {"name": f"Variant 3 (BS*={bs_star})", "I": bs_star / 12, "gflops": best["gflops"]},
    ]
    for m in mapping:
        m["attainable"] = min(Ps, m["I"] * BW)
        m["pct"] = 100 * m["gflops"] / m["attainable"]
        m["bound"] = "memory" if m["I"] < roof["ridge"] else "compute"
        log(out, f"{m['name']:22s} I={m['I']:.4f}  attainable={m['attainable']:.4f} GFLOP/s  "
                 f"achieved={m['gflops']:.5f}  ({m['pct']:.1f}%, {m['bound']}-bound region)")
    return {"N": N, "results": results, "bs_star": bs_star, "mapping": mapping}


def main():
    out = []
    P = p_phys()
    roof = roofline(out, P)
    g = gemm(out, P, roof)
    best = max((r for r in g["results"] if r["bs"]), key=lambda r: r["gflops"])
    tag = emit_verify_tag("PHASE3", STUDENT_ID, g["results"][0]["gflops"], best["gflops"])
    log(out, f"BS* = {g['bs_star']}  W_tile = {3*g['bs_star']**2*8} B")
    log(out, f"GFLOP/s V1={g['results'][0]['gflops']:.4f}  V3(BS*)={best['gflops']:.4f}")
    log(out, tag)
    save_results("phase3", {"P": P, "roofline": roof, **g, "verify": tag, "log": out})


if __name__ == "__main__":
    main()
