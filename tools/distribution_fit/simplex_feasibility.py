"""Long-double phase-I simplex for shared-mixture feasibility; verify raw output."""
import numpy as np


def phase_one(G, h, initial=None):
    dtype = np.longdouble
    G = np.asarray(G, dtype=dtype)
    h = np.asarray(h, dtype=dtype)
    n = G.shape[1]
    A = np.vstack((-G, np.ones(n), -np.ones(n)))
    b = np.r_[-h, dtype(1), dtype(-1)]
    scale = np.maximum(np.max(np.abs(A), axis=1), dtype(1))
    A /= scale[:, None]
    b /= scale
    m = len(b)
    D = np.zeros((m + 2, n + 2), dtype=dtype)
    D[:m, :n] = A
    D[:m, n] = -1
    D[:m, n + 1] = b
    D[m + 1, n] = 1
    basis = np.arange(n, n + m)
    nonbasis = np.r_[np.arange(n), -1]
    eps = dtype('1e-14')
    pivots = 0

    def pivot(r, s):
        nonlocal pivots
        value = D[r, s]
        col = D[:, s].copy()
        row = D[r, :].copy()
        D[:] -= np.outer(col, row) / value
        D[r, :] = row / value
        D[:, s] = -col / value
        D[r, s] = 1 / value
        basis[r], nonbasis[s] = nonbasis[s], basis[r]
        pivots += 1

    r = int(np.argmin(b))
    status = 'feasible tableau'
    if b[r] < -eps:
        pivot(r, n)
        for _ in range(20000):
            eligible = np.flatnonzero(D[m + 1, :n + 1] < -eps)
            if not len(eligible):
                break
            # Bland ordering makes degenerate pivots deterministic.
            s = int(min(eligible, key=lambda j: nonbasis[j]))
            eligible_rows = np.flatnonzero(D[:m, s] > eps)
            if not len(eligible_rows):
                status = 'phase-I unbounded or numerical failure'
                break
            ratios = D[eligible_rows, n + 1] / D[eligible_rows, s]
            best = ratios.min()
            tied = eligible_rows[np.abs(ratios - best) <= eps]
            r = int(min(tied, key=lambda i: basis[i]))
            pivot(r, s)
        else:
            status = 'pivot limit'
        if status == 'feasible tableau' and D[m + 1, n + 1] < -eps:
            status = 'positive phase-I residual; requires separate infeasibility proof'
    w = np.zeros(n, dtype=dtype)
    for i, variable in enumerate(basis):
        if 0 <= variable < n:
            w[variable] = D[i, n + 1]
    dual = np.zeros(m, dtype=dtype)
    for j, variable in enumerate(nonbasis):
        if n <= variable < n + m:
            dual[variable - n] = max(D[m + 1, j], dtype(0))
    dual = dual[:len(h)] / scale[:len(h)]
    if status == 'feasible tableau' and (w.min() < -1e-12 or abs(w.sum()-1)>1e-10 or np.min(G@w-h)<-1e-7):
        status = 'tableau rejected by raw constraints'
    history = [{'dual': dual.astype(float).tolist(), 'machine_epsilon': float(np.finfo(dtype).eps), 'pivots': pivots, 'phase_one_slack': float(-D[m + 1, n + 1]),
                'sum_weights': float(w.sum()), 'minimum_raw_slack': float(np.min(G @ w - h))}]
    return np.asarray(w, dtype=float), history, status
