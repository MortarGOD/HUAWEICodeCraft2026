"""Analytic tests only; artificial matrices are not a performance dataset."""
import torch
from probe_linear_deployed_joint import residual_update, choose_guard, tree_equal


def main():
    generator = torch.Generator().manual_seed(317)
    for rows, width, outputs in ((3, 7, 9), (8, 3, 10), (5, 5, 7)):
        z = torch.randn(rows, width, generator=generator, dtype=torch.float64)
        b = torch.randn(outputs, width, generator=generator, dtype=torch.float64)
        y = torch.randn(rows, outputs, generator=generator, dtype=torch.float64)
        saved = [v.clone() for v in (z, y, b)]
        delta, relative = residual_update(z, y, b, .1)
        lam = .1 * (z @ z.T).diagonal().mean()
        expected = torch.linalg.solve(z.T @ z + lam * torch.eye(width, dtype=z.dtype), z.T @ (y-z@b.T)).T
        assert torch.allclose(delta, expected, atol=1e-10, rtol=1e-10)
        assert relative < 1e-12
        assert (y - z @ (b+delta).T).square().sum() <= (y-z@b.T).square().sum()
        assert tree_equal(saved, [z, y, b])
    # One improved fold cannot overrule degradation of the other fold.
    rows = [dict(loss=1., folds=[1.,1.]), dict(loss=.9, folds=[.7,1.1]),
            dict(loss=.95, folds=[.96,.94])]
    assert choose_guard(rows) == 2
    assert choose_guard(rows[:2]) == 0
    assert choose_guard([rows[0], rows[0]]) == 0
    # If latent Z=A P, the exact continuous decoder is W P^{-T}.
    p = torch.tensor([[1., .3], [.1, 1.2]], dtype=torch.float64)
    w = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    decoder = torch.linalg.solve(p, w.T).T
    assert torch.allclose(decoder @ p.T, w, atol=1e-12, rtol=1e-12)
    zero = torch.zeros(2, 3, dtype=torch.float64)
    delta, relative = residual_update(zero, torch.ones(2, 4, dtype=torch.float64),
                                     torch.zeros(4, 3, dtype=torch.float64), .1)
    assert torch.isfinite(delta).all() and delta.count_nonzero() == 0 and relative == 0
    print('PASS: dual/primal equivalence, residual descent, zero input, immutability, two-fold guard')


if __name__ == '__main__':
    main()
