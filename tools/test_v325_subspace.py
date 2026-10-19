"""Check variable-projection gradient and native candidate invariants."""
import torch
from probe_v325_subspace import candidates, projection
from probe_v319_value import ROOT, load_solution
from self_check import validate_hif4_params


def main():
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(325)
    b = torch.randn(12, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.randn(17, 12, generator=generator, dtype=torch.float64)
    z = torch.linalg.solve(b.T @ b, (y @ b).T).T
    r = y - z @ b.T
    analytic = -2 * r.detach().T @ z.detach()
    gradient, = torch.autograd.grad(r.square().sum(), b)
    assert torch.allclose(analytic, gradient, atol=1e-9, rtol=1e-9)
    m = load_solution(ROOT/'solution/solution_v324.py')
    original = torch.randn(128, 64, generator=generator)
    params = m._quantize_linear_values(original)
    saved = {k: v.clone() for k,v in params.items()}
    state = torch.random.get_rng_state().clone()
    results = candidates(m, dict(weight_params=params), original.T)
    assert len(results) == 4
    for _, p, history in results:
        assert not validate_hif4_params(p, original.shape, 'weight')
        assert all(torch.isfinite(torch.tensor(history)))
    assert all(torch.equal(params[k], saved[k]) for k in params)
    assert torch.equal(state, torch.random.get_rng_state())
    z, r = projection(y.detach(), b.detach())
    assert torch.isfinite(z).all() and torch.isfinite(r).all()
    print('PASS: analytic gradient, four legal finite candidates, input/RNG preservation')


if __name__ == '__main__':
    main()
