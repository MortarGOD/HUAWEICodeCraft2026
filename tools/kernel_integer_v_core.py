"""v392: replace kernel V feedback with bounded integer coordinate descent."""
_KC_ROUNDS = 4

def _kc_descent(x, y, step, p):
    # Channels are independent; each round moves one token per channel.
    t = x.shape[0]
    penalty = 1.0 / (t*t)
    initial_sse = (x-y).square().sum(0)
    cap = initial_sse * 1.05
    h = p.square().mean(0)[:, None] + penalty
    for _ in range(_KC_ROUNDS):
        e = x-y
        g = p.transpose(0, 1) @ (p @ e) / p.shape[0] + penalty*e
        budget = (cap-e.square().sum(0)).clamp_min(0)
        radius = (e.square()+budget[None]).sqrt()
        old = torch.round(y/step).clamp(-7, 7)
        lo = torch.minimum(torch.ceil((x-radius)/step).clamp(-7, 7), old)
        hi = torch.maximum(torch.floor((x+radius)/step).clamp(-7, 7), old)
        code = torch.maximum(lo, torch.minimum(hi, torch.round((y+g/h)/step)))
        delta = code*step-y
        change = h*delta.square()-2*g*delta
        value, index = change.min(0)
        columns = torch.arange(x.shape[1])
        y = y.clone()
        y[index, columns] += torch.where(value < 0, delta[index, columns], 0.)
    return y



def _vk_refine(x, p, state, kh, d):
    if x.shape[0] <= 1 or x.shape[0] > 4096 or state is None or not bool(state["active"].any()):
        return p
    probs = _vk_probabilities(x, state, kh)
    original = _deq_params(p).float()
    step = _params_unit_flat(p)*.25
    proposal = original.clone()
    for head in torch.where(state["active"])[0].tolist():
        sl = slice(head*d, (head+1)*d)
        proposal[:, sl] = _kc_descent(x[:, sl], original[:, sl], step[:, sl], probs[head])
    candidate = _values_to_params(proposal, p)
    actual = _deq_params(candidate).float()
    keep = torch.zeros(x.shape[1], dtype=torch.bool)
    for head in torch.where(state["active"])[0].tolist():
        sl = slice(head*d, (head+1)*d)
        e0, e1 = x[:, sl]-original[:, sl], x[:, sl]-actual[:, sl]
        mse0, mse1 = e0.square().mean(0), e1.square().mean(0)
        cost0 = (probs[head]@e0).square().mean(0)+mse0/x.shape[0]
        cost1 = (probs[head]@e1).square().mean(0)+mse1/x.shape[0]
        keep[sl] = (cost1 < cost0) & (mse1 <= mse0*_VK_MSE_CAP)
    mask = keep.reshape(1, -1, 8, 2, 4)
    return {key: torch.where(mask, candidate[key], value) if key in ("sign", "mant") else value
            for key, value in p.items()}
