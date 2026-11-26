"""Reweight the three original public checkpoints of the official leader."""
import torch


def reweight_layer_tensor(name, leader, spec, sources, *, third):
    prefix = 'model.language_model.layers.'
    if not name.startswith(prefix) or not leader.is_floating_point():
        return None
    layer = int(name.removeprefix(prefix).split('.', 1)[0])
    if not spec['first_layer'] <= layer <= spec['last_layer']:
        return None
    inputs = [source.get_tensor(name) for source in (*sources, third)]
    assert all(t.dtype == leader.dtype and t.shape == leader.shape for t in inputs)
    first, second, last = [tensor.float() for tensor in inputs]
    canonical = ((2 * first + 2 * second + last) / 5).to(leader.dtype)
    weights = spec['numerators']
    blend = ((weights[0] * first + weights[1] * second + weights[2] * last) / sum(weights)).to(leader.dtype)
    matches = canonical == leader
    return torch.where(matches, blend, leader), {'eligible': int(matches.sum()),
                                                 'protected_mismatch': int((~matches).sum())}
