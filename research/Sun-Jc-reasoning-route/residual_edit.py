"""Compare individual seed updates against the fixed three-source leader."""
from delta_cli.artifacts import verify_artifact_bytes
import json
import torch


def seed_sources(root, rule):
    ancestor = root.parent / 'top-eff1e1b7/model'
    meta = json.loads((ancestor / 'delta-model.json').read_text())
    assert meta['id'] == rule['ancestor'] and meta['content_hash'] == rule['ancestor_hash']
    verify_artifact_bytes(ancestor, meta['files'])
    return ancestor, root.parent / 'models' / rule['donor'][:8]


def residual_tensor(name, leader, spec, sources):
    prefix = 'model.language_model.layers.'
    if not name.startswith(prefix) or leader.ndim != 2:
        return None
    number, module = name.removeprefix(prefix).split('.', 1)
    if not spec['first_layer'] <= int(number) <= spec['last_layer']:
        return None
    if not (module.endswith('_proj.weight') or module.startswith('linear_attn.in_proj_') and module.endswith('.weight')):
        return None
    ancestor, donor = (file.get_tensor(name) for file in sources)
    assert leader.dtype == ancestor.dtype == donor.dtype and leader.shape == ancestor.shape == donor.shape
    parent = leader.float()
    old = ancestor.float()
    seed_delta = donor.float() - old
    family = spec['family']
    if family == 'missing-seed':
        eligible = (leader == ancestor) & (seed_delta != 0)
        changed = torch.where(eligible, donor, leader)
    elif family == 'opposed-rollback':
        eligible = (parent - old) * seed_delta < 0
        changed = torch.where(eligible, ancestor, leader)
    elif family == 'scaled-seed':
        eligible = seed_delta != 0
        changed = (parent + spec['factor'] * seed_delta).to(leader.dtype)
    else:
        raise ValueError(f'Unsupported seed edit: {family}')
    return changed, {'eligible': int(eligible.sum())}
