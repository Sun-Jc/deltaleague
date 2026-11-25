"""Produce a registered routing artifact using only public Delta model downloads."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile

from delta_cli.artifacts import build_sparse, file_hash, verify_artifact_bytes
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from residual_edit import residual_tensor


def verified_path(models, version, content_hash):
    directory = models / version[:8]
    meta = json.loads((directory / 'delta-model.json').read_text())
    assert meta['id'] == version and meta['content_hash'] == content_hash
    verify_artifact_bytes(directory, meta['files'])
    return directory / 'model.safetensors'


def build(models, protocol, arm, output):
    if output.exists():
        raise FileExistsError(f'Choose a new destination: {output}')
    rule = json.loads(protocol.read_text())
    spec = rule['arms'][arm]
    expected = json.loads((protocol.parent / 'results/builds.json').read_text())[arm]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    parent_file = verified_path(models, rule['parent'], rule['parent_hash'])
    donor_file = verified_path(models, rule['donor'], rule['donor_hash'])
    with tempfile.TemporaryDirectory(dir=output.parent) as temp, ExitStack() as stack:
        def open_model(path):
            return stack.enter_context(safe_open(path, framework='pt'))

        parent, donor = open_model(parent_file), open_model(donor_file)
        ancestor = None
        if 'ancestor' in rule:
            ancestor = open_model(verified_path(models, rule['ancestor'], rule['ancestor_hash']))
        tensors = {name: parent.get_tensor(name) for name in parent.keys()}
        prefix = 'model.language_model.layers.'
        for name, old in tensors.items():
            if ancestor is not None:
                result = residual_tensor(name, old, spec, (ancestor, donor))
                if result is not None:
                    tensors[name] = result[0]
            elif name.startswith(prefix):
                layer, module = name.removeprefix(prefix).split('.', 1)
                if spec['first_layer'] <= int(layer) <= spec['last_layer'] and module in spec['modules']:
                    tensors[name] = donor.get_tensor(name)
        modified = Path(temp) / 'model.safetensors'
        save_file(tensors, modified, metadata=parent.metadata())
        artifact = Path(temp) / 'submission'
        build_sparse(parent_file, modified, artifact, rule['parent'], rule['parent_hash'], 'qwen35-08b-v1')
        for filename, key in (('weights.safetensors', 'patch_sha256'), ('manifest.json', 'manifest_sha256')):
            assert file_hash(artifact / filename) == expected[key]
        artifact.rename(output)
    print(expected['patch_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('arm')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.models, args.protocol, args.arm, args.output)
