"""Build source-matched routing grafts and run paired public evaluations on Metal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DEFAULT_HELPERS = Path('/Users/jc/.codex/worktrees/leader96-mixture/deltaleague/research/Sun-Jc-leader96-mixture')


def configure(protocol):
    helpers = Path(os.environ.get('DELTA_MIXTURE_STUDY', DEFAULT_HELPERS))
    for name, digest in protocol['pinned_helpers'].items():
        if hashlib.sha256((helpers / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Unexpected shared runner: ' + name)
    sys.path.insert(0, str(helpers))


def build_models(root, rule):
    import torch
    from build import build_candidate, verified, write
    from safetensors import safe_open

    torch.set_num_threads(2)
    parent, donor = [root.parent / 'models' / rule[key][:8] for key in ('parent', 'donor')]
    meta = verified(parent, rule['parent_hash'])
    assert meta['id'] == rule['parent'] and verified(donor, rule['donor_hash'])['id'] == rule['donor']
    prior_plan = root.parent / 'leaderd012-transfer-v1/attention-transfer/plan.json'
    template = json.loads(prior_plan.read_text())
    assert template['generation']['seed'] == rule['development']['seed']
    reports = {}
    for name, spec in rule['arms'].items():
        report = build_candidate(root, name, spec, parent, meta, (parent, donor), template)
        with safe_open(donor / 'model.safetensors', framework='pt') as src, safe_open(root / name / 'model/model.safetensors', framework='pt') as out:
            assert all(torch.equal(src.get_tensor(k), out.get_tensor(k)) for k in report['by_tensor'])
        reports[name] = report
        print(name, report['changed_values'], report['patch_sha256'], flush=True)
    write(root / 'builds.json', reports)


def evaluate(root, rule):
    from evaluate_ablation import compare
    from evaluate_round import check_versions, native, native_text, previous, prior

    check_versions()
    parent_run = root.parent / 'leaderd012-transfer-v1'
    parent_arm = 'attention-transfer'
    (root / 'records').mkdir(exist_ok=True)
    dev = prior.suite_from(root.parent / 'leader-round2-v1/general/suite.json', rule['development']['suite_sha256'])
    conf = prior.suite_from(root.parent / 'ten-hour-confirmation-v1/suite.json', rule['confirmation']['suite_sha256'])
    selection = native.read(root / 'fresh-selection.json')
    fresh = prior.suite_from(root / 'fresh-suite.json', selection['suite_sha256'])
    dev_ids, conf_ids, fresh_ids = map(native.expected_ids, (dev, conf, fresh))
    assert len(dev_ids) == len(conf_ids) == 18 and all(dev_ids[g].isdisjoint(conf_ids[g]) for g in dev_ids)
    assert all(fresh_ids[g].isdisjoint(dev_ids[g] | conf_ids[g]) for g in previous.FOCUS)
    original = native_text.evaluate_tasks

    def evaluate_long_requests(**kwargs):
        kwargs['model'].timeout = 1800
        return original(**kwargs)

    def broad(arm, phase, suite):
        assert 'Manifest and artifact hashes are valid' in (root / arm / 'validation.log').read_text()
        return compare(root, arm, phase, suite, rule[phase]['seed'], parent_run, parent_arm)

    with patch.object(native_text, 'evaluate_tasks', evaluate_long_requests):
        development = {arm: broad(arm, 'development', dev) for arm in rule['arms']}
        ranked = sorted(development, key=lambda arm: previous.rank(development[arm]), reverse=True)
        shortlist = ranked[:2]
        previous.record(root, 'selection', {'ordered': shortlist, 'ranking': rule['ranking']})
        confirmation = {}
        for arm in shortlist:
            # The shared runner uses one confirmation directory per run.
            # Give each candidate an independent run root on the same T9 filesystem.
            arm_root = root.parent / (root.name + '-confirm-' + arm)
            arm_root.mkdir(exist_ok=True)
            (arm_root / 'records').mkdir(exist_ok=True)
            arm_dir = arm_root / arm
            arm_dir.mkdir(exist_ok=True)
            for filename in ('plan.json', 'build.json', 'validation.log'):
                source = root / arm / filename
                target = arm_dir / filename
                if not target.exists():
                    target.write_bytes(source.read_bytes())
                assert source.read_bytes() == target.read_bytes()
            summary = compare(arm_root, arm, 'confirmation', conf, rule['confirmation']['seed'], parent_run, parent_arm)
            record_name = 'confirmation-' + arm
            previous.record(root, record_name, native.read(arm_root / 'records' / (record_name + '.json')))
            confirmation[arm] = summary
        eligible = [arm for arm in shortlist if development[arm]['delta'] > 0 and confirmation[arm]['delta'] > 0]
        previous.record(root, 'fresh-candidates', {'ordered': eligible, 'fresh_suite_sha256': selection['suite_sha256']})
        scores = {}
        reference = None
        for arm in eligible:
            scores[arm] = previous.fresh_result(root, arm, fresh, rule['fresh']['seed'], reference)
            if reference is None:
                reference = root / ('fresh-' + arm)
        suitable = [arm for arm in eligible if all(scores[arm][g]['wins'] > scores[arm][g]['losses'] for g in previous.FOCUS)]
        previous.record(root, 'decision', dict(development=development, confirmation=confirmation, fresh=scores,
                                               qualifying=suitable, nominate=suitable[0] if suitable else None))
        native.note(root, 'complete', suitable[0] if suitable else 'none', 'all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'fresh', 'evaluate'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    args = parser.parse_args()
    rule = json.loads(args.protocol.read_text())
    configure(rule)
    root = args.output.resolve()
    root.mkdir(exist_ok=True)
    if args.action == 'build':
        build_models(root, rule)
    elif args.action == 'fresh':
        from fresh_panel import prepare
        prepare(root, rule)
    else:
        evaluate(root, rule)
