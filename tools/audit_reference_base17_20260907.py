"""Read-only audit of the fixed base_1.7 update; never execute donor code."""
from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import math
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'docs/papers/hif4-public-references/test'
BEFORE = '52b490459d117c7982d2ed7d14cbe791a756c193'
AFTER = 'c09ab6f2f1a5b0fef97d2957b95f12dd21deb4cd'
OUTPUT = ROOT / 'docs/test-base_1.7-update-20260907-1736.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REF), *args])


def source_functions(data):
    tree = ast.parse(data.decode('utf-8-sig'))
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return tree, {n.name: n for n in functions}, dict(Counter(n.name for n in functions))


def structural(node):
    return ast.dump(node, include_attributes=False)


def function_closure(functions, entries):
    """Static direct Name-call closure, not a proof about dynamic dispatch."""
    todo, seen = list(entries), set()
    while todo:
        name = todo.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        todo.extend(n.func.id for n in ast.walk(functions[name])
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in functions)
    return seen


def default_block_moment_matches(old_node, new_node):
    """Substitute the unchanged Linear call default, without running the function."""
    node = copy.deepcopy(new_node)
    index = next(i for i, arg in enumerate(node.args.kwonlyargs) if arg.arg == 'block_size')
    assert isinstance(node.args.kw_defaults[index], ast.Constant)
    assert node.args.kw_defaults[index].value == 64
    del node.args.kwonlyargs[index]
    del node.args.kw_defaults[index]

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, name):
            return ast.copy_location(ast.Constant(value=64), name) if name.id == 'block_size' else name

    return structural(old_node) == structural(Substitute().visit(node))


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def loss_difference_check():
    """Cauchy-Schwarz bound on real-vector losses, using binary64 diagnostics.

    Approximate residual e and change d; unknown shared error f and change
    error g have norms <= b and t. This does not model BF16 scoring rounding.
    """
    rng = random.Random(202609071736)
    minimum_margin = float('inf')
    for _ in range(400):
        vectors = [[rng.gauss(0, scale) for _ in range(16)]
                   for scale in (1.0, 0.15, 0.07, 0.01)]
        e, d, f, g = vectors
        a, c, b, t = map(norm, vectors)
        approximate_decrease = -2 * dot(e, d) - dot(d, d)
        base = [x + y for x, y in zip(e, f)]
        candidate = [x + y + z for x, y, z in zip(base, d, g)]
        actual_decrease = dot(base, base) - dot(candidate, candidate)
        bound = 2 * (a + c) * t + 2 * b * (c + t) + t * t
        margin = actual_decrease - (approximate_decrease - bound)
        assert margin >= -1e-12
        minimum_margin = min(minimum_margin, margin)
    # A small update rejected by separately bounded loss intervals can be
    # accepted after accounting for the shared baseline compression error.
    a, c, b, t = 1.0, 0.01, 0.1, 0.001
    decrease = 1.0 - 0.99 ** 2
    bound = 2 * (a + c) * t + 2 * b * (c + t) + t * t
    separate_accepts = 0.99 + b + t < a - b
    difference_accepts = decrease > bound
    assert not separate_accepts and difference_accepts
    return {'random_cases': 400, 'passed': True, 'minimum_lower_bound_margin': minimum_margin,
            'shared_error_example': {'separate_accepts': separate_accepts,
                                     'difference_accepts': difference_accepts,
                                     'approximate_decrease': decrease, 'difference_bound': bound},
            'scope': 'Binary64 vector diagnostics of the analytic bound, not donor execution, HiF4, BF16 scoring, or online validation.'}


def main():
    assert git('rev-parse', 'HEAD').decode().strip() == AFTER
    assert not git('status', '--porcelain=v1').strip()
    old = git('show', BEFORE + ':solution/solution.py')
    new = git('show', AFTER + ':solution/solution.py')
    assert new == (REF / 'solution/solution.py').read_bytes()
    old_tree, old_fn, old_counts = source_functions(old)
    new_tree, new_fn, new_counts = source_functions(new)
    changes = []
    for name in sorted(old_fn.keys() | new_fn.keys()):
        kind = ('added' if name not in old_fn else 'removed' if name not in new_fn
                else 'changed' if structural(old_fn[name]) != structural(new_fn[name]) else 'same')
        if kind != 'same':
            node = new_fn.get(name, old_fn.get(name))
            changes.append({'name': name, 'kind': kind, 'start': node.lineno, 'end': node.end_lineno})
    entries = ('hif4_calibration_and_quantize_weight', 'hif4_dynamic_quantize_activation')
    old_closure, new_closure = function_closure(old_fn, entries), function_closure(new_fn, entries)
    changed_names = {item['name'] for item in changes}
    moment_calls = [{'line': n.lineno, 'call': ast.unparse(n)} for n in ast.walk(new_tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == '_block_second_moment']
    report = {'before': BEFORE, 'after': AFTER,
              'commits': git('log', '--format=%H %s', BEFORE + '..' + AFTER).decode().splitlines(),
              'source_sha256': {'before': sha(old), 'after': sha(new)},
              'duplicate_top_level_function_names': {k: v for k, v in new_counts.items() if v > 1},
              'function_changes': changes,
              'linear_static_call_scope': {
                  'entrypoints_unchanged': all(structural(old_fn[n]) == structural(new_fn[n]) for n in entries),
                  'old_function_count': len(old_closure), 'new_function_count': len(new_closure),
                  'closure_names_same': old_closure == new_closure,
                  'changed_reachable_functions': sorted(changed_names & (old_closure | new_closure)),
                  'block_moment_default_substitution_ast_equal': default_block_moment_matches(
                      old_fn['_block_second_moment'], new_fn['_block_second_moment']),
                  'caveat': 'Static direct Name-call closure only; shared block moment default semantics inspected separately.'},
              'block_second_moment_calls': moment_calls,
              'imports': {label: [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
                          for label, tree in [('before', old_tree), ('after', new_tree)]},
              'archives': {}, 'dataset_hash_checks': {},
              'graph': {'project': 'Users-ee-Code-2026-huawei-alogorithm', 'generation': '2026-09-07T09:00:46Z',
                        'reference_freshness': 'metadata_changed', 'root_v422_v419_freshness': 'metadata_match',
                        'fallback': 'Direct source for changed reference functions and actual call routes; ZIPs read as data.'}}
    for name in ('submission.zip', 'submission-attention-native-v4.zip', 'submission-attention-medium-v5.zip'):
        data = git('show', AFTER + ':solution/' + name)
        assert data == (REF / 'solution' / name).read_bytes()
        with zipfile.ZipFile(REF / 'solution' / name) as archive:
            assert archive.namelist() == ['solution.py'] and archive.testzip() is None
            member = archive.read('solution.py')
            report['archives'][name] = {'zip_sha256': sha(data), 'source_sha256': sha(member),
                'source_bytes': len(member), 'matches_worktree_bytes': member == new,
                'matches_worktree_normalized_newlines': member.replace(b'\r\n', b'\n') == new.replace(b'\r\n', b'\n'),
                'matches_worktree_ast': structural(ast.parse(member.decode('utf-8-sig'))) == structural(new_tree)}
    for bank in ('public_quantizer', 'v419_mini_sample'):
        manifest = json.loads((REF / 'datasets' / bank / 'manifest.json').read_text())
        checked = {}
        for operator, info in manifest['files'].items():
            data = (REF / 'datasets' / bank / info['file']).read_bytes()
            actual = sha(data)
            assert actual == info['sha256'] and len(data) == info['bytes']
            item = {'sha256': actual, 'bytes': len(data), 'matches_manifest': True}
            if bank == 'v419_mini_sample':
                item['identical_to_root_mini_sample'] = data == (ROOT / 'mini_sample' / info['file']).read_bytes()
                assert item['identical_to_root_mini_sample']
            checked[operator] = item
        report['dataset_hash_checks'][bank] = checked
    matrix = json.loads((REF / 'tests/benchmarks/attention-medium-v5/matrix.json').read_text())
    report['donor_recorded_package'] = matrix['package']
    report['recorded_package_identity'] = {
        'archive_digest_matches': report['archives']['submission.zip']['zip_sha256'] == matrix['package']['zip_sha256'],
        'archived_source_digest_matches': report['archives']['submission.zip']['source_sha256'] == matrix['package']['source_sha256'],
        'worktree_digest_matches_recorded_prototype': sha(new) == matrix['package']['prototype_sha256'],
        'note': 'The recorded prototype digest is not the current Git source identity; current source and ZIP are compared directly above.'}
    assert report['recorded_package_identity']['archive_digest_matches']
    assert report['recorded_package_identity']['archived_source_digest_matches']
    report['donor_recorded_v5_original_attention'] = {
        'not_reexecuted': True,
        **{key: sum(d[key] for d in matrix['datasets'].values()) for key in ('cases', 'wins', 'ties', 'regressions')}}
    report['compression_difference_math'] = loss_difference_check()
    report['project_source_sha256'] = {name: sha((ROOT / 'solution' / name).read_bytes())
        for name in ('solution.py', 'solution_v419.py', 'solution_v422.py', 'solution_v424.py')}
    report['donor_code_executed'] = False
    report['complete'] = True
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'complete': True,
                      'source_sha256': report['source_sha256'],
                      'linear_static_call_scope': report['linear_static_call_scope'],
                      'donor_recorded_v5_original_attention': report['donor_recorded_v5_original_attention'],
                      'compression_difference_math': report['compression_difference_math']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
