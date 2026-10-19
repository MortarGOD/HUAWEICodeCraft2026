#!/usr/bin/env python3
"""Safely refresh an explicit snapshot of reference repos without executing code.

Reusable API: snapshot_repo(path), refresh_repo(snapshot), and
hydrate_heads(path, heads, record). Each record owns a commands list. Git object
checks disable lazy fetching; missing selected source blobs are fetched explicitly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

PRINT_LOCK = threading.Lock()
SOURCE_EXTENSIONS = {
    '.py', '.pyi', '.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx',
    '.cu', '.cuh', '.rs', '.go', '.java', '.js', '.jsx', '.ts', '.tsx', '.m',
    '.mm', '.swift', '.f', '.f90', '.f95', '.jl', '.r', '.R', '.sh', '.bash',
    '.pyx', '.pxd', '.inl', '.tpp', '.inc', '.bzl', '.bazel', '.mk', '.kt',
    '.kts', '.scala', '.pl', '.pm', '.rb', '.php', '.hs', '.lua', '.scm',
    '.lisp', '.clj', '.cljs', '.cljc', '.metal', '.wgsl', '.glsl', '.vert', '.frag',
    '.zsh', '.ps1', '.bat', '.cmake', '.proto', '.sql', '.s', '.S', '.asm',
}
TEXT_EXTENSIONS = {'.md', '.rst', '.txt', '.toml', '.yaml', '.yml', '.json',
                   '.jsonl', '.ini', '.cfg', '.conf', '.xml', '.html', '.css',
                   '.scss', '.ipynb', '.lock', '.in', '.map', '.tex', '.bib'}
SOURCE_BASENAMES = {'makefile', 'dockerfile', 'cmakelists.txt', 'license', 'copying',
                    'notice', 'readme', 'requirements', 'setup.cfg', '.gitignore', 'build', 'build.bazel', '.clang-format',
                    '.gitattributes', '.gitmodules', '.editorconfig'}
DATA_PATH_PARTS = {'data', 'datasets', 'dataset', 'model_weights', 'checkpoints', 'weights',
                   'data_cache', 'calibration_data', 'sample_data', 'test_data'}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def progress(name, stage):
    with PRINT_LOCK:
        print(json.dumps({'time': utc_now(), 'repo': name, 'stage': stage}), flush=True)


def git_command(path, args, record=None, timeout=30, input_text=None, no_lazy=True):
    path = Path(path)
    configs = ['core.hooksPath=/dev/null', 'core.fsmonitor=false',
               'filter.lfs.smudge=', 'filter.lfs.process=', 'filter.lfs.required=false',
               'merge.autoStash=false', 'protocol.ext.allow=never']
    if record:
        for name in record.get('disabled_filters', []):
            configs.extend([f'filter.{name}.clean=', f'filter.{name}.smudge=',
                            f'filter.{name}.process=', f'filter.{name}.required=false'])
    command = ['git', '--no-pager', '-C', str(path)]
    for config in configs:
        command.extend(['-c', config])
    command.extend(args)
    environment = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1',
                       GIT_OPTIONAL_LOCKS='0')
    if no_lazy:
        environment['GIT_NO_LAZY_FETCH'] = '1'
    started = time.monotonic()
    try:
        result = subprocess.run(command, input=None if input_text is None else input_text.encode(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=environment, timeout=timeout, check=False)
        output, error = result.stdout.decode('utf-8', 'replace'), result.stderr.decode('utf-8', 'replace')
        answer = {'returncode': result.returncode, 'stdout': output, 'stderr': error,
                  'elapsed_seconds': round(time.monotonic()-started, 3), 'timeout': False}
    except subprocess.TimeoutExpired as exc:
        answer = {'returncode': 124, 'stdout': (exc.stdout or b'').decode('utf-8', 'replace'),
                  'stderr': (exc.stderr or b'').decode('utf-8', 'replace'),
                  'elapsed_seconds': round(time.monotonic()-started, 3), 'timeout': True}
    if record is not None:
        entry = {'args': list(args), 'returncode': answer['returncode'],
                 'elapsed_seconds': answer['elapsed_seconds'], 'timeout': answer['timeout'],
                 'stdout_preview': answer['stdout'][:1200], 'stderr_preview': answer['stderr'][:3000]}
        if input_text is not None:
            entry['stdin_lines'] = len(input_text.splitlines())
        record.setdefault('commands', []).append(entry)
    return answer


def require_git(path, args, record, timeout=30, input_text=None):
    result = git_command(path, args, record, timeout, input_text)
    if result['returncode']:
        raise RuntimeError(f"git {args[0]} failed ({result['returncode']}): {result['stderr'][:1000]}")
    return result['stdout']


def config_value(path, key, record):
    result = git_command(path, ['config', '--local', '--get-all', key], record)
    return result['stdout'].splitlines() if result['returncode'] == 0 else []


def local_heads(path, record):
    output = require_git(path, ['for-each-ref', '--format=%(refname) %(objectname)', 'refs/remotes/origin/'], record)
    return {ref[len('refs/remotes/origin/'):]: sha for ref, sha in
            (line.split(' ', 1) for line in output.splitlines()) if ref != 'refs/remotes/origin/HEAD'}


def working_state(path, record):
    status = require_git(path, ['status', '--porcelain=v1', '-z', '--untracked-files=all'], record)
    diff = require_git(path, ['diff', '--no-ext-diff', '--no-textconv', '--no-renames', '--binary', 'HEAD', '--'], record)
    staged = require_git(path, ['diff', '--cached', '--no-ext-diff', '--no-textconv', '--no-renames', '--binary', '--'], record)
    return {'porcelain_z': status, 'status_entries': [s for s in status.split('\0') if s],
            'clean': not status, 'diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
            'diff_bytes': len(diff.encode()), 'staged_diff_sha256': hashlib.sha256(staged.encode()).hexdigest()}


def snapshot_repo(path):
    path = Path(path).resolve()
    record = {'name': path.name, 'path': str(path), 'snapshot_at': utc_now(), 'commands': []}
    filters = git_command(path, ['config', '--get-regexp', r'^filter\..*\.(clean|smudge|process|required)$'], record)
    record['disabled_filters'] = sorted({'.'.join(line.split(' ',1)[0].split('.')[1:-1]) for line in filters['stdout'].splitlines()})
    record['origin'] = require_git(path, ['remote', 'get-url', 'origin'], record).strip()
    record['head'] = require_git(path, ['rev-parse', 'HEAD'], record).strip()
    branch = git_command(path, ['symbolic-ref', '--quiet', '--short', 'HEAD'], record)
    record['branch'] = branch['stdout'].strip() if branch['returncode'] == 0 else None
    upstream = git_command(path, ['rev-parse', '--symbolic-full-name', '@{upstream}'], record)
    record['upstream'] = upstream['stdout'].strip() if upstream['returncode'] == 0 else None
    record['shallow'] = require_git(path, ['rev-parse', '--is-shallow-repository'], record).strip() == 'true'
    record['old_cached_heads'] = local_heads(path, record)
    record['working_state'] = working_state(path, record)
    record['settings'] = {key: config_value(path, key, record) for key in
                          ('remote.origin.fetch', 'remote.origin.promisor', 'remote.origin.partialclonefilter',
                           'core.sparseCheckout', 'core.sparseCheckoutCone', 'extensions.partialClone')}
    sparse_path = git_command(path, ['rev-parse', '--git-path', 'info/sparse-checkout'], record)['stdout'].strip()
    sparse = Path(sparse_path)
    if not sparse.is_absolute():
        sparse = path / sparse
    record['sparse_patterns_sha256'] = hashlib.sha256(sparse.read_bytes()).hexdigest() if sparse.is_file() else None
    return record


def remote_heads(path, record):
    output = require_git(path, ['ls-remote', '--heads', 'origin'], record, timeout=60)
    return {ref[len('refs/heads/'):]: sha for sha, ref in (line.split('\t') for line in output.splitlines())}


def check_objects(path, hashes, record):
    hashes = sorted(set(hashes))
    if not hashes:
        return {}, []
    output = require_git(path, ['cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                         record, timeout=60, input_text='\n'.join(hashes)+'\n')
    present, missing = {}, []
    for requested, line in zip(hashes, output.splitlines()):
        fields = line.split()
        if len(fields) == 3 and fields[1] != 'missing':
            present[requested] = {'type': fields[1], 'size': int(fields[2])}
        else:
            missing.append(requested)
    if len(output.splitlines()) < len(hashes):
        missing.extend(hashes[len(output.splitlines()):])
    return present, missing


def classify_blob(path, size):
    item = Path(path)
    extension = item.suffix
    if extension in SOURCE_EXTENSIONS:
        return 'source', None
    lower_name = item.name.lower()
    archive_formats = (('.tar.gz', 'tar.gz'), ('.tar.bz2', 'tar.bz2'),
                       ('.tar.xz', 'tar.xz'), ('.zip', 'zip'), ('.tar', 'tar'),
                       ('.tgz', 'tgz'), ('.tbz2', 'tbz2'), ('.txz', 'txz'), ('.7z', '7z'))
    for suffix, archive_format in archive_formats:
        if lower_name.endswith(suffix):
            # Naming is only a hint. Every archive must remain visible for
            # review because probe/code bundles need not be called solution.
            return 'archive', archive_format
    if extension.lower() in TEXT_EXTENSIONS or item.name.lower() in SOURCE_BASENAMES:
        if any(part.lower() in DATA_PATH_PARTS for part in item.parts) and extension.lower() in {'.json', '.jsonl', '.txt'}:
            return 'excluded_text', 'dataset_payload_path'
        if size is not None and size > 8 * 1024 * 1024:
            return 'excluded_text', 'text_blob_over_8_mib_review_required'
        return 'text', None
    return 'other', None


def hydrate_heads(path, heads, record):
    """Hydrate selected source/doc/config blobs for every fixed remote head.

    Binary data, model weights and archives are not implicitly downloaded.
    Source files are included regardless of size; dataset JSON/text and oversized
    prose/config blobs are explicitly listed for later review.
    """
    path = Path(path)
    eligible, archives, excluded, per_branch = {}, {}, [], {}
    for branch, sha in sorted(heads.items()):
        listing = require_git(path, ['ls-tree', '-r', '-z', sha], record, timeout=90)
        # Never request ls-tree -l: missing promisor blob sizes can cause a
        # download of every binary dataset or a BAD size with lazy fetch off.
        entries = []
        for line in (item for item in listing.split('\0') if item):
            metadata, filename = line.split('\t', 1)
            mode, kind, object_id = metadata.split()
            if kind == 'blob': entries.append((object_id, filename))
        local_objects, _ = check_objects(path, [item[0] for item in entries], record)
        branch_files = []
        for object_id, filename in entries:
            size = local_objects.get(object_id, {}).get('size')
            classification, reason = classify_blob(filename, size)
            if classification in {'source', 'text'}:
                value = eligible.setdefault(object_id, {'sha':object_id, 'size':size, 'paths':[], 'branches':[]})
                if filename not in value['paths']: value['paths'].append(filename)
                if branch not in value['branches']: value['branches'].append(branch)
                branch_files.append({'path':filename, 'sha':object_id, 'size':size, 'kind':classification})
            elif classification == 'archive':
                source_hint = any(word in Path(filename).name.lower() for word in ('solution', 'submission', 'submit'))
                value = archives.setdefault(object_id, {'sha':object_id, 'size':size, 'paths':[], 'branches':[],
                    'format':reason, 'source_name_hint':False, 'review_status':'requires_review'})
                value['source_name_hint'] = value['source_name_hint'] or source_hint
                if value['source_name_hint']: value['review_status'] = 'source_name_hint_unverified_contents'
                if filename not in value['paths']: value['paths'].append(filename)
                if branch not in value['branches']: value['branches'].append(branch)
            elif classification == 'excluded_text':
                excluded.append({'branch':branch, 'path':filename, 'sha':object_id, 'size':size, 'reason':reason})
        per_branch[branch] = {'head':sha, 'selected_files':branch_files}
    present, missing = check_objects(path, eligible, record)
    originally_missing = list(missing)
    fetch_errors = []
    for start in range(0, len(missing), 64):
        chunk = missing[start:start+64]
        progress(path.name, f'hydrate_source_blobs_{start+1}_{start+len(chunk)}')
        # Match Git's lazy-fetch negotiation: advertising local filtered
        # commits as HAVE can otherwise omit wanted blobs or require missing
        # delta bases. Explicit stdin requests fetch only selected objects.
        result = git_command(path, ['-c', 'fetch.negotiationAlgorithm=noop', 'fetch', '--filter=blob:none',
                                    '--no-tags', '--no-write-fetch-head', '--no-recurse-submodules',
                                    '--stdin', 'origin'], record, timeout=180, input_text='\n'.join(chunk)+'\n')
        if result['returncode']:
            fetch_errors.append({'hashes':chunk, 'returncode':result['returncode'], 'error':result['stderr'][:1000]})
    present, still_missing = check_objects(path, eligible, record)
    for sha, info in eligible.items():
        if sha in present: info['size'] = present[sha]['size']
    for info in per_branch.values():
        for item in info['selected_files']:
            if item['sha'] in present: item['size'] = present[item['sha']]['size']
    archive_present, _ = check_objects(path, archives, record)
    for sha, info in archives.items():
        info['available_locally'] = sha in archive_present
        info['hydrated_by_this_tool'] = False
    for branch, info in per_branch.items():
        info['selected_file_count'] = len(info['selected_files'])
        info['missing_selected_paths'] = [item['path'] for item in info['selected_files'] if item['sha'] in still_missing]
        info['all_selected_blobs_local'] = not info['missing_selected_paths']
    return {'selected_blob_count':len(eligible), 'selected_total_bytes':sum(item['size'] or 0 for item in eligible.values()),
            'initially_missing_count':len(originally_missing), 'hydrated_count':len(originally_missing)-len(still_missing),
            'verified_local_blob_count':len(present), 'all_selected_blobs_local':not still_missing,
            'missing_blobs':[eligible[sha] for sha in still_missing], 'fetch_errors':fetch_errors,
            'branches':per_branch, 'archive_candidates':list(archives.values()),
            # Retain the old narrower field for callers that automatically
            # hydrate known submission names; unclassified data archives must
            # be reviewed through archive_candidates before such downloads.
            'source_archive_candidates':[item for item in archives.values() if item['source_name_hint']],
            'archive_candidate_scope':'All ZIP/tar/tgz/7z bundles, including names without solution/submission; contents require review.',
            'excluded_text_files':excluded,
            'scope':'Source files plus docs/config at every current remote head; binary datasets and archives excluded. Local text blobs over 8 MiB excluded; missing promisor text size is known only after selected fetch.'}


def refresh_repo(initial):
    path = Path(initial['path'])
    record = {'name':initial['name'], 'path':str(path), 'started_at':utc_now(),
              'initial':initial, 'commands':[], 'disabled_filters':initial.get('disabled_filters', []),
              'errors':[]}
    progress(path.name, 'fetch_all_heads_start')
    try:
        record['remote_heads_before_fetch'] = remote_heads(path, record)
        require_git(path, ['config', '--local', '--replace-all', 'remote.origin.fetch',
                           '+refs/heads/*:refs/remotes/origin/*'], record)
        args = ['fetch', '--no-recurse-submodules', '--tags', 'origin', '+refs/heads/*:refs/remotes/origin/*']
        if initial['shallow']:
            args.insert(1, '--unshallow')
        fetched = git_command(path, args, record, timeout=180)
        if fetched['returncode']:
            record['errors'].append({'stage':'initial_fetch', 'returncode':fetched['returncode'], 'message':fetched['stderr'][:1500]})
            if initial['shallow']:
                fetched = git_command(path, ['fetch', '--no-recurse-submodules', '--tags', 'origin',
                                             '+refs/heads/*:refs/remotes/origin/*'], record, timeout=120)
        latest = remote_heads(path, record)
        cached = local_heads(path, record)
        mismatch = {name:{'expected':sha,'local':cached.get(name)} for name,sha in latest.items() if cached.get(name) != sha}
        if mismatch:
            record['refetch_reason'] = mismatch
            git_command(path, ['fetch', '--no-recurse-submodules', '--tags', 'origin',
                               '+refs/heads/*:refs/remotes/origin/*'], record, timeout=180)
            latest = remote_heads(path, record)
            cached = local_heads(path, record)
            mismatch = {name:{'expected':sha,'local':cached.get(name)} for name,sha in latest.items() if cached.get(name) != sha}
        record['remote_heads'] = latest
        record['cached_heads_after_fetch'] = cached
        record['head_hash_mismatches'] = mismatch
        if mismatch: record['errors'].append({'stage':'remote_head_verification','mismatches':mismatch})
        old = initial['old_cached_heads']
        record['branch_changes'] = {'new':{name:sha for name,sha in latest.items() if name not in old},
            'changed':{name:{'old':old[name],'new':sha} for name,sha in latest.items() if name in old and old[name] != sha},
            'deleted_remote_preserved_locally':{name:sha for name,sha in old.items() if name not in latest}}
        object_checks = {}
        for branch, sha in latest.items():
            commit = git_command(path, ['cat-file', '-e', sha+'^{commit}'], record)
            tree = git_command(path, ['cat-file', '-e', sha+'^{tree}'], record)
            object_checks[branch] = {'head':sha, 'commit_local':commit['returncode']==0, 'tree_local':tree['returncode']==0}
        record['branch_objects'] = object_checks
        record['changed_branch_files'] = {}
        for name, pair in record['branch_changes']['changed'].items():
            diff = git_command(path, ['diff', '--no-ext-diff', '--no-textconv', '--no-renames', '--name-status', pair['old'], pair['new'], '--'], record)
            record['changed_branch_files'][name] = {'old':pair['old'], 'new':pair['new'],
                'files':diff['stdout'].splitlines(), 'available':diff['returncode']==0}
        record['hydration'] = hydrate_heads(path, latest, record)
        if not record['hydration']['all_selected_blobs_local']:
            record['errors'].append({'stage':'source_hydration','missing_count':len(record['hydration']['missing_blobs'])})
        current = working_state(path, record)
        current_branch = git_command(path, ['symbolic-ref', '--quiet', '--short', 'HEAD'], record)['stdout'].strip()
        upstream = initial.get('upstream')
        if not initial['working_state']['clean']:
            record['worktree_action'] = 'fetch_only_initial_worktree_dirty'
        elif not current['clean'] or current_branch != initial.get('branch'):
            record['worktree_action'] = 'preserved_worktree_changed_since_snapshot'
        elif path.name == 'test' and current_branch != 'base_1.7':
            record['worktree_action'] = 'preserved_unexpected_test_branch'
        elif not upstream or not upstream.startswith('refs/remotes/origin/'):
            record['worktree_action'] = 'fetch_only_no_origin_tracking_branch'
        elif upstream[len('refs/remotes/origin/'):] not in latest:
            record['worktree_action'] = 'fetch_only_upstream_deleted'
        else:
            target = latest[upstream[len('refs/remotes/origin/'):]]
            ancestor = git_command(path, ['merge-base', '--is-ancestor', 'HEAD', target], record)
            if ancestor['returncode'] == 0:
                merge = git_command(path, ['merge', '--ff-only', '--no-edit', '--no-verify', target], record, timeout=120)
                record['worktree_action'] = 'fast_forward_or_already_current' if merge['returncode']==0 else 'fast_forward_failed_preserved'
                if merge['returncode']: record['errors'].append({'stage':'fast_forward','message':merge['stderr'][:1500]})
            else:
                record['worktree_action'] = 'fetch_only_local_head_not_ancestor'
        record['final_head'] = require_git(path, ['rev-parse', 'HEAD'], record).strip()
        record['final_branch'] = git_command(path, ['symbolic-ref', '--quiet', '--short', 'HEAD'], record)['stdout'].strip()
        record['final_working_state'] = working_state(path, record)
        record['final_shallow'] = require_git(path, ['rev-parse','--is-shallow-repository'], record).strip() == 'true'
        record['full_history_fetched'] = not record['final_shallow']
        record['history_scope'] = 'Full commit topology when non-shallow; partial clones may still omit nonselected historical blobs.'
        record['final_settings'] = {key:config_value(path,key,record) for key in initial['settings']}
        preserved_keys = set(initial['settings']) - {'remote.origin.fetch'}
        sparse_path = git_command(path, ['rev-parse', '--git-path', 'info/sparse-checkout'], record)['stdout'].strip()
        sparse = Path(sparse_path)
        if not sparse.is_absolute(): sparse = path / sparse
        record['final_sparse_patterns_sha256'] = hashlib.sha256(sparse.read_bytes()).hexdigest() if sparse.is_file() else None
        record['sparse_and_partial_settings_preserved'] = (all(initial['settings'][key] == record['final_settings'][key] for key in preserved_keys)
            and record['final_sparse_patterns_sha256'] == initial['sparse_patterns_sha256'])
        if not initial['working_state']['clean']:
            record['dirty_worktree_preserved'] = (record['final_head']==initial['head']
                and record['final_branch']==initial['branch'] and record['final_working_state']==initial['working_state'])
            if not record['dirty_worktree_preserved']:
                record['errors'].append({'stage':'dirty_worktree_preservation','message':'Initial and final state differ; no reset attempted.'})
        record['commit_count_all_local_refs'] = require_git(path, ['rev-list','--all','--count'], record).strip()
        record['completed'] = True
        record['all_remote_heads_verified'] = not mismatch and all(v['commit_local'] and v['tree_local'] for v in object_checks.values())
    except Exception as exc:
        record['errors'].append({'stage':'refresh_exception','message':str(exc)})
        record['completed'] = False
    record['finished_at'] = utc_now()
    progress(path.name, 'complete' if record['completed'] else 'failed')
    return record


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/papers/hif4-public-references'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', action='append', required=True, help='Explicit repo directory name; frozen before networking.')
    parser.add_argument('--workers', type=int, default=3, choices=(1,2,3))
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    names = list(dict.fromkeys(args.repo))
    if any(Path(name).name != name or name in {'.','..'} for name in names):
        parser.error('--repo must name a single child directory')
    snapshot = [snapshot_repo(args.root/name) for name in names]
    report = {'started_at':utc_now(), 'scope':'Explicit existing-repository snapshot; no discovery after network begins.',
              'repo_count':len(snapshot), 'repo_names':names, 'initial_snapshot':snapshot,
              'results':[], 'completed':False}
    write_json(args.output, report)
    progress('all', f'initial_snapshot_saved_{len(snapshot)}_repos')
    if args.snapshot_only:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(refresh_repo, item):item['name'] for item in snapshot}
        for future in as_completed(futures):
            report['results'].append(future.result())
            report['results'].sort(key=lambda item:names.index(item['name']))
            write_json(args.output, report)
    report['completed'] = all(item.get('completed') for item in report['results'])
    report['all_remote_heads_verified'] = all(item.get('all_remote_heads_verified') for item in report['results'])
    report['all_selected_source_blobs_local'] = all(item.get('hydration',{}).get('all_selected_blobs_local') for item in report['results'])
    report['finished_at'] = utc_now()
    write_json(args.output, report)


if __name__ == '__main__':
    main()
