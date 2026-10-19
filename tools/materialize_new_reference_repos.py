#!/usr/bin/env python3
"""Fetch/hydrate explicitly listed fresh clones and materialize source-only HEAD.

No reference-repo code runs. APIs reusable by the coordinator:
  snapshot_fresh_clone(path)
  hydrate_source_archives(path, archives, record)
  materialize_clone(initial)
Source archives stay Git blobs; this tool never extracts archive members.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.refresh_reference_repos import (
    git_command, require_git, config_value, local_heads, remote_heads,
    hydrate_heads, check_objects, utc_now, write_json, progress,
)


def worktree_entries(path):
    return sorted(item.name for item in Path(path).iterdir() if item.name != '.git')


def snapshot_fresh_clone(path):
    path = Path(path).resolve()
    record = {'name':path.name, 'path':str(path), 'snapshot_at':utc_now(), 'commands':[]}
    record['non_git_entries'] = worktree_entries(path)
    filters = git_command(path, ['config','--get-regexp',r'^filter\..*\.(clean|smudge|process|required)$'],record)
    record['disabled_filters'] = sorted({'.'.join(line.split(' ',1)[0].split('.')[1:-1])
                                         for line in filters['stdout'].splitlines()})
    record['origin'] = require_git(path,['remote','get-url','origin'],record).strip()
    record['head'] = require_git(path,['rev-parse','HEAD'],record).strip()
    record['branch'] = git_command(path,['symbolic-ref','--quiet','--short','HEAD'],record)['stdout'].strip()
    record['upstream'] = git_command(path,['rev-parse','--symbolic-full-name','@{upstream}'],record)['stdout'].strip()
    record['index_paths'] = [x for x in require_git(path,['ls-files','-z'],record).split('\0') if x]
    record['fresh_empty_worktree_and_index'] = not record['non_git_entries'] and not record['index_paths']
    record['shallow'] = require_git(path,['rev-parse','--is-shallow-repository'],record).strip() == 'true'
    record['cached_heads'] = local_heads(path,record)
    record['settings'] = {key:config_value(path,key,record) for key in
                          ('remote.origin.fetch','remote.origin.promisor','remote.origin.partialclonefilter',
                           'core.sparseCheckout','core.sparseCheckoutCone')}
    return record


def hydrate_source_archives(path, archives, record):
    """Download identified solution/submission ZIP blobs, without extraction."""
    by_sha = {entry['sha']:dict(entry) for entry in archives}
    present, missing = check_objects(path,by_sha,record)
    initially_missing = list(missing)
    errors = []
    for start in range(0,len(missing),64):
        chunk = missing[start:start+64]
        progress(Path(path).name,f'hydrate_source_archives_{start+1}_{start+len(chunk)}')
        result = git_command(path,['-c','fetch.negotiationAlgorithm=noop','fetch','--filter=blob:none',
                                   '--no-tags','--no-write-fetch-head','--no-recurse-submodules','--stdin','origin'],
                             record,timeout=180,input_text='\n'.join(chunk)+'\n')
        if result['returncode']:
            errors.append({'hashes':chunk,'returncode':result['returncode'],'stderr':result['stderr'][:1500]})
    present, remaining = check_objects(path,by_sha,record)
    for sha,item in by_sha.items():
        item['available_locally'] = sha in present
        item['hydrated_by_this_tool'] = sha in initially_missing and sha in present
        item['size'] = present.get(sha,{}).get('size',item.get('size'))
    return {'unique_archive_count':len(by_sha), 'archive_path_count':sum(len(item['paths']) for item in by_sha.values()),
            'initially_missing_count':len(initially_missing), 'hydrated_count':len(initially_missing)-len(remaining),
            'all_archives_local':not remaining, 'total_bytes':sum(item.get('size') or 0 for item in by_sha.values()),
            'archives':list(by_sha.values()), 'missing':remaining,'errors':errors,
            'scope':'Only identified solution/submission ZIP blobs; no archive content executed or extracted.'}


def sparse_literal_path(path):
    if '\n' in path or '\r' in path:
        raise ValueError('Git sparse pattern cannot represent an embedded newline safely')
    # Non-cone patterns follow gitignore syntax. Anchor each selected file and
    # quote every metacharacter, including trailing/leading whitespace.
    return '/' + ''.join('\\'+char if char in '\\[]?!*# ' else char for char in path)


def tree_entries(path, head, record):
    output = require_git(path,['ls-tree','-r','-z',head],record,timeout=90)
    items = {}
    for value in output.split('\0'):
        if not value: continue
        meta,filename = value.split('\t',1)
        mode,kind,sha = meta.split()
        items[filename] = {'mode':mode,'type':kind,'sha':sha}
    return items


def materialize_clone(initial):
    path=Path(initial['path'])
    record={'name':initial['name'],'path':str(path),'started_at':utc_now(),'initial':initial,
            'disabled_filters':initial['disabled_filters'],'commands':[],'errors':[]}
    progress(path.name,'newclone_fetch_start')
    try:
        record['remote_heads_before_fetch']=remote_heads(path,record)
        require_git(path,['config','--local','--replace-all','remote.origin.fetch',
                          '+refs/heads/*:refs/remotes/origin/*'],record)
        args=['fetch','--no-recurse-submodules','--tags','origin','+refs/heads/*:refs/remotes/origin/*']
        if initial['shallow']: args.insert(1,'--unshallow')
        result=git_command(path,args,record,timeout=180)
        if result['returncode']:
            record['errors'].append({'stage':'fetch','returncode':result['returncode'],'stderr':result['stderr'][:1500]})
        heads=remote_heads(path,record);cached=local_heads(path,record)
        mismatch={name:{'expected':sha,'local':cached.get(name)} for name,sha in heads.items() if cached.get(name)!=sha}
        if mismatch:
            record['refetch_reason']=mismatch
            git_command(path,['fetch','--no-recurse-submodules','--tags','origin',
                              '+refs/heads/*:refs/remotes/origin/*'],record,timeout=180)
            heads=remote_heads(path,record);cached=local_heads(path,record)
            mismatch={name:{'expected':sha,'local':cached.get(name)} for name,sha in heads.items() if cached.get(name)!=sha}
        record['remote_heads']=heads;record['cached_heads']=cached;record['head_hash_mismatches']=mismatch
        old=initial['cached_heads']
        record['branch_changes']={'new':{name:sha for name,sha in heads.items() if name not in old},
            'changed':{name:{'old':old[name],'new':sha} for name,sha in heads.items() if name in old and old[name]!=sha},
            'deleted_remote_preserved_locally':{name:sha for name,sha in old.items() if name not in heads}}
        checks={}
        for branch,sha in heads.items():
            commit=git_command(path,['cat-file','-e',sha+'^{commit}'],record)
            tree=git_command(path,['cat-file','-e',sha+'^{tree}'],record)
            checks[branch]={'head':sha,'commit_local':commit['returncode']==0,'tree_local':tree['returncode']==0}
        record['branch_objects']=checks
        record['all_remote_heads_verified']=not mismatch and all(x['commit_local'] and x['tree_local'] for x in checks.values())
        record['hydration']=hydrate_heads(path,heads,record)
        record['archive_hydration']=hydrate_source_archives(path,record['hydration']['source_archive_candidates'],record)
        record['hydration']['source_archive_candidates']=record['archive_hydration']['archives']
        if not record['all_remote_heads_verified']:
            record['errors'].append({'stage':'head_verification','mismatches':mismatch})
        if not record['hydration']['all_selected_blobs_local']:
            record['errors'].append({'stage':'source_hydration','missing_count':len(record['hydration']['missing_blobs'])})
        if not record['archive_hydration']['all_archives_local']:
            record['errors'].append({'stage':'archive_hydration','missing_count':len(record['archive_hydration']['missing'])})
        current_entries=worktree_entries(path)
        current_index=[x for x in require_git(path,['ls-files','-z'],record).split('\0') if x]
        current_head=require_git(path,['rev-parse','HEAD'],record).strip()
        current_branch=git_command(path,['symbolic-ref','--quiet','--short','HEAD'],record)['stdout'].strip()
        checkout={'started':False,'initial_empty':initial['fresh_empty_worktree_and_index'],
                  'precheckout_non_git_entries':current_entries,'precheckout_index_count':len(current_index)}
        record['checkout']=checkout
        if not initial['fresh_empty_worktree_and_index'] or current_entries or current_index:
            checkout['action']='preserved_unexpected_worktree_or_index'
        elif current_head!=initial['head'] or current_branch!=initial['branch']:
            checkout['action']='preserved_head_changed_since_snapshot'
        elif not record['hydration']['all_selected_blobs_local'] or not record['all_remote_heads_verified']:
            checkout['action']='skipped_incomplete_source_or_refs'
        else:
            head=current_head;upstream=initial['upstream']
            if upstream.startswith('refs/remotes/origin/'):
                remote_name=upstream[len('refs/remotes/origin/'):]
                target=heads.get(remote_name)
                if target and target!=head:
                    ancestry=git_command(path,['merge-base','--is-ancestor',head,target],record)
                    if ancestry['returncode']==0:
                        require_git(path,['update-ref','refs/heads/'+current_branch,target,head],record)
                        checkout['fresh_branch_fast_forward']={'old':head,'new':target}
                        head=target
                    else:
                        checkout['local_head_preserved_non_fast_forward']=True
            matching=next((info for info in record['hydration']['branches'].values() if info['head']==head),None)
            if matching is None:
                supplemental=hydrate_heads(path,{'local-head':head},record)
                record['local_head_hydration']=supplemental
                matching=supplemental['branches']['local-head']
            selected=sorted({item['path'] for item in matching['selected_files']})
            patterns=[];omitted=[]
            for filename in selected:
                try: patterns.append(sparse_literal_path(filename))
                except ValueError as exc: omitted.append({'path':filename,'reason':str(exc)})
            if not patterns:
                checkout['action']='skipped_no_selected_source_files'
            else:
                sparse_path=Path(require_git(path,['rev-parse','--git-path','info/sparse-checkout'],record).strip())
                if not sparse_path.is_absolute(): sparse_path=path/sparse_path
                sparse_path.parent.mkdir(parents=True,exist_ok=True)
                content='\n'.join(patterns)+'\n';sparse_path.write_text(content)
                require_git(path,['config','--local','core.sparseCheckout','true'],record)
                require_git(path,['config','--local','core.sparseCheckoutCone','false'],record)
                checkout['started']=True
                done=git_command(path,['read-tree','-mu','HEAD'],record,timeout=180)
                checkout['action']='source_only_sparse_materialized' if done['returncode']==0 else 'read_tree_failed_preserved'
                checkout['pattern_sha256']=hashlib.sha256(content.encode()).hexdigest()
                checkout['selected_paths']=selected;checkout['omitted_paths']=omitted
                checkout['head']=head;checkout['selected_path_count']=len(selected)
                if done['returncode']:
                    record['errors'].append({'stage':'read_tree','stderr':done['stderr'][:1500]})
                else:
                    entries=tree_entries(path,head,record)
                    mismatches=[];actual_paths=[]
                    # os.walk does not follow symlinks. Hash symlink text, never
                    # the target, so untrusted links cannot read outside the repo.
                    for directory,dirs,files in os.walk(path,followlinks=False):
                        if Path(directory)==path and '.git' in dirs: dirs.remove('.git')
                        linked_dirs=[name for name in dirs if (Path(directory)/name).is_symlink()]
                        for name in linked_dirs: dirs.remove(name)
                        for name in files+linked_dirs:
                            file=Path(directory)/name;relative=file.relative_to(path).as_posix()
                            actual_paths.append(relative)
                            if relative not in entries or entries[relative]['type']!='blob':
                                mismatches.append({'path':relative,'reason':'unexpected_untracked_path'});continue
                            data=os.readlink(file).encode() if file.is_symlink() else file.read_bytes()
                            digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                            if digest!=entries[relative]['sha']:
                                mismatches.append({'path':relative,'expected':entries[relative]['sha'],'actual':digest})
                    missing=sorted(set(selected)-set(actual_paths)-{item['path'] for item in omitted})
                    extra=sorted(set(actual_paths)-set(selected))
                    checkout['actual_path_count']=len(actual_paths);checkout['missing_selected_paths']=missing
                    checkout['unexpected_paths']=extra;checkout['content_hash_mismatches']=mismatches
                    checkout['exact_selected_source_files']=not missing and not extra and not mismatches and not omitted
                    if not checkout['exact_selected_source_files']:
                        record['errors'].append({'stage':'checkout_verification','missing':missing,'extra':extra,'hash_mismatches':mismatches})
        record['final_head']=require_git(path,['rev-parse','HEAD'],record).strip()
        record['final_branch']=git_command(path,['branch','--show-current'],record)['stdout'].strip()
        record['final_shallow']=require_git(path,['rev-parse','--is-shallow-repository'],record).strip()=='true'
        record['full_commit_history']=not record['final_shallow']
        record['history_scope']='Complete commit topology; nonselected binary or historical blobs can remain promised.'
        record['final_status']=require_git(path,['status','--porcelain=v1','-z','--untracked-files=all'],record)
        record['final_settings']={key:config_value(path,key,record) for key in initial['settings']}
        record['partial_clone_settings_preserved']=all(record['final_settings'][key]==initial['settings'][key]
            for key in ('remote.origin.promisor','remote.origin.partialclonefilter'))
        record['completed']=not record['errors']
    except Exception as exc:
        record['errors'].append({'stage':'materialization_exception','message':str(exc)})
        record['completed']=False
    record['finished_at']=utc_now()
    progress(path.name,'newclone_complete' if record['completed'] else 'newclone_needs_review')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/papers/hif4-public-references'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo',action='append',required=True)
    parser.add_argument('--workers',type=int,default=3,choices=(1,2,3))
    args=parser.parse_args();names=list(dict.fromkeys(args.repo))
    if any(Path(name).name!=name or name in {'.','..'} for name in names):parser.error('repo must be a direct child name')
    snapshots=[snapshot_fresh_clone(args.root/name) for name in names]
    report={'started_at':utc_now(),'repo_names':names,'initial_snapshot':snapshots,'results':[],'completed':False,
            'scope':'Exactly the supplied new clones; no discovery or changes to previously existing mirrors.'}
    write_json(args.output,report);progress('newclones',f'initial_snapshot_saved_{len(snapshots)}')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(materialize_clone,item) for item in snapshots]
        for future in as_completed(futures):
            report['results'].append(future.result());report['results'].sort(key=lambda x:names.index(x['name']))
            write_json(args.output,report)
    report['completed']=all(x['completed'] for x in report['results'])
    report['all_remote_heads_verified']=all(x.get('all_remote_heads_verified') for x in report['results'])
    report['all_selected_source_blobs_local']=all(x.get('hydration',{}).get('all_selected_blobs_local') for x in report['results'])
    report['all_source_archives_local']=all(x.get('archive_hydration',{}).get('all_archives_local') for x in report['results'])
    report['finished_at']=utc_now()
    report['tool_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.output,report)


if __name__=='__main__':main()
