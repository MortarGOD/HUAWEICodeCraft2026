"""Resume the post-push v432/v433 banks, pressure, and campaign diagnostics."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def command(args, log):
    print('START', log, flush=True)
    with (ROOT/log).open('w') as output:
        subprocess.run([sys.executable,*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    print('DONE',log,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,default=0)
    args=parser.parse_args()
    if args.wait_pid:
        while True:
            try:os.kill(args.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(5)
    for v in (429,430,431):
        for label in ('paired','covariance','row_permute','block_resample'):
            assert json.loads((ROOT/f'docs/v{v}-attention-{label}.json').read_text())['completed']
    assert json.loads((ROOT/'docs/attention-proxy-audit-20260908.json').read_text())['completed']
    for v in (432,433):
        for label,bank,extra in (
            ('paired','local_fitted_models_attention_feedback_v2',[]),
            ('covariance','local_fitted_models_covariance_v1',[]),
            ('row_permute','local_fitted_models_attention_feedback_v2',['--groups','0,24,49','--isolated','--decoy-mode','row_permute']),
            ('block_resample','local_fitted_models_attention_feedback_v2',['--groups','0,24,49','--isolated','--decoy-mode','block_resample'])):
            stem=f'docs/v{v}-attention-{label}';path=ROOT/(stem+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                sha=hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest()
                if record['completed'] and record['source_sha256']==sha:
                    print('KEEP completed',stem,flush=True);continue
            command(['tools/verify_attention_campaign.py','--candidate',str(v),'--dataset',bank,'--output',stem+'.json',*extra],stem+'.log')
    command(['tools/diagnose_attention_concentration.py'],'docs/attention-concentration-diagnostic.log')
    command(['tools/check_attention_campaign_mha.py'],'docs/attention-campaign-mha.log')
    command(['tools/summarize_attention_campaign.py'],'docs/attention-campaign-summary.log')
    print('ALL FOLLOWUP DONE',flush=True)


if __name__=='__main__':main()
