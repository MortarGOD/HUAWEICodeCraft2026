import json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT=Path('.').resolve(); D=ROOT/'docs/distribution-fit'
keys=json.loads((D/'score-keys.json').read_text()); atoms=keys['atoms']
def complete(s):
    r,a=s.split('/');p=D/r/'cells'/f'v443--{a}.json'
    return p.exists() and json.loads(p.read_text()).get('complete')
tasks=[s.split('/') for s in atoms if not complete(s)]
print('v443 pending:',len(tasks),flush=True)
def run(t):
    rnd,atom=t
    try:
        p=subprocess.run([sys.executable,'tools/distribution_fit/evaluate_cell.py','--version','443','--atom',atom,'--round',rnd],cwd=str(ROOT),capture_output=True,timeout=180)
        return 'ok' if p.returncode==0 else 'err'
    except subprocess.TimeoutExpired: return 'timeout'
with ThreadPoolExecutor(max_workers=4) as ex:
    fs={ex.submit(run,t):t for t in tasks}
    for f in as_completed(fs):
        if f.result()!='ok': print(f.result(),fs[f],flush=True)
print('DONE443',flush=True)
