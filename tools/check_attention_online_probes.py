"""Offline official self_check and static audit for each actual probe file."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from build_attention_online_probes import ROOT,MODES,candidate_source


def main():
    rows=[]
    parent=ast.parse((ROOT/'solution/solution_v324.py').read_text()).body
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for version,mode in MODES.items():
        source=(ROOT/f'solution/solution_v{version}.py').read_text()
        assert source==candidate_source(version)
        tree=ast.parse(source)
        assert [ast.dump(n) for n in parent[1:]]==[ast.dump(n) for n in tree.body[1:len(parent)]]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert not any(n.name.split('.')[0] in banned for n in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
        with tempfile.TemporaryDirectory(prefix=f'hif4-v{version}-') as folder:
            # Mechanical staging for the unchanged official CLI, never part of submission.
            Path(folder,'solution.py').write_text(source)
            result=subprocess.run([sys.executable,str(ROOT/'self_check.py'),'--solution_dir',folder,
                '--datasets_dir',str(ROOT/'mini_sample')],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                capture_output=True,text=True,timeout=240)
        row=dict(version=version,mode=mode,sha256=hashlib.sha256(source.encode()).hexdigest(),
                 passed=result.returncode==0,stdout=result.stdout,stderr=result.stderr)
        rows.append(row);print(version,row['passed'],flush=True)
        assert row['passed'],row
    (ROOT/'docs/v326-v338-self-check.json').write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
