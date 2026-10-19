"""Official mini self-check for each standalone output-risk version."""
import ast,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
from build_output_risk_versions import ROOT,candidate_source


def main():
    parent=ast.parse((ROOT/'solution/solution_v324.py').read_text()).body
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    rows=[]
    for version in (339,340):
        source=(ROOT/f'solution/solution_v{version}.py').read_text();assert source==candidate_source(version)
        tree=ast.parse(source)
        assert [ast.dump(n) for n in tree.body[1:len(parent)]]==[ast.dump(n) for n in parent[1:]]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):assert all(a.name.split('.')[0] not in banned for a in n.names)
            if isinstance(n,ast.ImportFrom):assert n.module.split('.')[0] not in banned
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in ('open','exec','eval','compile','__import__')
        with tempfile.TemporaryDirectory(prefix=f'hif4-v{version}-') as folder:
            Path(folder,'solution.py').write_text(source)
            r=subprocess.run([sys.executable,str(ROOT/'self_check.py'),'--solution_dir',folder,'--datasets_dir',str(ROOT/'mini_sample')],
                cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),capture_output=True,text=True,timeout=240)
        row=dict(version=version,sha256=hashlib.sha256(source.encode()).hexdigest(),passed=r.returncode==0,stdout=r.stdout,stderr=r.stderr)
        rows.append(row);print(version,row['passed'],flush=True);assert row['passed'],row
    (ROOT/'docs/v339-v340-self-check.json').write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
