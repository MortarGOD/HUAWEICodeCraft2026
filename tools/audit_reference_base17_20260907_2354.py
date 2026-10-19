"""Read-only AST and package audit; never imports or runs reference code."""
import ast,hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'docs/papers/hif4-public-references/test'
def blob(commit,path):return subprocess.check_output(['git','-C',str(REPO),'show',f'{commit}:{path}'])
def norm(b):return b.replace(b'\r\n',b'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def index(b):
    tree=ast.parse(b);out={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):out.setdefault(node.name,[]).append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            for t in node.targets if isinstance(node,ast.Assign) else [node.target]:
                for n in ast.walk(t):
                    if isinstance(n,ast.Name):out.setdefault(n.id,[]).append(node)
    return out
old='c09ab6f2f1a5b0fef97d2957b95f12dd21deb4cd';new='d759c10b926b72dee7ddc98214dc17f7e37d2c53'
a=blob(old,'solution/solution.py');b=blob(new,'solution/solution.py');aa,bb=index(a),index(b)
def serial(nodes):return [ast.dump(n,include_attributes=False) for n in nodes]
changed=[k for k in sorted(set(aa)|set(bb)) if serial(aa.get(k,[]))!=serial(bb.get(k,[]))]
seen=set();todo=['hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation']
while todo:
    k=todo.pop()
    if k in seen:continue
    seen.add(k)
    for node in bb.get(k,[]):
        todo.extend(n.id for n in ast.walk(node) if isinstance(n,ast.Name) and n.id in bb and n.id not in seen)
report={'old':old,'new':new,'changed_top_level_bindings':changed,'linear_reachable_names':sorted(seen),'linear_reachable_changed':sorted(set(changed)&seen),'source_sha256':sha(b),'packages':{}}
assert not report['linear_reachable_changed']
for name in ('submission.zip','submission-attention-value-risk-v7.zip'):
    path=REPO/'solution'/name
    with zipfile.ZipFile(path) as z:
        names=z.namelist();assert names==['solution.py'];code=z.read('solution.py')
    assert norm(code)==norm(b)
    report['packages'][name]={'sha256':sha(path.read_bytes()),'entry_sha256':sha(code),'normalized_identical':True}
reproduction=blob(new,'solution/v419-baseline/solution.py');ours=(ROOT/'solution/solution_v419.py').read_bytes()
report['v419_reproduction']={'reference_sha256':sha(reproduction),'local_sha256':sha(ours),'normalized_identical':norm(reproduction)==norm(ours)}
assert report['v419_reproduction']['normalized_identical']
report['final_public_definitions']={name:[n.lineno for n in nodes] for name,nodes in bb.items() if name.startswith('hif4_') and isinstance(nodes[-1],ast.FunctionDef)}
(ROOT/'docs/test-base_1.7-update-20260907-2354.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
