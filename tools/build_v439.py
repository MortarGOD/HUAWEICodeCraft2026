"""Namespace the archived reference Attention closure; keep v435 Linear."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'docs/test分支审计-20260907/history-sources/5505f64e-rrs-fast-fp16-online-refresh-safe/paper-solutions/rrs-fast-fp16-online-refresh-safe/solution.py'
assert hashlib.sha256(REF.read_bytes()).hexdigest()=='3140ce9bc5d52a9e65d84601d82460ab50cde5c10ccb10b5a0cfc0201ec8723e'
parent=(ROOT/'solution/solution_v435.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest()=='cdf094ec4b3de9d5fe835ea085ab25a8f70c049a7f771308ec0a3834262917ba'
tree=ast.parse(REF.read_text());nodes={}
for node in tree.body:
    if isinstance(node,ast.FunctionDef):nodes[node.name]=node
    elif isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Name):nodes[target.id]=node
roots=['hif4_calibration_attention']+['hif4_dynamic_quantize_'+r for r in 'qkv']
needed=set(roots);pending=list(roots)
while pending:
    for node in ast.walk(nodes[pending.pop()]):
        if isinstance(node,ast.Name) and node.id in nodes and node.id not in needed:
            needed.add(node.id);pending.append(node.id)
assert not any(n in needed for n in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation'))
class Rename(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id in needed:node.id='_v439_ref_'+node.id
        return node
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if node.name in needed:node.name='_v439_ref_'+node.name
        return node
selected=[node for node in tree.body if any(node is nodes[n] for n in needed)]
module=ast.Module(body=copy.deepcopy(selected),type_ignores=[])
body=ast.unparse(ast.fix_missing_locations(Rename().visit(module)))
source=parent+'\n\nimport math\n\n'+body+'\n\n'
for name in roots:source+=f'_v439_parent_{name} = {name}\n'
source+='''
@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    if d % 64:
        return _v439_parent_hif4_calibration_attention(samples, qh, kh, d)
    result = _v439_ref_hif4_calibration_attention(samples, qh, kh, d)
    return {name: dict(state, _v439_reference=True) for name, state in result.items()}
'''
for r in 'qkv':
    name='hif4_dynamic_quantize_'+r
    source+=f'''
def {name}(quant, scale, heads, d, state):
    if isinstance(state, dict) and state.get('_v439_reference'):
        return _v439_ref_{name}(quant, scale, heads, d, state)
    return _v439_parent_{name}(quant, scale, heads, d, state)
'''
ast.parse(source)
(ROOT/'solution/solution_v439.py').write_text(source)
staging=ROOT/'.v439_self_check';staging.mkdir(exist_ok=True);(staging/'solution.py').write_text(source)
archive=ROOT/'solution/submission_v439.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    info=zipfile.ZipInfo('solution.py',date_time=(2026,9,8,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,source)
manifest={'version':439,'parent':435,'reference_path':str(REF.relative_to(ROOT)),
    'reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'closure_symbols':sorted(needed),
    'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
(ROOT/'docs/v439-artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
