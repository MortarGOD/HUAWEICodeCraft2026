import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parent=(ROOT/'solution/solution_v435.py').read_text()
parent_sha='cdf094ec4b3de9d5fe835ea085ab25a8f70c049a7f771308ec0a3834262917ba'
assert hashlib.sha256(parent.encode()).hexdigest()==parent_sha
a=parent.index('def _joint_quant_queries(');b=parent.index('\n\n',a)
source=parent[:a]+(ROOT/'tools/attention_nongptq_query_core.py').read_text().rstrip()+parent[b:]
old="if len(samples)<2 or d%64 or not parent['q_state'].get('gq') or not 4<=len(samples[-1]['v'][0])<=4096:return parent"
assert source.count(old)==1
source=source.replace(old,"if len(samples)<2 or d%64 or not 4<=len(samples[-1]['v'][0])<=4096:return parent")
ast.parse(source)
(ROOT/'solution/solution_v443.py').write_text(source)
staging=ROOT/'.v443_self_check';staging.mkdir(exist_ok=True)
(staging/'solution.py').write_text(source)
archive=ROOT/'solution/submission_v443.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    info=zipfile.ZipInfo('solution.py',date_time=(2026,9,8,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
    z.writestr(info,source)
manifest={'version':443,'parent':435,'parent_sha256':parent_sha,
    'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_members':['solution.py']}
(ROOT/'docs/v443-artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
