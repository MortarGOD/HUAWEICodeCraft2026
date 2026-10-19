"""Freeze receipts, exclusions, source identity and structural role signatures."""
import dis,hashlib,inspect,json,sys,types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
EXCLUDE={278:(277,'square ridge'),279:(278,'up encoder'),280:(279,'down projection'),281:(279,'Linear cross-block'),285:(283,'up fixed point'),296:(293,'down frame'),301:(285,'reference Linear'),304:(303,'down decoder'),317:(315,'Linear joint calibration'),324:(322,'Linear Gram performance'),368:(324,'square output encoding'),420:(419,'Linear prepared reuse'),421:(420,'Linear prefix search'),422:(421,'Linear reflection'),423:(422,'Linear pair search'),424:(422,'Linear Procrustes'),425:(422,'Linear shared scale'),427:(422,'Linear W/P joint rebuild')}
def code(c):
 return [c.co_code.hex(),[code(x) if isinstance(x,types.CodeType) else repr(x) for x in c.co_consts],c.co_names,c.co_varnames,c.co_freevars,c.co_cellvars,c.co_argcount,c.co_kwonlyargcount]
def fingerprint(m,roots,details=False):
 seen={}
 def walk(name,obj):
  if name in seen:return
  if inspect.isfunction(obj):obj=inspect.unwrap(obj)
  if inspect.isfunction(obj) and obj.__globals__ is m.__dict__:
   seen[name]={'code':code(obj.__code__),'defaults':repr(obj.__defaults__),'kwdefaults':repr(obj.__kwdefaults__)}
   def globals_used(c):
    result={i.argval for i in dis.get_instructions(c) if i.opname in ('LOAD_GLOBAL','LOAD_NAME')}
    for x in c.co_consts:
     if isinstance(x,types.CodeType):result.update(globals_used(x))
    return result
   for n in globals_used(obj.__code__):
    if n in m.__dict__:walk(n,m.__dict__[n])
  elif isinstance(obj,(str,int,float,bool,type(None),tuple,list,dict)):
   seen[name]=repr(obj)
  elif isinstance(obj,torch.Tensor):seen[name]=[str(obj.dtype),list(obj.shape),hashlib.sha256(obj.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()]
  elif isinstance(obj,types.ModuleType):seen[name]=['module',obj.__name__]
  else:seen[name]=['external',getattr(obj,'__module__',None),getattr(obj,'__qualname__',type(obj).__name__)]
 for n in roots:walk(n,getattr(m,n))
 return seen if details else hashlib.sha256(json.dumps(seen,sort_keys=True).encode()).hexdigest()
def main():
 out=ROOT/'docs/distribution-fit';out.mkdir(exist_ok=True)
 rows=json.loads((ROOT/'docs/online-receipts-new-dataset-20260908.json').read_text())['rows'];rows=[r for r in rows if r['status']=='scored']
 result=[];torch.set_num_threads(4)
 for r in rows:
  v=r['version'];p=ROOT/f'solution/solution_v{v}.py';m=load(v,'fit_inventory')
  extra={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'included':v not in EXCLUDE,
   'attention_signature':fingerprint(m,['hif4_calibration_attention','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k','hif4_dynamic_quantize_v']),
   'linear_signature':fingerprint(m,['hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation'])}
  if v in EXCLUDE:extra.update(excluded_parent=EXCLUDE[v][0],reason=EXCLUDE[v][1])
  result.append(dict(r,**extra));print(v,extra['included'],flush=True)
 lookup={r['version']:r for r in result}
 for r in result:
  if not r['included']:
   p=lookup.get(r['excluded_parent']);r['attention_signature_equal_to_parent']=p is not None and p['attention_signature']==r['attention_signature']
 report={'original_count':len(rows),'included_count':sum(r['included'] for r in result),'tolerance':{'relative_delta':.05,'absolute_points':1},'rows':result,'limits':'Structural signatures are screening evidence, not a proof for reflection/dynamic dispatch. Exclusions also require benchmark lineage; performance-only Attention variants remain included.'}
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('counts',report['included_count'])
if __name__=='__main__':main()
