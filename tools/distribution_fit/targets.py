"""Remove measured Linear changes without training version-specific score offsets."""
import collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
r=json.loads((D/'manifest.json').read_text());rows=r['rows'];by={x['version']:x for x in rows}
graph=collections.defaultdict(list);constraints=[]
def edge(a,b,delta,reason):
 sa,sb=by[a]['linear_signature'],by[b]['linear_signature'];graph[sa].append((sb,delta));graph[sb].append((sa,-delta));constraints.append({'from':a,'to':b,'linear_delta':delta,'reason':reason})
for x in rows:
 if not x['included']:
  assert x['attention_signature_equal_to_parent']
  p=x['excluded_parent'];edge(p,x['version'],float(x['score'])-float(by[p]['score']),'Measured Linear-only edge, matching Attention dependency fingerprint')
# Documented Linear retention across implementation cleanup; not fitted parameters.
edge(285,303,0,'benchmark-log v303 freezes v285 Linear')
edge(301,302,0,'benchmark-log v302 only replaces v301 Attention')
edge(317,318,0,'benchmark-log v318: Linear inherited; GPTQ/Attention performance change')
edge(360,362,0,'v362 modifies Attention Q-to-K path; Linear retained')
base=by[277]['linear_signature'];offset={base:0.};queue=[base];conflicts=[]
for a in queue:
 for b,d in graph[a]:
  wanted=offset[a]+d
  if b not in offset:offset[b]=wanted;queue.append(b)
  elif abs(offset[b]-wanted)>1e-8:conflicts.append([a,b,offset[b]-wanted])
selected=[]
for x in rows:
 if x['included']:
  o=offset.get(x['linear_signature']);selected.append(dict(x,linear_offset=o,adjusted_total=None if o is None else float(x['score'])-o))
# Research lineage, not chronological adjacency. Sources: candidate-dag.md,
# benchmark-log.md, version builders and per-version validation metadata.
parents={277:275,283:282,284:283,287:286,293:287,302:301,303:302,
305:303,306:304,307:306,308:307,309:308,310:309,311:310,312:311,
314:310,315:314,316:315,318:317,319:318,320:319,321:320,322:321,
323:322,325:324,326:324,330:324,333:324,335:324,337:324,
339:324,340:324,341:324,343:324,344:324,345:343,346:343,
358:353,360:324,362:324,363:324,364:324,365:324,367:324,
374:324,375:374,377:375,378:324,379:365,380:365,381:365,
382:381,383:381,384:382,385:382,386:382,387:382,
388:387,389:387,390:387,391:387,392:387,393:387,394:387,395:387,
396:387,397:387,398:387,399:387,400:387,401:400,402:400,
403:402,404:403,405:402,406:405,407:400,408:400,409:400,
410:400,411:400,412:400,413:412,414:412,415:412,416:412,
417:412,418:412,419:412,426:422,429:422,430:422,431:422,
432:422,433:422,434:422,435:430,436:435,437:435,438:435,439:435,440:435}
assert set(parents)=={x['version'] for x in selected}
edges=[];unavailable=[];unresolved=[]
for b in selected:
 v=b['version'];parent=parents[v];b['research_parent']=parent
 if parent not in by:
  unavailable.append({'version':v,'research_parent':parent,'reason':'No complete valid online parent receipt'})
  continue
 a=by[parent]
 controls=[x for x in selected if x['attention_signature']==a['attention_signature']]
 if not controls:
  unresolved.append(v);continue
 control=next((x for x in controls if x['version']==parent),controls[0])
 if a['linear_signature']==b['linear_signature']:correction=0.
 elif a['linear_signature'] in offset and b['linear_signature'] in offset:
  correction=offset[b['linear_signature']]-offset[a['linear_signature']]
 else:
  unresolved.append(v);continue
 raw=float(b['score'])-float(a['score']);delta=raw-correction
 edges.append({'from':control['version'],'research_parent':parent,'to':v,
 'parent_attention_proxy':control['version']!=parent,'online_total_delta':raw,
 'linear_delta':correction,'target':delta,'tolerance':max(1.,abs(raw)*.15),
 'direction_required':abs(delta)>1})
report={'versions':selected,'edges':edges,'linear_constraints':constraints,'conflicts':conflicts,
 'unresolved':unresolved,'unavailable_parent_receipts':unavailable,
 'limits':'Actual research-parent deltas only. Excluded Linear-only parents use fingerprint-equivalent Attention measurement controls and their actual online receipts. Missing effective parent receipts are separately marked; absolute Attention anchor remains unidentified.'}
assert not conflicts
path=D/'targets.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
print('retained',len(selected),'edges',len(edges),'unavailable',unavailable,'unresolved',unresolved)
