"""Audited benchmark-log extraction; unique versions, not submission-event count."""
import csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
lines=(ROOT/'benchmark-log.md').read_text().splitlines();records={}
for n,line in enumerate(lines,1):
 if not line.startswith('|'):continue
 c=[x.strip().replace('**','').replace('`','') for x in line.strip('|').split('|')]
 j=next((i for i,x in enumerate(c[:3]) if re.fullmatch(r'v\d+',x)),None)
 if j is None or int(c[j][1:])<275:continue
 v=int(c[j][1:]);si=3 if j==2 else next((i for i in (1,2,3) if i<len(c) and re.match(r'^-?\d+(?:\.\d+)?(?:$|[（；])',c[i])),None)
 if si is None:continue
 match=re.match(r'^-?\d+(?:\.\d+)?(?:$|[（；])',c[si])
 if not match:continue
 score=re.match(r'^-?\d+(?:\.\d+)?',c[si]).group()
 rawtime=c[6] if j==2 else c[si+1];tm=re.match(r'^\d+',rawtime)
 records[v]={'version':v,'score':score,'seconds':int(tm.group()) if tm else None,'status':'security_failure' if v in (275,300) else 'scored','evidence_line':n}
# These six receipts are in narrative headlines, while their tables remain pending.
for v,score,seconds in [(341,'20720',252),(343,'20803',248),(344,'20803',252),(345,'20805',255),(346,'20803.75',254),(368,'20803',244)]:
 pattern=rf'v{v}[^\n]{{0,20}}{re.escape(score)}(?:分)?/{seconds}'
 n=next(i for i,l in enumerate(lines,1) if re.search(pattern,l))
 records[v]={'version':v,'score':score,'seconds':seconds,'status':'scored','evidence_line':n}
for v in (282,369):
 n=next(i for i,l in enumerate(lines,1) if re.search(rf'v{v}\b',l) and ('线上反馈/超时' in l or '无有效得分回执' in l))
 records[v]={'version':v,'score':None,'seconds':None,'status':'timeout','evidence_line':n}
rows=[records[v] for v in sorted(records)]
counts={k:sum(r['status']==k for r in rows) for k in ('scored','security_failure','timeout')}
summary={'first_new_dataset_version':275,'last_reported_version':440,'unique_versions':len(rows),'counts':counts,'numeric_scores':sum(r['score'] is not None for r in rows),'limits':'Version-deduplicated receipts, not independent data cases or exact submission-event counts. Corrected v302 and v424 not counted twice. v441 blank excluded.'}
base=ROOT/'docs/online-receipts-new-dataset-20260908'
base.with_suffix('.json').write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2)+'\n')
with base.with_suffix('.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
text='# 新数据集线上反馈版本统计\n\n以benchmark-log明确记载的新集首次评测v275为起点；只统计本仓版本，不混入参考仓库。每版本取确认回执一次，待测/本地估分/更正前误记不计。\n\n'
text+=f"共 **{len(rows)}个有线上反馈的版本**：**{counts['scored']}个正常得分且有耗时**，{counts['security_failure']}个安全失败，{counts['timeout']}个超时。有数字分数的记录{summary['numeric_scores']}个，但两条安全失败不应用于精度拟合。\n\n"
text+='v275安全失败8303、耗时缺失；v300安全失败−30000、耗时缺失；v282/v369超时无有效得分。v440新回执21304/261已计入，v441留空不计。v302采用更正后的19392/281；此前误记不是另一独立回执。v423后一次误写经用户确认属v424。无法仅凭重复叙述还原真实提交次数。\n\n这些是版本级实验样本；同一线上数据集反复评测，不等于新增独立题目。\n\n|版本|得分|耗时/秒|状态|依据行|\n|---|---:|---:|---|---:|\n'
for r in rows:text+=f"|v{r['version']}|{r['score'] if r['score'] is not None else '—'}|{r['seconds'] if r['seconds'] is not None else '—'}|{r['status']}|{r['evidence_line']}|\n"
base.with_suffix('.md').write_text(text)
print(json.dumps(summary,ensure_ascii=False))
