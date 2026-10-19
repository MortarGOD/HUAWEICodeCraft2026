"""Summarize complete fixed-bank experiments without promoting unverified tiers."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'docs/gptq-integer-campaign'
rows=[]
for f in out.glob('v*.json'):
 r=json.loads(f.read_text())
 if 'conditional_online_estimate' in r:rows.append(r)
rows.sort(key=lambda r:r['version'])
text='# 固定原始整数权重 GPTQ 实验\n\n固定250整数计数，父v441线上21308；条件估分=21308+固定bank实测Attention增量，不保证线上结果。达到一档交付一档；每档须完整评估及格式自检。v491线上21274，与原预测方向相反；本表不代表线上承诺。\n\n| 实验 | Attention增量 | 条件估分 | 达到的估分档位 |\n|---|---:|---:|---|\n'
for r in rows:text+=f'| v{r["version"]} | {r["raw_delta"]:.3f} | {r["conditional_online_estimate"]:.3f} | {r["thresholds_reached"]} |\n'
text+='\noracle诊断仅用于误差归因，不进入提交。v446非均匀格点探针与445整数码相同，排除重复路线。数据和整数权重未修改。\n'
text+='\n接线审计：450/452/459/466等V动态包装可能同时影响父校准，不能解释为纯动态消融；470/471使用最终标记总门控，group15验证除新增字段外与441校准相等。\n'
(out/'progress.md').write_text(text)
print(text)
