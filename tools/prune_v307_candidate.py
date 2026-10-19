"""Emit an apply_patch for a conservatively pruned v308; never edit a parent.

Keeps all module assignments/imports. Only removes unreachable/overridden
functions and historical standalone string expressions. No score-based prune.
"""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'solution/solution_v307.py').read_text()
tree=ast.parse(source)
lines=source.splitlines(keepends=True)
functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
roots={n for n in functions if n.startswith('hif4_')}
# Conservatively retain callable references captured at module execution.
for node in tree.body:
    if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
        roots.update(n.id for n in ast.walk(node) if isinstance(n,ast.Name) and n.id in functions)
keep=set()
todo=list(roots)
while todo:
    name=todo.pop()
    if name in keep: continue
    keep.add(name)
    todo.extend(n.id for n in ast.walk(functions[name])
                if isinstance(n,ast.Name) and n.id in functions and n.id not in keep)
removed=[]
mask=set()
for node in tree.body:
    erase=False
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
        erase=node.name not in keep or node is not functions[node.name]
        if erase: removed.append({'name':node.name,'line':node.lineno})
    elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
        erase=True
    if erase:
        start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
        mask.update(range(start,node.end_lineno+1))
header='''"""v308: evidence-only cleanup of v307; complete GPTQ/refinement retained.

Historical overridden and unreachable functions are removed, never selected by
local score. Current six public APIs retain the v307 numerical algorithm.
"""
'''
result=header+''.join(line for i,line in enumerate(lines,1) if i not in mask)
print('*** Begin Patch')
print('*** Add File: '+str(ROOT/'solution/solution_v308.py'))
print('\n'.join('+'+line for line in result.splitlines()))
print('*** Add File: '+str(ROOT/'docs/v308-pruned-functions.json'))
print('\n'.join('+'+line for line in json.dumps({'retained_functions':sorted(keep),
    'removed_definitions':removed},indent=2).splitlines()))
print('*** End Patch')
