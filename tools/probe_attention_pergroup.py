"""Per-group attention means for several solutions on one bank (ranking only)."""
import io, contextlib, os, sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, evaluate_attention

torch.set_num_threads(int(os.environ.get('EVAL_THREADS', '4')))
data = sys.argv[1]
sol_paths = sys.argv[2:]
groups = torch.load(data, map_location='cpu', weights_only=False)
res = {}
for sp in sol_paths:
    sol = load_solution(Path(sp).resolve())
    with contextlib.redirect_stdout(io.StringIO()):
        sc, _ = evaluate_attention(sol, groups)
    per = [sum(sc[i*5:(i+1)*5])/5 for i in range(len(groups))]
    res[Path(sp).stem] = per
    print('%-22s mean=%.6f' % (Path(sp).stem, sum(sc)/len(sc)))
print()
base = res[Path(sol_paths[0]).stem]
print('%-6s %10s %s' % ('group', 'base', ' '.join('%10s' % Path(s).stem for s in sol_paths[1:])))
for i in range(len(groups)):
    cells = []
    for s in sol_paths[1:]:
        d = res[Path(s).stem][i] - base[i]
        cells.append('%+10.5f' % d)
    print('%-6d %10.5f %s' % (i, base[i], ' '.join(cells)))
