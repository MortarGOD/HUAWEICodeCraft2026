"""Build 13 standalone online ablations; no runtime module or file loading."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODES={326:'q_sharp',327:'both_shrink',328:'q_flat',329:'both_flat',
       330:'key_sorted',331:'query_sorted',332:'joint_sorted',333:'cross_pair_mix',
       334:'balance_0.5',335:'balance_1.0',336:'balance_-1.0',337:'whiten_q',338:'whiten_k'}


class Unbind(ast.NodeTransformer):
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='m':
            return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
        return self.generic_visit(node)


def extract(path,names,unbind=False):
    source=(ROOT/path).read_text();result=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            if unbind:
                assert node.args.args[0].arg=='m'
                node.args.args=node.args.args[1:]
                node=Unbind().visit(node);ast.fix_missing_locations(node)
                result.append(ast.unparse(node))
            else:result.append(ast.get_source_segment(source,node))
    assert len(result)==len(names)
    return '\n\n'.join(result)


def candidate_source(version):
    parent=(ROOT/'solution/solution_v324.py').read_text()
    doc=ast.get_source_segment(parent,ast.parse(parent).body[0])
    parent=parent.replace(doc,f'"""v{version}: isolated {MODES[version]} Attention probe; Linear v324."""',1)
    shared=extract('tools/learned_rotation_core.py',('_lr_states','_lr_guard'))
    shared+='\n\n'+extract('tools/probe_v326_regroup.py',('nonorthogonal_states','metric_states'),True)
    tail=(ROOT/'tools/attention_online_probe_tail.py').read_text()
    return parent+'\n\n'+shared+'\n\n'+f'_RG_MODE={MODES[version]!r}\n'+tail


if __name__=='__main__':
    for version in MODES:
        source=candidate_source(version);compile(source,f'v{version}','exec')
        path=ROOT/f'solution/solution_v{version}.py'
        if path.exists():assert path.read_text()==source,'Refusing to overwrite differing existing version'
        else:path.write_text(source)
        print(path.name,len(source.splitlines()))
