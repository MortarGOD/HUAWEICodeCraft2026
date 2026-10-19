"""Static V-only pipeline cloning; introspection is offline and never submitted."""
import ast,inspect
from pathlib import Path
from probe_v319_value import load_solution
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(version=347):
    assert version==347
    parent=(ROOT/'solution/solution_v324.py').read_text();m=load_solution(ROOT/'solution/solution_v324.py')
    mapping={'_dyn_table':'_vg_table','_vk_parent_v':'_vg_base_v','_dyn_v':'_vg_kernel_v','_vm_parent_dynamic':'_vg_entry_v','_vr_parent_dynamic':'_vg_forward_v'}
    class Rewrite(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in mapping:node.id=mapping[node.id]
            return node
    chunks=[]
    for function,name in ((m._vk_parent_v,'_vg_base_v'),(m._dyn_v,'_vg_kernel_v'),(m._vm_parent_dynamic,'_vg_entry_v'),(m._vr_parent_dynamic,'_vg_forward_v'),(m.hif4_dynamic_quantize_v,'_vg_v')):
        tree=ast.parse(inspect.getsource(function));tree=Rewrite().visit(tree);tree.body[0].name=name
        chunks.append(ast.unparse(ast.fix_missing_locations(tree)))
    doc=ast.get_source_segment(parent,ast.parse(parent).body[0])
    parent=parent.replace(doc,'"""v347: wider native V scales, original V postprocessing; Linear/Q/K v324."""',1)
    core=(ROOT/'tools/value_grid_tail.py').read_text()
    return parent+'\n\n'+'\n\n'.join(chunks)+'\n\n'+core


if __name__=='__main__':
    source=candidate_source();compile(source,'v347','exec')
    (ROOT/'solution/solution_v347.py').write_text(source);print(len(source.splitlines()))
