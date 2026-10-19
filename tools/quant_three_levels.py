"""Mechanical, exact-candidate reuse: lv2/lv3 have only three products."""
import ast


def rewrite(source):
    for name in ('_quant_chunk','_quant_chunk_vec'):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        old=ast.get_source_segment(source,node)
        if name=='_quant_chunk':
            start=old.index('        for lv2_c in (1.0, 2.0):')
            end=old.index('            take1 = e3_list[0]',start)
            fresh='''        # Products 1*2 and 2*1 are identical; compute their errors once.
        product_errors = []
        for factor in (1.0, 2.0, 4.0):
            unit = sf5 * factor
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            product_errors.append(((mant * unit - ab) ** 2 * wblk).sum(dim=4))
        for level, lv2_c in enumerate((1.0, 2.0)):
            e3_list = product_errors[level:level+2]
'''
        else:
            start=old.index('        for lv2_c in (1.0, 2.0):')
            end=old.index('            take1 = e3_list[0]',start)
            fresh='''        product_errors = []
        for factor in (1.0, 2.0, 4.0):
            unit = sf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * factor
            tgt = tmp[:, :, :kB] if kB < KB else tmp
            torch.div(abB, unit, out=tgt)
            tgt.mul_(4.0).round_().mul_(0.25).clamp_(0.0, 1.75)
            tgt.mul_(unit).sub_(abB).pow_(2).mul_(wbB)
            product_errors.append(tgt.sum(dim=5))
        for level, lv2_c in enumerate((1.0, 2.0)):
            e3_list = product_errors[level:level+2]
'''
        source=source.replace(old,old[:start]+fresh+old[end:])
    return source
