"""Exercise exact generated selection block, not a duplicate implementation."""
import torch
from probe_v319_value import ROOT
from joint_cross_sequence_source import frontier_source


def main():
    source=frontier_source((ROOT/'solution/solution_v324.py').read_text())
    begin=source.index('        for hv in range(kh):\n            candidates=costs')
    end=source.index('        answer = dict(states)',begin)
    body='\n'.join(line[4:] for line in source[begin:end].splitlines())
    namespace={'torch':torch}
    exec('def select(costs):\n    kh=costs.shape[-1]\n    bmask=torch.zeros(kh,dtype=torch.bool)\n    vmask=bmask.clone()\n'+body+'\n    return bmask.long()*2+vmask.long()',namespace)
    gen=torch.Generator().manual_seed(352);checks=0
    for _ in range(1000):
        costs=torch.rand(2,2,2,3,generator=gen)+.01
        chosen=namespace['select'](costs)
        for h in range(3):
            candidates=costs[:,:,:,h].reshape(4,2);best=candidates[0]
            for v in candidates:
                if bool((v<best*.99).all()):best=v
            selected=candidates[chosen[h]]
            assert selected.mean()<=best.mean()+1e-7
            assert chosen[h]==0 or bool((selected<candidates[0]*.99).all())
            checks+=2
    print({'checks':checks,'meaning':'No worse held-out mean than greedy; both original-baseline folds meet 1% improvement for nonbaseline selection.'})


if __name__=='__main__':main()
