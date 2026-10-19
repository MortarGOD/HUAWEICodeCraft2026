"""Check role preservation, scale pairing and deterministic hypothesis generation."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.check_role_decoy_calibration import role_samples
torch.manual_seed(17)
samples=[{r:(torch.randint(0,4,(16,64)).float(),torch.rand(16,4)+.1) for r in 'qkv'} for _ in range(3)]
snapshot=[{r:tuple(x.clone() for x in s[r]) for r in 'qkv'} for s in samples]
rng=torch.random.get_rng_state().clone()
for role in 'qkv':
    for mode in ('row_permute','block_resample'):
        a=role_samples(samples,role,mode,13);b=role_samples(samples,role,mode,13)
        for source,result,repeat in zip(samples,a,b):
            for r in 'qkv':
                assert all(torch.equal(x,y) for x,y in zip(result[r],repeat[r]))
                if r==role:
                    assert all(torch.equal(x,y) for x,y in zip(source[r],result[r]))
                else:
                    oq,os=source[r];nq,ns=result[r]
                    for i in range(16):
                        for block in range(4):
                            same=(oq[:,block*16:(block+1)*16]==nq[i,block*16:(block+1)*16]).all(1)&(os[:,block]==ns[i,block])
                            assert bool(same.any())
assert all(torch.equal(x,y) for a,b in zip(samples,snapshot) for r in 'qkv' for x,y in zip(a[r],b[r]))
assert torch.equal(rng,torch.random.get_rng_state())
print('PASS: all 3 real roles preserved; both hypothesis modes reproducible; every quant/scale block comes from same source row; inputs and global RNG unchanged')
