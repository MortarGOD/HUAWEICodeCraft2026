"""Analytic derivatives against autograd; no score fitting."""
import torch


def main():
    torch.manual_seed(365)
    dtype=torch.float64
    n,d=11,8
    q=torch.randn(d,dtype=dtype)
    k=torch.randn(n,d,dtype=dtype)
    v=torch.randn(n,d,dtype=dtype)
    def output(query,key):
        return torch.softmax(query@key.T/d**.5,-1)@v
    p=torch.softmax(q@k.T/d**.5,-1); out=p@v
    jq=k.T@(p[:,None]*(v-out))/d**.5
    reference=torch.autograd.functional.jacobian(lambda x:output(x,k),q)
    torch.testing.assert_close(jq.T,reference,atol=1e-12,rtol=1e-10)
    jk=torch.autograd.functional.jacobian(lambda x:output(q,x),k)
    for j in range(n):
        expected=p[j]**2*((v[j]-out)**2).sum()*torch.outer(q,q)/d
        torch.testing.assert_close(jk[:,j].T@jk[:,j],expected,atol=1e-12,rtol=1e-10)
    shift=torch.randn(d,dtype=dtype)
    torch.testing.assert_close(output(q,k+shift),out,atol=1e-12,rtol=1e-10)
    assert torch.linalg.eigvalsh(jq@jq.T).min()>-1e-12
    print('PASS: Q full Jacobian, K token-block Gram, PSD, softmax translation invariance')


if __name__=='__main__':
    main()
