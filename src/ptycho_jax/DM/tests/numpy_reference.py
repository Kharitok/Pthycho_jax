"""Independent integer-position NumPy DM; no JAX or production operators."""
import numpy as np


def iterate(psi,probe,obj,a,origins,margin,root,iterations,n_inner=2,alpha_o=1e-3,alpha_p=1e-3):
    ny,nx=probe.shape[-2:]
    positions=np.asarray(origins)+margin
    for _ in range(iterations):
        f=np.fft.fft2(psi,norm='ortho')
        amplitude=np.sqrt(np.sum(abs(f)**2,axis=1))
        projected=np.fft.ifft2(f*(root/np.maximum(amplitude,1e-30))[:,None],norm='ortho')
        target=2*projected-psi
        for _ in range(n_inner):
            numerator=np.zeros_like(obj);denominator=np.zeros_like(obj.real)
            for j,(y,x) in enumerate(positions):
                u=a[j,:,None,None]*probe
                numerator[y:y+ny,x:x+nx]+=np.sum(u.conj()*target[j],axis=0)
                denominator[y:y+ny,x:x+nx]+=np.sum(abs(u)**2,axis=0)
            alpha=max(alpha_o*denominator.max(),np.finfo(obj.real.dtype).eps)
            obj=(numerator+alpha*obj)/(denominator+alpha)
            numerator=np.zeros_like(probe);denominator=np.zeros_like(probe.real)
            for j,(y,x) in enumerate(positions):
                patch=obj[y:y+ny,x:x+nx]
                u=a[j,:,None,None]*patch
                numerator+=u.conj()*target[j]
                denominator+=abs(u)**2
            alpha=np.maximum(alpha_p*np.max(denominator,axis=(-2,-1)),np.finfo(probe.real.dtype).eps)
            probe=(numerator+alpha[:,None,None]*probe)/(denominator+alpha[:,None,None])
        overlap=np.stack([a[j,:,None,None]*probe*obj[y:y+ny,x:x+nx]
                          for j,(y,x) in enumerate(positions)])
        psi=psi+overlap-projected
    return psi,probe,obj,a
