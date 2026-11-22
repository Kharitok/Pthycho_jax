# Audited transmission conventions

| Topic | Choice and evidence |
| --- | --- |
| Reference | `transmission/transmission_ptycho_models.py`: `construct_point_model_ptycho_transmission`, `prepare_correcting_sample_selector`, `fluctuating_probe`, `fourier_propagator`, `intensifier` |
| Object | Complex canvas `(H,W)`; complex64 in production, complex128 in algebra tests |
| Probe | Global complex modes `(K,Ny,Nx)`; DM uses unit norms, AD stores physical amplitude scale in these modes |
| Positions | Fixed array coordinates `(row,column)` in sample pixels. Floor integer plus fraction in `[0,1)`; positive integer moves crop origin forward |
| NanoMAX orientation | Retain the validated loader: row-flipped detector and beam coordinates `(-sy,+sx)`, motors in micrometres, geometry from raw energy and supplied detector settings |
| Actual selector | Extract enlarged patch at `n`; multiply cosine taper over margin `m`; Fourier translate by `-delta`; crop away `m` on every side |
| Fourier ramp | `exp(-2πi*(fy*shift_y+fx*shift_x))`; negative fractional shift samples the object at the positive fractional coordinate |
| DM equation | `psi[j,k]=a[j,k]*P[k]*L[j](O)` with the exact selector operator `L`, rather than an approximately equivalent shifted-probe model |
| Weights | DM amplitude `a=sqrt(w)>=0`, shape `(J,K)`; AD amplitudes are full complex matrices `(J,K,K)`. DM maps to their real nonnegative diagonal; general AD matrices cannot map back to this restricted family |
| FFT | `fft2(...,norm='ortho',axes=(-2,-1))`, no shifts in the propagator. Measured data and detector masks are already `ifftshift`ed once by the loader/pipeline |
| Detector | Incoherent sum of squared absolute Fourier fields; no additive background, detector gain or propagation nuisance term in the chosen reference configuration |
| Data/mask | Photon intensities are nonnegative, finite, masked and cropped. `Data.sqrt_I` is amplitude `(J,Ny,Nx)`. True mask pixels are valid; detector mask may broadcast over shots |
| Loss | `gauss_loss`: mean over valid pixels of `(sqrt(I+1e-20)-sqrt_I)**2`, then mean over shots. DM M3 uses the same reduction; R is separately normalized by total valid counts |
| Adjoint | Complex Hermitian inner product `vdot`; `conj(linear_transpose(f)(conj(y)))` is the independent reference |
| Damping | Proximal to previous object/probe, never toward zero. CG object solve uses exact shifted-object normal operator; probe normal is exactly diagonal |
| Boundaries | Full enlarged patches must fit. Fourier translation is circular on the enlarged tapered patch, exactly as in the reference |
| AD positions | Frozen latents live in fixed parameters; the wrapper injects them into the untouched point model. No position optimizer leaf exists |
| Numerics | Installed JAX 0.11.1; tests enable x64, production explicitly converts arrays to float32/complex64 |

No unsupported detector or background terms are silently dropped. Fractional positions use identical finite-grid operators, not a tolerance-based approximate handoff.
