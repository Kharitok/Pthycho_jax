"""Independent calls to the untouched AD reference; no DM stage operators."""
import jax
import jax.numpy as jnp
from ptycho_jax.transmission.transmission_ptycho_models import construct_point_model_ptycho_transmission

def reference_model(geom):
    return construct_point_model_ptycho_transmission({
        'probe_type':'fluctuating','sample_type':'complex','binning_factor':(1,1),
        'sample_selector_type':'correcting','propagator_type':'Fourrier',
        'probe_shape':geom.frame_shape,'shift_margin':geom.margin,'max_correction_magnitude':5.0,
    })['point_model']

def reference_intensity(probe,obj,a,geom):
    model=reference_model(geom)
    params={'probe_modes':probe,'sample':obj,'modal_weights':jax.vmap(jnp.diag)(a).astype(probe.dtype),
            'scan_mistakes':jnp.arctanh(-geom.delta/5)}
    return jax.vmap(model,in_axes=(None,None,0))(params,{'sample_positions':geom.n},jnp.arange(len(a)))
