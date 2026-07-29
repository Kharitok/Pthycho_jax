# %%
# limit jax memory
%load_ext autoreload
%autoreload 2
import os
import random

from scipy.ndimage import zoom

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
import time
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.ndimage import gaussian_filter
from skimage import data

# %%


def conj_grads() -> optax.GradientTransformation:
    """Applies a complex conjugate to the gradients statelessly."""
    # Add 'params=None' (or '_') to satisfy the Optax stateless signature
    return optax.stateless(lambda grads, params=None: jax.tree.map(jnp.conj, grads))


# # Use it at the front of a chain:
# optimizer = optax.chain(
#     conj_grads(),
#     optax.adam(learning_rate=1e-2),
# )


# %%

obj_size = 128
probe_size = 64
scan_type = "random"
n_pos = 100


# Load a built-in sample image
image = data.camera().astype(jnp.float32)  # Normalize to [0, 1]
# resample the image to the desired size
image = zoom(image, (obj_size / image.shape[0], obj_size / image.shape[1]), order=3)

# Display the image
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()

p_image = data.gravel().astype(jnp.float32)  # Normalize to [0, 1]
p_image = p_image[:probe_size, :probe_size]
probe_mask = np.zeros((probe_size, probe_size))
probe_mask[
    probe_size // 4 : 3 * probe_size // 4, probe_size // 4 : 3 * probe_size // 4
] = 1

# %%
true_probe = (
    probe_mask * np.abs(p_image) * np.exp(1j * 2 * np.pi * (p_image.T / p_image.max()))
)
true_sample = (0.5 + np.abs(image) / np.abs(image).max()) * np.exp(
    1j * 2 * np.pi * (image.T / image.max())
)

true_scan_positions = np.random.randint(0, obj_size - (5 + probe_size), size=(n_pos, 2))
# %%

true_diff_params = {
    "sample": true_sample,
    "probe_modes": true_probe,
    "modal_weights": jnp.ones((1, 1)),
    "scan_mistakes": jnp.zeros((n_pos, 2)).astype(jnp.float32),
}

true_non_diff_params = {
    "sample_positions":  jnp.asarray(true_scan_positions, dtype=jnp.int32),
}


model_parameters = {
    "probe_type": "static",
    "sample_type": "complex",
    "binning_factor": (1, 1),
    "shift_margin": 4,
    "sample_selector_type": "correcting",
    "propagator_type": "Fourrier",
    "probe_shape": (probe_size, probe_size),
    'max_correction_magnitude':5
}

mask = jnp.ones((probe_size, probe_size), dtype=jnp.bool_)
# %%

from transmission_ptycho_models import construct_point_model_ptycho_transmission

models = construct_point_model_ptycho_transmission(model_parameters=model_parameters)


#### MODEL TESTS
# %%

f_sample = models["sample_constructor"]
f_probe = models["probe_constructor"]

# %%
s = f_sample(true_diff_params, true_non_diff_params, 0)
print(s.shape)

p = f_probe(true_diff_params, true_non_diff_params, 0)
print(p.shape)


point_model = models["point_model"]
I = point_model(true_diff_params, true_non_diff_params, 0)
# %%
batched_model = jax.jit(jax.vmap(point_model, in_axes=(None, None, 0)))
I_batched = batched_model(true_diff_params, true_non_diff_params, jnp.arange(50).astype(jnp.int32))


measured_intensity = I_batched 



from loss_and_reg import (
    gauss_loss,
    join_loss_and_forward,
    join_loss_and_forward_batched,
    poisson_loss,
)

# %%  #### Test reconstructions

# init_params
diff_params = {
    "sample": jnp.ones_like(true_sample),
    "probe_modes": jnp.asarray(true_probe.copy()),
    "modal_weights": jnp.ones((1, 1)),
    "scan_mistakes": jnp.zeros((n_pos, 2)).astype(jnp.float32),
}

non_diff_params = {
    "sample_positions":  jnp.asarray(true_scan_positions, dtype=jnp.int32),
}





get_loss = join_loss_and_forward(gauss_loss, point_model)


get_loss_v = jax.jit(join_loss_and_forward_batched(gauss_loss, point_model))

get_loss(true_diff_params,true_non_diff_params,0,jnp.sqrt(I_batched[0]), mask)


get_loss_v(true_diff_params,true_non_diff_params,jnp.arange(50).astype(jnp.int32),jnp.sqrt(I_batched[:50]), mask)
get_loss_and_grad = jax.jit(jax.value_and_grad(get_loss, argnums=0))
get_loss_and_grad_v = jax.jit(jax.value_and_grad(get_loss_v, argnums=0))
# %%
l,g = get_loss_and_grad_v(true_diff_params,true_non_diff_params,jnp.arange(49).astype(jnp.int32),jnp.sqrt(I_batched[1:50]), mask)
# %%



# init optimizer


# def conj_grads() -> optax.GradientTransformation:
#     """Applies a complex conjugate to the gradients statelessly."""
#     # Add 'params=None' (or '_') to satisfy the Optax stateless signature
#     return optax.stateless(lambda grads, params=None: jax.tree.map(jnp.conj, grads))


# # # Use it at the front of a chain:
# # optimizer = optax.chain(
# #     conj_grads(),
# #     optax.adam(learning_rate=1e-2),
# )
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import optax

from optimizers import prepare_optimizer

# def conj_grads() -> optax.GradientTransformation:
#     return optax.stateless(
#         lambda grads, params=None: jax.tree.map(jnp.conj, grads)
#     )


# def prepare_optimizer(opt_params_per_leaf: Dict) -> optax.GradientTransformation:
#     transforms = {}

#     for param_name, opt_params in opt_params_per_leaf.items():
#         opt_type = opt_params.get("type", "adam")
#         learning_rate = opt_params["learning_rate"]

#         if opt_type == "adam":
#             transforms[param_name] = optax.adam(learning_rate)
#         else:
#             raise ValueError(f"Unsupported optimizer type: {opt_type}")

#     def label_fn(params):
#         unknown = set(params.keys()) - set(transforms.keys())
#         if unknown:
#             raise ValueError(f"Missing optimizer config for parameters: {sorted(unknown)}")
#         return {name: name for name in params.keys()}

#     return optax.chain(
#         conj_grads(),
#         optax.multi_transform(transforms, label_fn),
#     )


opt_params_per_leaf = {
    "sample": {"type": "adam", "learning_rate": 1e-1},
    "probe_modes": {"type": "adam", "learning_rate": 0},
    "modal_weights": {"type": "adam", "learning_rate": 0},
    "scan_mistakes": {"type": "adam", "learning_rate": 0},
}

optimizer = prepare_optimizer(opt_params_per_leaf)
#%%
diff_params = {
    "sample": jnp.ones_like(true_sample),
    "probe_modes": jnp.asarray(true_probe.copy()),
    "modal_weights": jnp.ones((1, 1)),
    "scan_mistakes": jnp.zeros((n_pos, 2)).astype(jnp.float32),
}

non_diff_params = {
    "sample_positions":  jnp.asarray(true_scan_positions, dtype=jnp.int32),
}
#%%
# l,g  =get_loss_and_grad_v(diff_params, non_diff_params, jnp.arange(50).astype(jnp.int32), jnp.sqrt(I_batched[:50]), mask)

opt_state = optimizer.init(diff_params)
params = diff_params

batch_idx = jnp.arange(50).astype(jnp.int32)
batch_measured = jnp.sqrt(I_batched[:50])

for step in range(50):
    loss, grads = get_loss_and_grad_v(
        params,
        non_diff_params,
        batch_idx,
        batch_measured,
        mask,
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    if step % 1 == 0:
        print(f"step {step:03d} loss = {loss}")


plt.imshow(jnp.abs(params["sample"]), cmap="gray")
#%%

def prepare_opt_step(optimizer: optax.GradientTransformation, loss_and_grad_fn: Callable) -> Callable:
    """
    Prepares an optimization step function that computes the loss and gradients,
    applies the optimizer update, and returns the new parameters and optimizer state.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        The Optax optimizer to use for parameter updates.
    loss_and_grad_fn : Callable
        A function that computes the loss and gradients given parameters.

    Returns
    -------
    Callable
        A function that performs a single optimization step.
    """

    @jax.jit
    def opt_step(params: Dict, opt_state: optax.OptState, non_diff_params: Dict, batch_idx: jnp.ndarray, batch_measured: jnp.ndarray, mask: jnp.ndarray) -> Tuple[Dict, optax.OptState, jnp.float32]:
        loss, grads = loss_and_grad_fn(
            params,
            non_diff_params,
            batch_idx,
            batch_measured,
            mask,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params,opt_state,loss
        

    return opt_step

# def step_fn(carry, _):
#     params, opt_state = carry
#     params, opt_state, loss_val = opt_step(
#         params, opt_state, non_diff_params, batch_idx, batch_measured, mask
#     )
#     return (params, opt_state), loss_val

# (params, opt_state), loss_hist = jax.lax.scan(
#     step_fn,
#     (params, opt_state),
#     xs=None,
#     length=n_iter,
# )

# plt.plot(np.asarray(loss_hist))
from optim_loop import prepare_opt_step

#%%


opt_state = optimizer.init(diff_params)
params = diff_params

opt_step = prepare_opt_step(optimizer, get_loss_and_grad_v)

batch_idx = jnp.arange(50).astype(jnp.int32)
batch_measured = jnp.sqrt(I_batched[:50])
n_iter = 50

loss = []


before = time.time()
for step in range(n_iter):
    params, opt_state, loss_iter = opt_step(
        params,
        opt_state,
        non_diff_params,
        batch_idx,
        batch_measured,
        mask,
    )

    loss.append(loss_iter)
# plt.plot(loss)

after = time.time()
print(f"Time taken for {n_iter} iterations: {after - before:.2f} seconds {(after - before)/n_iter:.4f} seconds per iteration")



plt.imshow(jnp.abs(params["sample"]), cmap="gray")
# %% with sharding



from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

devices = np.array(jax.devices())
n_devices = devices.size
if n_devices < 2:
    print("Only one device visible; sharding won't give multi-GPU speedup.")



mesh = Mesh(devices, axis_names=("data",))
rep = NamedSharding(mesh, P())                        # replicated
idx_shard = NamedSharding(mesh, P("data"))            # [B]
meas_shard = NamedSharding(mesh, P("data", None, None))  # [B, H, W]
#%%

batch_size = 50
batch_idx_host = jnp.arange(batch_size).astype(jnp.int32)
batch_measured_host = jnp.sqrt(I_batched[:batch_size])

if batch_size % n_devices != 0:
    raise ValueError(f"Batch size {batch_size} must be divisible by number of devices {n_devices}.")


opt_state = optimizer.init(diff_params)
params = diff_params


params = jax.device_put(diff_params, rep)
opt_state = jax.device_put(optimizer.init(diff_params), rep)

non_diff_params_sh = jax.device_put(non_diff_params, rep)
mask_sh = jax.device_put(mask, rep)

batch_idx = jnp.arange(50).astype(jnp.int32)
batch_measured = jnp.sqrt(I_batched[:50])

batch_idx = jax.device_put(batch_idx_host, idx_shard)
batch_measured = jax.device_put(batch_measured_host, meas_shard)


#%%



opt_step = prepare_opt_step(optimizer, get_loss_and_grad_v)



n_iter = 5000
# loss = jnp.zeros(n_iter)

loss = []
before = time.time()
for step in range(n_iter):
    params, opt_state, loss_iter = opt_step(
        params,
        opt_state,
        non_diff_params,
        batch_idx,
        batch_measured,
        mask,
    )

    # loss.append(loss_iter)
# plt.plot(loss)
after = time.time()
print(f"Time taken for {n_iter} iterations: {after - before:.2f} seconds {(after - before)/n_iter:.4f} seconds per iteration")

params_host = jax.device_get(params)
plt.imshow(jnp.abs(params_host["sample"]), cmap="gray")
# %%

opt_params_per_leaf = {
    "sample": {"type": "adam", "learning_rate": 1e-1},
    "probe_modes": {"type": "adam", "learning_rate": 1e1},
    "modal_weights": {"type": "adam", "learning_rate": 0},
    "scan_mistakes": {"type": "adam", "learning_rate": 1e-2},
}





optimizer = prepare_optimizer(opt_params_per_leaf)


from optim_loop import run_optimization_loop

params = diff_params
params['scan_mistakes'] = (jnp.zeros((n_pos, 2)) + np.random.randn(n_pos, 2)*0.3).astype(jnp.float32)
r_noise = (np.random.randn(*true_probe.shape)*80).astype(jnp.float32)
params['probe_modes'] = jnp.asarray(true_probe.copy()) + (r_noise - r_noise.mean())
non_diff_params = {
    "sample_positions":  jnp.asarray(true_scan_positions + np.random.randint(-2, 2, true_scan_positions.shape), dtype=jnp.int32),
}


opt_state = optimizer.init(params)

measured_pool = jnp.sqrt(I_batched[:50])  # shape [n_positions, H, W]


get_loss_and_grad_v = jax.jit(jax.value_and_grad(get_loss_v, argnums=0))
get_loss_and_grad_v = jax.jit(jax.value_and_grad(get_loss_v, argnums=0))

from loss_and_reg import create_regularizer, create_tv_reg, join_loss_and_reg

reg_s_tv = create_regularizer(
                               diff_params_name='sample',
                                 reg_weight_name='tv_sample_weight',
                                 reg_func = create_tv_reg(p=2,q= 2,conv_func = jnp.angle))
reg_loss = join_loss_and_reg(get_loss_v, reg_s_tv)
get_loss_and_grad_v = jax.jit(jax.value_and_grad(reg_loss, argnums=0))
non_diff_params['tv_sample_weight'] = 1e0
###


import projectors as pj
from optim_loop import run_optimization_loop

proj_sample_clip = pj.create_projection_applier(
    pj.create_limiter_scaling_complex(jnp.abs(true_sample).max(), jnp.abs(true_sample).min()),
    diff_param_name = 'sample'
)

proj_scan_limit = pj.create_projection_applier(
    pj.create_shift_limiter(max_allowed_shift=10,initial_positions=non_diff_params['sample_positions']),
    non_diff_param_name='sample_positions'
)



proj_shift_rounder = pj.create_projection_applier(
    pj.create_shift_rounder(
        max_correction_magnitude = 5,
        rounding_step = 3,
    ),
    diff_param_name = 'scan_mistakes',
    non_diff_param_name='sample_positions'
)

projector = pj.merge_multiple_projections([ proj_shift_rounder])
###


plt.imshow(jnp.abs(params["sample"]), cmap="gray")
plt.show()
#%%
# Mode 1: one update per full pass (gradient accumulation)
params, opt_state, loss_hist = run_optimization_loop(
    params=params,
    opt_state=opt_state,
    optimizer=optimizer,
    loss_and_grad_fn=get_loss_and_grad_v,
    non_diff_params=non_diff_params,
    measured_batch_pool=batch_measured,
    mask=mask,
    mode="accumulate_full_pass",
    n_steps=500,      # epochs
    batch_size=50,    # memory-fit batch
    seed=0,
    shuffle_each_epoch=True,
    projection_fn = projector,
    use_multigpu = True
)

plt.plot(loss_hist)
plt.show()
plt.figure()
# show abs and angle of the sample
plt.subplot(1, 2, 1)
plt.imshow(jnp.abs(params["sample"]), cmap="turbo")#[30:100,40:110]
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(jnp.angle(params["sample"]), cmap="turbo")
#switch off axis 
plt.axis("off")
plt.tight_layout()
plt.show()
plt.figure()
if params["probe_modes"].ndim <3:
    plt.figure()
    # show abs and angle of the sample
    plt.subplot(1, 2, 1)
    plt.imshow(jnp.abs(params["probe_modes"]), cmap="turbo")#[30:100,40:110]
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(jnp.angle(params["probe_modes"]), cmap="turbo")
    #switch off axis 
    plt.axis("off")
    plt.tight_layout()
    plt.show()
#%%
# Mode 2: update every batch, no repeats within each epoch
params, opt_state, loss_hist = run_optimization_loop(
    params=params,
    opt_state=opt_state,
    optimizer=optimizer,
    loss_and_grad_fn=get_loss_and_grad_v,
    non_diff_params=non_diff_params,
    measured_batch_pool=measured_pool,
    mask=mask,
    mode="sequential_no_repeats",
    n_steps=250,      # epochs
    batch_size=30,
    seed=0,
    shuffle_each_epoch=True,
    use_multigpu = False
)


plt.plot(loss_hist)
plt.show()
plt.imshow(jnp.abs(params["sample"])[30:100,40:110], cmap="turbo")
plt.show()
#%%
# Mode 3: update every random batch (with replacement)
params, opt_state, loss_hist = run_optimization_loop(
    params=params,
    opt_state=opt_state,
    optimizer=optimizer,
    loss_and_grad_fn=get_loss_and_grad_v,
    non_diff_params=non_diff_params,
    measured_batch_pool=measured_pool,
    mask=mask,
    mode="random_with_replacement",
    n_steps=100,     # update steps
    batch_size=20,
    seed=0,
    shuffle_each_epoch=True,  # ignored in this mode
    use_multigpu = True
)

plt.plot(loss_hist)
plt.show()
plt.imshow(jnp.abs(params["sample"]), cmap="gray")
plt.show()
# %%


import projectors as pj

# %%
