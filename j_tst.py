# %%
# limit jax memory
import os
import random

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.ndimage import gaussian_filter

# %%


z_targ = 1 - 1j


def loss(z):
    return jnp.abs(z - z_targ) ** 2


def conj_grads() -> optax.GradientTransformation:
    def init_fn(params):
        return ()  # no state needed

    def update_fn(grads, state, params=None):
        conj = jax.tree.map(lambda g: g.conj(), grads)
        return conj, state

    return optax.GradientTransformation(init_fn, update_fn)


def conj_grads() -> optax.GradientTransformation:
    """Applies a complex conjugate to the gradients statelessly."""
    # Add 'params=None' (or '_') to satisfy the Optax stateless signature
    return optax.stateless(lambda grads, params=None: jax.tree.map(jnp.conj, grads))


# Use it at the front of a chain:
optimizer = optax.chain(
    conj_grads(),
    optax.adam(learning_rate=1e-2),
)


# optimizer = optax.adam(learning_rate=1e-2)

# optimizer= optax.chain(
#     optax.contrib.split_real_and_imaginary(
#     optax.adam(learning_rate=1e-2),
# ))

# %% SGD


z = jnp.array(1e-5 + 1e-5j)
# opt = optax.sgd(learning_rate=1e1)
opt_state = optimizer.init(z)


# %%
# jitted optim step
@jax.jit
def opt_step(z, opt_state):
    l = loss(z)
    g = jax.grad(loss)(z)
    updates, opt_state = optimizer.update(g, opt_state)
    z = optax.apply_updates(z, updates)
    return z, opt_state, l


# optimization loop
n_iters = 300
losses = np.zeros(n_iters)
values = np.zeros(n_iters, dtype=complex)
for i in range(n_iters):
    z, opt_state, l = opt_step(z, opt_state)
    losses[i] = l
    values[i] = z


# %%

# plot results

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.yscale("log")
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(
    values.real,
    values.imag,
)
plt.plot(z_targ.real, z_targ.imag, marker="x", color="red", label="Target")
plt.title("Optimization Path")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.tight_layout()

# plott the loss landscape and trajectory
x = np.linspace(-1, 3, 100)
y = np.linspace(-3, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.abs(X + 1j * Y - z_targ) ** 2
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, Z, levels=50, cmap="magma")
plt.plot(values.real, values.imag, marker=",", color="blue", label="Trajectory")
plt.plot(z_targ.real, z_targ.imag, marker="x", color="red", label="Target")
plt.title("Loss Landscape with Trajectory")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.tight_layout()

# %%


import jax
import jax.numpy as jnp


@jax.jit
def get_scan_slices(object_array, patch_size, stride):
    """
    Extracts 2D slices from a 2D array representing a ptychographic scan.

    Args:
        object_array: (H, W) array.
        patch_size: Tuple (h, w) of the probe/scan window.
        stride: Tuple (sy, sx) of the scanning step size.

    Returns:
        (N, h, w) array of slices, where N is total scan positions.
    """
    # conv_general_dilated_patches expects (Batch, Channel, Height, Width)
    # We add 1 for Batch and 1 for Channel
    x = object_array[jnp.newaxis, jnp.newaxis, :, :]

    # Extract patches
    patches = jax.lax.conv_general_dilated_patches(
        lhs=x, filter_shape=patch_size, window_strides=stride, padding="VALID"
    )

    # The output shape is (1, 1, N, h, w)
    # We reshape to (N, h, w)
    num_patches = patches.shape[2]
    return patches.reshape((num_patches, patch_size[0], patch_size[1]))


# Example Usage:
# Object: 100x100, Probe: 16x16, Stride: 8
obj = jnp.ones((100, 100))
slices = get_scan_slices(obj, (16, 16), (8, 8))
print(f"Extracted shape: {slices.shape}")
# Output: (121, 16, 16) - 121 scan positions


# %%


import jax
import jax.numpy as jnp
import optax

# The ground truth complex number we want to find
z_target = 3.0 - 4.0j


def loss_fn(z):
    """A simple quadratic bowl in complex space."""
    return jnp.abs(z - z_target) ** 2


# ---------------------------------------------------------
# 1. Pure SGD (Proves JAX's Wirtinger direction is correct)
# ---------------------------------------------------------
def test_sgd():
    z = jnp.array(0.0 + 0.0j)
    optimizer = optax.sgd(learning_rate=0.1)
    opt_state = optimizer.init(z)

    for _ in range(100):
        g = jax.grad(loss_fn)(z)
        updates, opt_state = optimizer.update(g, opt_state)
        z = optax.apply_updates(z, updates)
    print(f"SGD (Raw Complex) Result:  {z:.3f}")


# ---------------------------------------------------------
# 2. Adam on Raw Complex (Proves Adam ruins the angle)
# ---------------------------------------------------------
def test_adam_raw():
    z = jnp.array(0.0 + 0.0j)
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(z)

    for _ in range(100):
        g = jax.grad(loss_fn)(z)
        updates, opt_state = optimizer.update(g, opt_state)
        z = optax.apply_updates(z, updates)
    print(f"Adam (Raw Complex) Result: {z:.3f}  <-- Fails to converge")


# ---------------------------------------------------------
# 3. Adam with Real/Imag Split (The Fix)
# ---------------------------------------------------------
def split_loss_fn(z_split):
    # Recombine inside the loss
    z_complex = z_split["real"] + 1j * z_split["imag"]
    return jnp.abs(z_complex - z_target) ** 2


def test_adam_split():
    z_split = {"real": jnp.array(0.0), "imag": jnp.array(0.0)}
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(z_split)

    for _ in range(100):
        g = jax.grad(split_loss_fn)(z_split)
        updates, opt_state = optimizer.update(g, opt_state)
        z_split = optax.apply_updates(z_split, updates)

    print(f"Adam (Split Dict) Result:  {z_split['real']:.3f} + {z_split['imag']:.3f}j")


# Run the tests
print(f"Target is:                 {z_target:.3f}\n")
test_sgd()
test_adam_raw()
test_adam_split()
# %%


import jax
import jax.numpy as jnp
import optax


def loss(z):
    return jnp.abs(z - (1 + 2j)) ** 2


opt = optax.adam(1e-2)

z = jnp.array(0.0 + 0.0j)

state = opt.init(z)

for _ in range(1000):
    g = jax.grad(loss)(z)
    updates, state = opt.update(g, state, z)
    z = optax.apply_updates(z, updates)

print(z)
# %%


import matplotlib.pyplot as plt
from skimage import data

# Load a built-in sample image
image = data.camera().astype(jnp.float32)  # Normalize to [0, 1]

# Display the image
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()


# %%
###################################################################
def get_probe_for_point(params, fixed_params):
    # Placeholder function for getting the probe for a specific point
    return params["modes"] * params["modal_weight"]


def get_sample_for_point(params, fixed_params):
    beginings = fixed_params["beginings"]
    return jax.lax.dynamic_slice(params["sample"], beginings, (64, 64))


def get_exit_wave_for_point(params, fixed_params):
    sample = get_sample_for_point(params, fixed_params)
    probe = get_probe_for_point(params, fixed_params)
    return sample * probe


def get_diffraction_for_point(params, fixed_params):
    exit_wave = get_exit_wave_for_point(params, fixed_params)
    return jnp.abs(jnp.fft.fft2(exit_wave, axes=(-2, -1))) ** 2


#####################################################################

# %%

params = {
    "sample": data.camera().astype(jnp.float32)
    + 1j * np.transpose(data.camera().astype(jnp.float32)),  # Normalize to [0, 1]
    "modes": jnp.exp(
        1j * data.camera().astype(jnp.float32)[256 : 256 + 64, 256 : 256 + 64]
    ),
    "modal_weight": np.array((1)),
}

fixed_params = {
    "beginings": [100, 50],
}
# %%


p = get_probe_for_point(params, fixed_params)
s = get_sample_for_point(params, fixed_params)

plt.imshow(jnp.abs(s), cmap="gray")
plt.figure()
plt.imshow(jnp.angle(s), cmap="gray")

ew = get_exit_wave_for_point(params, fixed_params)

plt.imshow(jnp.abs(ew), cmap="gray")
plt.figure()
plt.imshow(jnp.angle(ew), cmap="gray")


di = get_diffraction_for_point(params, fixed_params)

plt.imshow(jnp.log1p(di), cmap="gray")
plt.title("Diffraction Pattern")
plt.show()


# %%
n_meas = 200


params = {
    "sample": data.camera().astype(jnp.float32)
    + 1j * np.transpose(data.camera().astype(jnp.float32)),  # Normalize to [0, 1]
    "modes": jnp.exp(
        1j * data.camera().astype(jnp.float32)[256 : 256 + 64, 256 : 256 + 64]
    ),
    "modal_weight": np.random.rand(n_meas) + 1,
}

fixed_params = {
    "beginings": np.random.randint(0, 420, size=(n_meas, 2)),
}


in_ax_params = {"sample": None, "modes": None, "modal_weight": 0}

in_ax_fixed_params = {
    "beginings": 0,
}

get_diff_vect = jax.vmap(
    get_diffraction_for_point, in_axes=(in_ax_params, in_ax_fixed_params)
)

# %%
ve = get_diff_vect(params, fixed_params)

# %%

init_parms = {
    "sample": np.ones_like(params["sample"]),  # Normalize to [0, 1]
    "modes": jnp.exp(
        1j * data.camera().astype(jnp.float32)[256 : 256 + 64, 256 : 256 + 64]
    ),
    "modal_weight": np.random.rand(n_meas) + 1,
}


# %%
# %%

# %%


def get_slices(image, beginnings):
    return jax.lax.dynamic_slice(image, beginnings, (64, 64))


vgs = jax.vmap(get_slices, in_axes=(None, 0))


def get_prop(image):
    return jnp.fft.fft2(image, axes=(-2, -1))


vgp = jax.vmap(get_prop, in_axes=0)


dummy_probe = 1 * np.exp(
    1j * 2 * np.pi * gaussian_filter(np.random.rand(64, 64), sigma=1)
)
dummy_probe = dummy_probe - jnp.mean(dummy_probe)


def get_fm(image, beginnings):
    slice_ = get_slices(image, beginnings)
    return jnp.abs(jnp.fft.fft2(slice_ * dummy_probe, axes=(-2, -1))) ** 2


vgf = jax.vmap(get_fm, in_axes=(None, 0))

scan_points = 200

dummy_beginnings = jnp.array(np.random.randint(0, 420, size=(scan_points, 2))).astype(
    jnp.int32
)


measurements = vgf(image, dummy_beginnings)
print(measurements.shape)  # Should be (100, 64, 64)


def get_loss(measurement, simmulation):
    return (1 / 2) * jnp.mean(
        (jnp.sqrt(measurement + 1e-9) - jnp.sqrt(simmulation + 1e-9)) ** 2
    )


def loss(params, fixed_params):
    # params: the object we want to optimize (e.g. a 2D array)
    # fixed_params: the known parameters (e.g. probe, scan positions)
    simmulation: jax.Array = vgf(
        params["object"], fixed_params["beginnings"]
    )  # Simulate measurements from current guess
    return get_loss(
        fixed_params["measurements"], simmulation
    )  # Compare to actual measurements


loss_and_grads = jax.value_and_grad(
    loss, argnums=0
)  # We want gradients w.r.t. the object

# %% prepare optimizers

obj_init = jnp.ones((512, 512)) * jnp.mean(image) + 0j  # Initial guess for the object
params = {
    "object": obj_init.copy(),  # Initial guess for the object
}
fixed_params = {
    "beginnings": dummy_beginnings,  # Known scan positions
    "measurements": measurements,  # Actual measurements to fit
}


optimizer = optax.chain(
    conj_grads(),
    optax.adam(learning_rate=5e0),
)

opt_state = optimizer.init(params)


# optimization step
@jax.jit
def opt_step(params, opt_state, fixed_params):
    loss_value, grads = loss_and_grads(params, fixed_params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


n_iters = 10
losses = np.zeros(n_iters)

# check batch size influence
batch_size = scan_points // 50

# iteration loop

for i in range(n_iters):
    random_indicies = np.random.permutation(scan_points)
    for j in range(0, scan_points, batch_size):
        batch_fixed_params = {
            "beginnings": fixed_params["beginnings"][
                random_indicies[j : j + batch_size]
            ],
            "measurements": fixed_params["measurements"][
                random_indicies[j : j + batch_size]
            ],
        }
        params, opt_state, losses[i] = opt_step(params, opt_state, batch_fixed_params)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.yscale("log")
plt.title("Loss over Iterations")
plt.subplot(1, 3, 2)
plt.imshow(jnp.abs(params["object"]), cmap="gray")
plt.title("Reconstructed Object")

plt.subplot(1, 3, 3)
plt.imshow(jnp.abs(obj_init), cmap="gray")
plt.title("Initial Object")
plt.axis("off")
plt.tight_layout()


# check batch size influence
batch_size = scan_points // 8

# iteration loop

for i in range(n_iters):
    random_indicies = np.random.permutation(scan_points)
    for j in range(0, scan_points, batch_size):
        batch_fixed_params = {
            "beginnings": fixed_params["beginnings"][
                random_indicies[j : j + batch_size]
            ],
            "measurements": fixed_params["measurements"][
                random_indicies[j : j + batch_size]
            ],
        }
        params, opt_state, losses[i] = opt_step(params, opt_state, batch_fixed_params)


# for i in range(n_iters):
#     params, opt_state, losses[i] = opt_step(params, opt_state, fixed_params)
#     # print(f"Iteration {i+1}, Loss: {losses[i]:.4f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.yscale("log")
plt.title("Loss over Iterations")
plt.subplot(1, 3, 2)
plt.imshow(jnp.abs(params["object"]), cmap="gray")
plt.title("Reconstructed Object")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(jnp.abs(obj_init), cmap="gray")
plt.title("Initial Object")
plt.axis("off")
plt.tight_layout()


# check batch size influence
batch_size = scan_points // 3

# iteration loop

for i in range(n_iters):
    random_indicies = np.random.permutation(scan_points)
    for j in range(0, scan_points, batch_size):
        batch_fixed_params = {
            "beginnings": fixed_params["beginnings"][
                random_indicies[j : j + batch_size]
            ],
            "measurements": fixed_params["measurements"][
                random_indicies[j : j + batch_size]
            ],
        }
        params, opt_state, losses[i] = opt_step(params, opt_state, batch_fixed_params)


# for i in range(n_iters):
#     params, opt_state, losses[i] = opt_step(params, opt_state, fixed_params)
#     # print(f"Iteration {i+1}, Loss: {losses[i]:.4f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.yscale("log")
plt.title("Loss over Iterations")
plt.subplot(1, 3, 2)
plt.imshow(jnp.abs(params["object"]), cmap="gray")
plt.title("Reconstructed Object")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(jnp.abs(obj_init), cmap="gray")
plt.title("Initial Object")
plt.axis("off")
plt.tight_layout()


# check batch size influence
batch_size = scan_points // 1

# iteration loop

for i in range(n_iters):
    random_indicies = np.random.permutation(scan_points)
    for j in range(0, scan_points, batch_size):
        batch_fixed_params = {
            "beginnings": fixed_params["beginnings"][
                random_indicies[j : j + batch_size]
            ],
            "measurements": fixed_params["measurements"][
                random_indicies[j : j + batch_size]
            ],
        }
        params, opt_state, losses[i] = opt_step(params, opt_state, batch_fixed_params)


# for i in range(n_iters):
#     params, opt_state, losses[i] = opt_step(params, opt_state, fixed_params)
#     # print(f"Iteration {i+1}, Loss: {losses[i]:.4f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.yscale("log")
plt.title("Loss over Iterations")
plt.subplot(1, 3, 2)
plt.imshow(jnp.abs(params["object"]), cmap="gray")
plt.title("Reconstructed Object")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(jnp.abs(obj_init), cmap="gray")
plt.title("Initial Object")
plt.axis("off")
plt.tight_layout()


# %%


slice__ = vgs(image, jnp.array([[100, 100], [110, 110]]))


plt.imshow(slice__[0])
plt.figure()
plt.imshow(slice__[1])


# %%


# vectorized version
# gsj = jax.jit(jax.vmap(get_slices, in_axes=(None, 0)))


slice_starts = np.random.randint(
    0, 420, size=(100, 2)
)  # 3 random starting points (y,x)

t_slices = gsj(image, slice_starts)
print(t_slices.shape)  # Should be (3, 64, 64)


def fm(image, idx):
    return (
        jnp.abs(
            jnp.fft.fft2(
                get_slices(
                    image,
                    jax.lax.dynamic_slice(slice_starts, (idx, 0), (1, 2)).squeeze(),
                ),
                axes=(-1, -2),
            )
        )
        ** 2
    )


jvfm = jax.jit(jax.vmap(fm, in_axes=(None, 0)))


test = jvfm(image, jnp.arange(3))


# %%
measured = jvfm(image, jnp.arange(100))
# %%


@jax.jit
def extract_slices(image, beginnings):
    return jax.lax.dynamic_slice(image, beginnings, (64, 64))


@jax.jit
def compute_power_spectrum(slices):
    # jnp.fft.fft2 handles N-dimensional arrays natively.
    # We specify axes=(-2, -1) to compute FFT over the 64x64 patches.
    return jnp.abs(jnp.fft.fft2(slices, axes=(-2, -1))) ** 2
