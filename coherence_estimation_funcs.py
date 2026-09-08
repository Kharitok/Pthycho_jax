# %%
import jax
import jax.numpy as jnp


def compute_ensemble_mutual_intensity(
    probe_modes: jnp.ndarray, modal_weights: jnp.ndarray  # [N, Y, X]  # [P, N, M]
) -> jnp.ndarray:
    """
    Constructs the spatial Mutual Intensity Matrix J [Y*X, Y*X]
    averaged over all scan positions P.
    """
    P, N, M = modal_weights.shape
    Y, X = probe_modes.shape[1:]
    K = Y * X

    # 1. Generate full propagating probes across all positions: [P, M, Y, X]
    # P_m = sum_n Phi_n * W_nm
    full_probes = jnp.einsum("nxy,pnm->pmxy", probe_modes, modal_weights)

    # 2. Reshape spatial dimensions to 1D vectors: [P*M, K]
    probes_flat = full_probes.reshape(P * M, K)

    # 3. Compute J = (1/P) * sum( P_k^T @ P_k* ): [K, K]
    J = (probes_flat.T @ jnp.conj(probes_flat)) / P
    return J


def evaluate_coherence_reconstruction(
    modes_sim: jnp.ndarray,  # Ground truth [N_sim, Y, X]
    weights_sim: jnp.ndarray,  # Ground truth [P, N_sim, M_sim]
    modes_rec: jnp.ndarray,  # Reconstructed [N_rec, Y, X]
    weights_rec: jnp.ndarray,  # Reconstructed [P, N_rec, M_rec]
) -> dict:
    """
    Compares simulated vs reconstructed coherence metrics in a gauge-invariant way.
    """
    # 1. Build Mutual Intensity Matrices
    J_sim = compute_ensemble_mutual_intensity(modes_sim, weights_sim)
    J_rec = compute_ensemble_mutual_intensity(modes_rec, weights_rec)

    # 2. Compute Eigenvalues (Mode Energies)
    eigvals_sim = jnp.linalg.eigvalsh(J_sim)[::-1]  # Sort descending
    eigvals_rec = jnp.linalg.eigvalsh(J_rec)[::-1]

    # Truncate near-zero tail for noise cleanliness
    k_max = min(10, len(eigvals_sim))
    power_sim = jnp.maximum(0.0, eigvals_sim[:k_max]) / jnp.sum(eigvals_sim)
    power_rec = jnp.maximum(0.0, eigvals_rec[:k_max]) / jnp.sum(eigvals_rec)

    # 3. Global Degree of Coherence (mu)
    mu_sim = jnp.sqrt(jnp.sum(eigvals_sim**2)) / jnp.sum(eigvals_sim)
    mu_rec = jnp.sqrt(jnp.sum(eigvals_rec**2)) / jnp.sum(eigvals_rec)

    # 4. Mutual Intensity Operator Fidelity
    # Tr(A B) = sum( A_ij * B_ji )
    overlap = jnp.real(jnp.sum(J_sim * jnp.conj(J_rec.T)))
    norm_sim = jnp.real(jnp.sum(J_sim * jnp.conj(J_sim.T)))
    norm_rec = jnp.real(jnp.sum(J_rec * jnp.conj(J_rec.T)))

    fidelity = overlap / jnp.sqrt(norm_sim * norm_rec + 1e-12)

    return {
        "fidelity": float(fidelity),
        "degree_of_coherence_sim": float(mu_sim),
        "degree_of_coherence_rec": float(mu_rec),
        "eigenvalues_sim": power_sim,
        "eigenvalues_rec": power_rec,
    }
