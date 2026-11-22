"""Static DM settings; changing schedule gates does not change this config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DMConfig:
    n_inner: int = 3
    beta: float = 1.0
    alpha_object: float = 1e-3
    alpha_probe: float = 1e-3
    object_cg_iterations: int = 5
    object_cg_tolerance: float = 1e-5
    probe_start: int = 10
    global_weights_start: int = 30
    per_position_weights_start: int = 50
    weight_strategy: str = "staged"
    weight_blend: float = 1.0
    weight_eps_relative: float = 1e-8
    fourier_eps_relative: float = 1e-8
    collapse_threshold: float = 1e-3
    chunk_size: int = 0

    def __post_init__(self):
        if self.n_inner < 1 or self.object_cg_iterations < 1 or self.beta == 0:
            raise ValueError("Positive inner/CG iterations and nonzero beta required.")
        if min(self.alpha_object, self.alpha_probe, self.weight_eps_relative) < 0:
            raise ValueError("Damping and weight epsilon cannot be negative.")
        if self.weight_strategy not in ("staged", "fixed", "global", "per_position"):
            raise ValueError("Invalid weight strategy.")
        if not 0 <= self.weight_blend <= 1:
            raise ValueError("weight_blend must be in [0,1].")
