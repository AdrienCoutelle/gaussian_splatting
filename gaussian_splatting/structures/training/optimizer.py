from typing import Any

import mlx.core as mx
import mlx.optimizers as opt
from pydantic import BaseModel, ConfigDict

Params = dict[str, mx.array]


class LearningRatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr_positions: float
    lr_opacities: float
    lr_scales: float
    lr_quaternions: float
    lr_sh_coeffs: float


class GaussianOptimizer:
    """One Adam optimizer per Gaussian parameter group, each with its own learning rate."""

    def __init__(
        self,
        learning_rates: LearningRatesConfig,
    ) -> None:
        self._optimizers = {
            "positions": opt.Adam(learning_rate=learning_rates.lr_positions),
            "opacities": opt.Adam(learning_rate=learning_rates.lr_opacities),
            "scales": opt.Adam(learning_rate=learning_rates.lr_scales),
            "quaternions": opt.Adam(learning_rate=learning_rates.lr_quaternions),
            "sh_coeffs": opt.Adam(learning_rate=learning_rates.lr_sh_coeffs),
        }

    def step(
        self,
        params: Params,
        grads: Params,
    ) -> Params:
        """Applies one optimization step and returns the updated parameters."""
        updated = dict(params)
        for name, optimizer in self._optimizers.items():
            param_dict = {name: params[name]}
            optimizer.update(param_dict, {name: grads[name]})
            updated[name] = param_dict[name]
        return updated

    def reindex(
        self,
        indices: mx.array,
        num_gaussians: int,
    ) -> None:
        """Keeps the optimizer state aligned with the Gaussians after pruning / cloning / splitting.

        Every per-Gaussian state tensor (first dim == `num_gaussians`) is gathered with `indices`.
        """
        for optimizer in self._optimizers.values():
            optimizer.state = self._reindex_state(optimizer.state, indices, num_gaussians)

    def reset_state(
        self,
        name: str,
        num_gaussians: int,
    ) -> None:
        """Zeroes the per-Gaussian state (e.g. Adam moments) of one parameter group."""
        optimizer = self._optimizers[name]
        optimizer.state = self._reset_state(optimizer.state, num_gaussians)

    @classmethod
    def _reset_state(
        cls,
        state: Any,
        num_gaussians: int,
    ) -> Any:
        if isinstance(state, mx.array):
            if state.ndim > 0 and state.shape[0] == num_gaussians:
                return mx.zeros_like(state)
            return state
        if isinstance(state, dict):
            return {k: cls._reset_state(v, num_gaussians) for k, v in state.items()}
        if isinstance(state, list):
            return [cls._reset_state(v, num_gaussians) for v in state]
        return state

    @classmethod
    def _reindex_state(
        cls,
        state: Any,
        indices: mx.array,
        num_gaussians: int,
    ) -> Any:
        if isinstance(state, mx.array):
            if state.ndim > 0 and state.shape[0] == num_gaussians:
                return state[indices]
            return state
        if isinstance(state, dict):
            return {k: cls._reindex_state(v, indices, num_gaussians) for k, v in state.items()}
        if isinstance(state, list):
            return [cls._reindex_state(v, indices, num_gaussians) for v in state]
        return state
