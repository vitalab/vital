import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.signal import convolve

logger = logging.getLogger(__name__)


class Snake:
    """Filters a signal by minimizing energies on the signal, an approach inspired by active contours."""

    @dataclass
    class SnakeState:
        """Collection of variables that are required to track the state of the snake's optimization process."""

        signal: np.ndarray
        velocity: np.ndarray = field(init=False)
        step: int = field(default=0, init=False)

        def __post_init__(self):  # noqa: D105
            self.velocity = np.zeros_like(self.signal)

    def __init__(
        self,
        grad_step: float = 1,
        momentum: float = 0.9,
        smoothness_weight: float = 0,
        num_neighbors: int = 1,
        neighbors_pad_mode: str = "edge",
        max_iterations: int = int(1e5),
        convergence_eps: float = np.finfo(np.float32).eps,
    ):
        """Initializes class instance.

        Args:
            grad_step: Step size to take at each optimization iteration.
            momentum: Momentum factor.
            smoothness_weight: Weight of the loss' smoothness term.
            num_neighbors: Number of neighbors to take into account when computing the smoothness term. This also
                dictates the width by which to pad the signal at the edges.
            neighbors_pad_mode: Mode used to determine how to pad points at the beginning/end of the array. The options
                available are those of the ``mode`` parameter of ``numpy.pad``.
            max_iterations: If the model doesn't converge to a stable configuration, number of steps after which to
                stop.
            convergence_eps: Threshold on the L2 norm between consecutive optimization steps below which the algorithm
                is considered to have converged.
        """
        if num_neighbors < 1:
            raise ValueError(
                f"``num_neighbors`` should always be > 1, but you set it to: {num_neighbors}. If your goal was to "
                f"disable local smoothness, the supported way to do this is to set ``smoothness_weight=0``."
            )

        self._grad_step = grad_step
        self._momentum = momentum
        self._max_iterations = max_iterations
        self._convergence_eps = convergence_eps
        self._grad_weights = {"smoothness": smoothness_weight}

        # Precompute kwargs for signal padding
        self._pad_kwargs = {"pad_width": ((num_neighbors, num_neighbors), (0, 0)), "mode": neighbors_pad_mode}

        # Precompute convolution kernel for weighted neighbor average
        neighbors_weights = [1 / neighbor_pos for neighbor_pos in range(1, num_neighbors + 1)]
        self._neighbors_win = np.array(list(reversed(neighbors_weights)) + [0] + neighbors_weights)[:, None]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Filters a signal by minimizing energies on the signal, an approach inspired by active contours.

        Args:
            signal: (n_samples, [n_features]), Noisy signal(s) to filter.

        Returns:
            (n_samples, [n_features]), Filtered signal(s) that minimizes different energies/constraints.
        """
        # Add any missing optional dimension to make sure the data is in (n_samples, n_features) format
        in_shape = signal.shape
        if signal.ndim == 1:
            signal = signal.reshape((len(signal), 1))
        elif signal.ndim > 2:
            raise ValueError(
                f"{self.__class__.__name__} received invalid `signal` input data. `signal` input data should have 1 or "
                f"2 dims for (n_samples, [n_features]). The input data it received was of shape: {signal.shape}."
            )

        smoothed_signal, state = self._optimize(signal)
        self._validate_optim_result(smoothed_signal, state)
        return smoothed_signal.reshape(in_shape)

    def _optimize(self, signal: np.ndarray) -> Tuple[np.ndarray, SnakeState]:
        """Runs the main gradient descent loop of the optimization process.

        Args:
            signal: (n_samples, n_features), Noisy signal(s) to filter.

        Returns:
            - (n_samples, n_features), Filtered signal(s) that minimizes different energies/constraints,
            - Collection of indicators tracking the state of the snake's optimization process.
        """
        # Init variables that will track the optimization's state
        state = self.SnakeState(signal)
        cur_signal = signal

        while True:
            # Compute weighted sum of gradients
            grad = sum(
                grad_val * self._grad_weights.get(grad_name, 1)
                for grad_name, grad_val in self._compute_grad(cur_signal, state).items()
            )
            # Update the points according to the computed gradients
            state.velocity = self._momentum * state.velocity + (1 - self._momentum) * grad
            next_signal = cur_signal - (self._grad_step * state.velocity)

            # Check if we reached one of our convergence/stop conditions
            if not self._continue_optim_loop(next_signal, cur_signal, state):
                break

            # Update variables in preparation of the next step
            cur_signal = next_signal
            state.step += 1

        return cur_signal, state

    def _continue_optim_loop(self, next_signal: np.ndarray, cur_signal: np.ndarray, state: SnakeState) -> bool:
        """Checks if the gradient descent optimization should continue for another step.

        Args:
            next_signal: (n_samples, n_features), Values of the signal after the update at the end of the current step.
                It is sometimes useful to check for convergence/stop conditions.
            cur_signal: (n_samples, n_features), Values of the signal at the current step, i.e. values that were used to
                compute the gradients.
            state: Collection of indicators tracking the state of the snake's optimization process.

        Returns:
            ``True`` if the conditions for another optimization step are met, ``False`` otherwise.
        """
        no_stop_reached = self._max_iterations is None or state.step < self._max_iterations
        is_next_step_diff = np.linalg.norm(next_signal - cur_signal, ord=2) > self._convergence_eps
        return no_stop_reached and is_next_step_diff

    def _compute_grad(self, signal: np.ndarray, state: SnakeState) -> Dict[str, np.ndarray]:
        """Computes the different components of the gradients to update the points.

        Args:
            signal: (n_samples, n_features), Values of the signal at the current step.
            state: Collection of indicators tracking the state of the snake's optimization process.

        Returns:
            Mapping between gradient descriptions and their values for the current step (prior to weighting them).
        """
        # 1. Difference to reference points
        diff_grad = signal - state.signal
        # 2. Global smoothness term
        padded_signal = np.pad(signal, **self._pad_kwargs)
        neighbors_avg = convolve(padded_signal, self._neighbors_win, mode="valid") / sum(self._neighbors_win)
        smoothness_grad = signal - neighbors_avg

        return {"diff": diff_grad, "smoothness": smoothness_grad}

    def _validate_optim_result(self, signal: np.ndarray, state: SnakeState) -> None:
        """Callback to check the result after the end of the optimization process.

        This can be useful to raise errors/warnings, e.g. optimization finished w/o converging.

        Args:
            signal: (n_samples, n_features), Filtered signal(s) obtained by optimizing different energies/constraints.
            state: Collection of indicators tracking the state of the snake's optimization process.
        """
        if self._max_iterations and state.step == self._max_iterations:
            logger.warning(
                f"{self.__class__.__name__} optimization reached the maximum number of iterations configured and was "
                f"stopped early."
            )


class _ConstrainedSnake(Snake):
    """Abstraction for snake variants that require the computation of a 'hard' smoothness constraint."""

    def __init__(self, smoothness_constraint_func: Callable[[np.ndarray], np.ndarray], **kwargs):
        """Initializes class instance.

        Args:
            smoothness_constraint_func: Function that computes whether each point violates a "hard" smoothness
                constraint (`True`) or satisfies the constraint (`False`).
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self._smoothness_constraint_func = smoothness_constraint_func

    def _validate_optim_result(self, signal: np.ndarray, state: Snake.SnakeState) -> None:
        """Adds a warning in case the optimization ended w/o satisfying the smoothness constraint."""
        super()._validate_optim_result(signal, state)
        if self._smoothness_constraint_func(signal).any():
            logger.warning(f"{self.__class__.__name__} optimization was not able to enforce the smoothness constraint.")


class PenaltySnake(_ConstrainedSnake):
    """Snake + penalty function to ensure a smoothness constraint is satisfied by all the points in the signal."""

    def _compute_grad(self, signal: np.ndarray, state: Snake.SnakeState) -> Dict[str, np.ndarray]:
        """Adds a penalty term on specific points violating the smoothness constraint."""
        grads = super()._compute_grad(signal, state)
        smoothness_grad = grads["smoothness"]
        # Compute the gradient associated to the smoothness penalty by:
        # 1) Masking the global smoothness to keep it only on points that fail the smoothness constraint
        # 2) Scale the gradient w.r.t. the penalty parameter schedule, so that it gets "sharper" during optim.
        penalty = self._smoothness_constraint_func(signal)
        smoothness_penalty_grad = smoothness_grad * penalty * (state.step + 1)
        grads["smoothness_penalty"] = smoothness_penalty_grad
        return grads

    def _continue_optim_loop(self, next_signal: np.ndarray, cur_signal: np.ndarray, state: Snake.SnakeState) -> bool:
        """Adds an early stopping condition in case an update would violate already satisfied penalty constraints."""
        continue_optim = super()._continue_optim_loop(next_signal, cur_signal, state)
        return continue_optim and (
            self._smoothness_constraint_func(cur_signal).any()
            or not self._smoothness_constraint_func(next_signal).any()
        )


class DualLagrangianRelaxationSnake(_ConstrainedSnake):
    """Snake + optimization of the smoothness weight to enforce the constraint for all the points in the signal."""

    def __init__(
        self,
        smoothness_weight_bounds: Tuple[int, int] = (0, 128),
        smoothness_weight_search_depth: int = 5,
        smoothness_weight_margin: float = 0,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            smoothness_weight_bounds: Min and max bounds between which to search for the optimal smoothness weight.
            smoothness_weight_search_depth: Number of iterations of binary search to run before settling for an
                "optimal" smoothness weight.
            smoothness_weight_margin: Since the binary search finds a value that lies right at the edge of the feasible
                domain, this parameter configures a margin between the "optimal" value at the edge of the feasible
                domain and a safer value to choose in practice. The margin is used as a factor on the "optimal" value,
                according to the following rule:
                ``smoothness_weight = optimal_smoothness_weight * (1 + smoothness_weight_margin)``.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self._smoothness_weight_bounds = smoothness_weight_bounds
        self._smoothness_search_depth = smoothness_weight_search_depth
        self._smoothness_weight_margin = smoothness_weight_margin

    def _optimize(self, signal: np.ndarray) -> Tuple[np.ndarray, Snake.SnakeState]:
        """Adds a binary search for the smallest smoothness weight that enforces the constraint for all the points."""
        low, high = self._smoothness_weight_bounds
        for _ in range(self._smoothness_search_depth):
            self._grad_weights["smoothness"] = (low + high) / 2
            smoothed_signal, _ = super()._optimize(signal)
            if self._smoothness_constraint_func(smoothed_signal).any():
                low = self._grad_weights["smoothness"]
            else:
                high = self._grad_weights["smoothness"]

        self._grad_weights["smoothness"] = high * (1 + self._smoothness_weight_margin)
        return super()._optimize(signal)
