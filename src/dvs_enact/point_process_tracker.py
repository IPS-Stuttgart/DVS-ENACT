"""Point-process SCGP tracker variant for DVS event batches."""

from __future__ import annotations

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array

from .event_likelihood import (
    ContourSample,
    PointProcessUpdateConfig,
    event_batch_log_likelihood_terms,
)
from .trackers import DVSFullSCGPTracker


class DVSPointProcessSCGPTracker(DVSFullSCGPTracker):
    """Experimental SCGP tracker using a point-process event likelihood.

    The update maximizes a contour-conditioned inhomogeneous point-process
    likelihood. It is intentionally kept separate from ``DVSFullSCGPTracker`` so
    that the current activity-weighted update remains available as a baseline.
    """

    def __init__(
        self,
        *args,
        point_process_update_config=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.point_process_update_config = (
            point_process_update_config or PointProcessUpdateConfig()
        )
        self.last_event_likelihood_terms = None
        self.last_event_likelihood_gradient = None
        self.last_event_likelihood_state_update = None
        self.last_event_log_likelihood = None

    def sample_contour(self, n=100):
        """Return sampled star-convex contour geometry for likelihood models."""
        if n <= 2:
            raise ValueError("n must be greater than 2")

        orientation = float(self.kinematic_state[2])
        position = np.asarray(self.kinematic_state[:2], dtype=float)
        shape_state = np.asarray(self.shape_state, dtype=float)
        body_angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        delta_angle = 2.0 * np.pi / float(n)
        points = []
        normals = []
        weights = []
        for body_angle in body_angles:
            world_angle = float(body_angle + orientation)
            unit_direction = np.array(
                [np.cos(world_angle), np.sin(world_angle)],
                dtype=float,
            )
            basis_row = np.asarray(
                self._basis_matrix(float(body_angle))[0],
                dtype=float,
            )
            derivative_row = np.asarray(
                self._basis_derivative(float(body_angle))[0],
                dtype=float,
            )
            radius = float(basis_row @ shape_state)
            radius_derivative = float(derivative_row @ shape_state)
            tangent = radius_derivative * unit_direction + radius * np.array(
                [-unit_direction[1], unit_direction[0]],
                dtype=float,
            )
            normal = np.array([tangent[1], -tangent[0]], dtype=float)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-12:
                normal = unit_direction
            else:
                normal = normal / normal_norm
            points.append(position + radius * unit_direction)
            normals.append(normal)
            weights.append(float(np.linalg.norm(tangent)) * delta_angle)

        return ContourSample(
            points=np.asarray(points, dtype=float),
            normals=np.asarray(normals, dtype=float),
            weights=np.asarray(weights, dtype=float),
            angles=body_angles,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def update(
        self,
        measurements,
        R=None,
        s_hat=None,
        sigma_squared_s=None,
        event_velocity=None,
        point_process_update_config=None,
        batch_duration=None,
        image_area=None,
    ):
        """Update the tracker state with the point-process event likelihood."""
        del R
        if s_hat is not None:
            self.scale_mean = float(s_hat)
        if sigma_squared_s is not None:
            self.scale_variance = float(sigma_squared_s)
        return self.update_event_batch(
            measurements,
            event_velocity=event_velocity,
            point_process_update_config=point_process_update_config,
            batch_duration=batch_duration,
            image_area=image_area,
        )

    def update_event_batch(
        self,
        measurements,
        *,
        event_velocity=None,
        point_process_update_config=None,
        batch_duration=None,
        image_area=None,
    ):
        """Run a MAP-style point-process likelihood update for one event batch."""
        config = point_process_update_config or self.point_process_update_config
        measurements = np.asarray(
            self._normalize_measurements(measurements),
            dtype=float,
        )
        velocity = np.asarray(self._get_event_velocity(event_velocity), dtype=float)
        if measurements.shape[0] == 0:
            self.last_event_likelihood_terms = self._likelihood_terms_for_current_state(
                measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
            return

        gradient = np.zeros_like(self._state_as_numpy())
        state_update = np.zeros_like(gradient)
        for _ in range(config.max_map_iterations):
            state = self._state_as_numpy()
            gradient = self._finite_difference_log_likelihood_gradient(
                state,
                measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
            covariance = np.asarray(self.covariance, dtype=float)
            state_update = config.map_step_size * (covariance @ gradient)
            state_update = self._clip_state_update(
                state_update,
                config.max_state_update_norm,
            )
            if float(np.linalg.norm(state_update)) <= 1e-12:
                break
            self._set_state_from_numpy(state + state_update)
            self.covariance = self._symmetrize(
                array(config.covariance_damping * np.asarray(self.covariance))
            )

        self.last_event_likelihood_terms = self._likelihood_terms_for_current_state(
            measurements,
            velocity,
            config,
            batch_duration,
            image_area,
        )
        self.last_event_likelihood_gradient = array(gradient)
        self.last_event_likelihood_state_update = array(state_update)
        self.last_event_activities = array(
            self.contour_event_activity(
                n=config.contour_samples,
                event_velocity=velocity,
            )
        )
        self.last_active_measurement_indices = list(range(measurements.shape[0]))
        self.last_event_log_likelihood = (
            self.last_event_likelihood_terms.log_likelihood
        )
        self.last_quadratic_form = None

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _finite_difference_log_likelihood_gradient(
        self,
        state,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        gradient = np.zeros_like(state)
        eps = config.finite_difference_eps
        for state_index in self._likelihood_state_indices(config):
            perturbation = np.zeros_like(state)
            perturbation[state_index] = eps
            plus = self._log_likelihood_for_state(
                state + perturbation,
                measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
            minus = self._log_likelihood_for_state(
                state - perturbation,
                measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
            gradient[state_index] = (plus - minus) / (2.0 * eps)
        return gradient

    def _log_likelihood_for_state(
        self,
        state,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        terms = self._likelihood_terms_for_state(
            state,
            measurements,
            velocity,
            config,
            batch_duration,
            image_area,
        )
        return terms.log_likelihood

    def _likelihood_terms_for_current_state(
        self,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        contour = self.sample_contour(config.contour_samples)
        return event_batch_log_likelihood_terms(
            measurements,
            contour,
            velocity,
            config.likelihood,
            batch_duration=batch_duration,
            image_area=image_area,
        )

    def _likelihood_terms_for_state(
        self,
        state,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        original_state = self._state_as_numpy()
        try:
            self._set_state_from_numpy(state)
            return self._likelihood_terms_for_current_state(
                measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
        finally:
            self._set_state_from_numpy(original_state)

    def _likelihood_state_indices(self, config):
        state_size = self._state_as_numpy().shape[0]
        kinematic_size = np.asarray(self.kinematic_state).shape[0]
        indices = [index for index in range(min(3, kinematic_size, state_size))]
        shape_start = kinematic_size
        available_shape_count = max(0, state_size - shape_start)
        shape_count = min(
            int(config.shape_update_modes),
            np.asarray(self.shape_state).shape[0],
            available_shape_count,
        )
        indices.extend(range(shape_start, shape_start + shape_count))
        return indices

    def _state_as_numpy(self):
        return np.asarray(self.state, dtype=float).copy()

    def _set_state_from_numpy(self, state):
        self.state = array(np.asarray(state, dtype=float))
        self._sync_state_views()

    @staticmethod
    def _clip_state_update(state_update, max_norm):
        update_norm = float(np.linalg.norm(state_update))
        if update_norm <= max_norm:
            return state_update
        return state_update * (float(max_norm) / update_norm)


DVSPointProcessSCGP = DVSPointProcessSCGPTracker
