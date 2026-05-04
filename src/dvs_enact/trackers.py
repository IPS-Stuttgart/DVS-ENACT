"""PyRecEst-based DVS-ENACT tracker variants."""

from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,too-many-locals
from pyrecest.backend import (
    arctan2,
    array,
    concatenate,
    cos,
    linalg,
    linspace,
    pi,
    sin,
    vstack,
)
from pyrecest.filters import FullSCGPTracker


class DVSFullSCGPTracker(FullSCGPTracker):
    """Star-convex GP tracker with a DVS-inspired active-contour update.

    Event cameras mostly observe moving brightness contours. A contour point is
    therefore most informative when the apparent image velocity has a component
    along the local contour normal. Measurements on nearly inactive contour
    parts can be skipped or strongly down-weighted instead of being interpreted
    as uniformly sampled extent returns.
    """

    def __init__(
        self,
        *args,
        event_activity_floor=1e-3,
        inactive_activity_threshold=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_activity_floor = float(event_activity_floor)
        self.inactive_activity_threshold = float(inactive_activity_threshold)
        if self.event_activity_floor <= 0.0:
            raise ValueError("event_activity_floor must be positive")
        if self.inactive_activity_threshold < 0.0:
            raise ValueError("inactive_activity_threshold must be non-negative")

        self.last_event_activities = None
        self.last_active_measurement_indices = None

    def _get_event_velocity(self, event_velocity):
        if event_velocity is None:
            if not self.velocities:
                raise ValueError(
                    "event_velocity must be provided when velocities=False"
                )
            return self.kinematic_state[3] * array(
                [cos(self.kinematic_state[2]), sin(self.kinematic_state[2])]
            )

        event_velocity = array(event_velocity)
        if event_velocity.shape != (self.measurement_dim,):
            raise ValueError("event_velocity must have shape (2,)")
        return event_velocity

    def _unit_direction_from_measurement(self, measurement):
        position = self.kinematic_state[:2]
        delta = measurement - position
        delta_norm = linalg.norm(delta)
        if float(delta_norm) <= 1e-12:
            return array([cos(self.kinematic_state[2]), sin(self.kinematic_state[2])])
        return delta / delta_norm

    def _contour_normal_from_unit_direction(self, unit_direction):
        orientation = self.kinematic_state[2]
        world_angle = arctan2(unit_direction[1], unit_direction[0])
        body_angle = world_angle - orientation
        basis_row = self._basis_matrix(body_angle)[0]
        basis_derivative_row = self._basis_derivative(body_angle)[0]
        radius = basis_row @ self.shape_state
        radius_derivative = basis_derivative_row @ self.shape_state

        tangent = radius_derivative * unit_direction + radius * array(
            [-unit_direction[1], unit_direction[0]]
        )
        normal = array([tangent[1], -tangent[0]])
        normal_norm = linalg.norm(normal)
        if float(normal_norm) <= 1e-12:
            return unit_direction
        return normal / normal_norm

    def event_activity_for_measurement(self, measurement, event_velocity=None):
        """Return normalized normal-flow activity for one event measurement."""
        measurement = array(measurement)
        if measurement.shape != (self.measurement_dim,):
            raise ValueError("measurement must have shape (2,)")

        velocity = self._get_event_velocity(event_velocity)
        velocity_norm = linalg.norm(velocity)
        if float(velocity_norm) <= 1e-12:
            return 0.0

        unit_direction = self._unit_direction_from_measurement(measurement)
        normal = self._contour_normal_from_unit_direction(unit_direction)
        return float(abs(normal @ velocity) / velocity_norm)

    def contour_event_activity(
        self,
        n=100,
        angles=None,
        event_velocity=None,
        body_frame=False,
        apply_floor=False,
    ):
        """Evaluate DVS contour activity over image-plane or body-frame angles."""
        if angles is None:
            angles = linspace(0.0, 2 * pi, n, endpoint=False)
        else:
            angles = array(angles)

        velocity = self._get_event_velocity(event_velocity)
        velocity_norm = linalg.norm(velocity)
        activities = []
        for angle in angles:
            world_angle = angle + self.kinematic_state[2] if body_frame else angle
            unit_direction = array([cos(world_angle), sin(world_angle)])
            normal = self._contour_normal_from_unit_direction(unit_direction)
            activity = (
                0.0
                if float(velocity_norm) <= 1e-12
                else float(abs(normal @ velocity) / velocity_norm)
            )
            if apply_floor and activity < self.event_activity_floor:
                activity = self.event_activity_floor
            activities.append(activity)
        return array(activities)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def update(
        self,
        measurements,
        R=None,
        s_hat=None,
        sigma_squared_s=None,
        event_velocity=None,
        event_activity_floor=None,
        inactive_activity_threshold=None,
    ):
        if s_hat is not None:
            self.scale_mean = float(s_hat)
        if sigma_squared_s is not None:
            self.scale_variance = float(sigma_squared_s)
        if event_activity_floor is None:
            event_activity_floor = self.event_activity_floor
        else:
            event_activity_floor = float(event_activity_floor)
        if inactive_activity_threshold is None:
            inactive_activity_threshold = self.inactive_activity_threshold
        else:
            inactive_activity_threshold = float(inactive_activity_threshold)
        if event_activity_floor <= 0.0:
            raise ValueError("event_activity_floor must be positive")
        if inactive_activity_threshold < 0.0:
            raise ValueError("inactive_activity_threshold must be non-negative")

        measurements = self._normalize_measurements(measurements)
        measurement_noise = self.measurement_noise
        if R is not None:
            measurement_noise = self._as_covariance_matrix(
                R,
                self.measurement_dim,
                "R",
                require_positive_semidefinite=False,
            )

        velocity = self._get_event_velocity(event_velocity)
        activities = []
        active_indices = []
        measurement_jacobians = []
        predicted_measurements = []
        noise_covariances = []
        for measurement_index, measurement in enumerate(measurements):
            activity = self.event_activity_for_measurement(measurement, velocity)
            activities.append(activity)
            if activity < inactive_activity_threshold:
                continue

            effective_activity = (
                activity if activity >= event_activity_floor else event_activity_floor
            )
            measurement_jacobian, predicted_measurement, noise_covariance = (
                self._measurement_model_terms(measurement, measurement_noise)
            )
            active_indices.append(measurement_index)
            measurement_jacobians.append(measurement_jacobian)
            predicted_measurements.append(predicted_measurement)
            noise_covariances.append(noise_covariance / effective_activity)

        self.last_event_activities = array(activities)
        self.last_active_measurement_indices = active_indices
        if not active_indices:
            self.last_quadratic_form = None
            return

        measurement_jacobian = vstack(measurement_jacobians)
        predicted_measurements = concatenate(predicted_measurements)
        noise_covariance = linalg.block_diag(*noise_covariances)
        residual = concatenate([measurements[index] for index in active_indices])
        residual = residual - predicted_measurements
        covariance_measurement = self._symmetrize(
            measurement_jacobian @ self.covariance @ measurement_jacobian.T
            + noise_covariance
        )
        cross_covariance = self.covariance @ measurement_jacobian.T
        gain = linalg.solve(covariance_measurement.T, cross_covariance.T).T
        self.state = self.state + gain @ residual
        self.covariance = self._symmetrize(
            self.covariance - gain @ covariance_measurement @ gain.T
        )
        self._sync_state_views()
        solved_residual = linalg.solve(covariance_measurement, residual)
        self.last_quadratic_form = residual @ solved_residual

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()


DVSSCGPTracker = DVSFullSCGPTracker
