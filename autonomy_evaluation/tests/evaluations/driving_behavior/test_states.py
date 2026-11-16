# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for reading the states of the ego vehicle and the other road users of the driving behavior evaluation."""

from __future__ import annotations

import math

import pytest
from autonomy_evaluation.evaluations.driving_behavior.states import ego_state, object_states, to_ego_frame
from autonomy_evaluation.evaluations.Metric import Sample
from perception_msgs.msg import EGO
from tests.evaluations.driving_behavior.messages import ego_data, object_list, road_user, static_tf_buffer


def _sample(ego, objects=None, tf_buffer=None, ego_object_id: int = -1) -> Sample:
    """Hold the messages of a sample, configured like the driving behavior evaluation."""
    return Sample(
        {"ego_data": ego, "objects": objects if objects is not None else object_list()},
        config={"ego_object_id": ego_object_id},
        tf_buffer=tf_buffer,
    )


class TestEgoState:
    """Tests reading the state of the ego vehicle."""

    def test_reads_the_kinematics_of_the_ego_vehicle(self):
        """The set entries of the state are read along with the extent of the ego vehicle."""
        ego = ego_state(_sample(ego_data(x=1.0, y=2.0, yaw=0.5, vel_lon=10.0, acc_lon=-2.0, yaw_rate=0.1, length=4.5)))

        assert (ego.x, ego.y, ego.yaw, ego.length, ego.width) == (1.0, 2.0, 0.5, 4.5, 2.0)
        assert (ego.vel_lon, ego.acc_lon, ego.yaw_rate) == (10.0, -2.0, 0.1)

    def test_entries_marked_as_not_set_are_unavailable(self):
        """An entry perception_msgs marks as not set in the state covariance is no value, e.g. a dataset without accelerations."""
        ego = ego_state(_sample(ego_data(vel_lon=10.0)))

        assert ego.vel_lat is None
        assert ego.acc_lon is ego.acc_lat is ego.yaw_rate is None

    def test_state_without_covariance_has_all_entries_set(self):
        """A simulation publishing no covariance sets every entry of the state."""
        message = ego_data()
        message.state.continuous_state[EGO.ACC_LON] = 1.5
        message.state.continuous_state_covariance = []

        assert ego_state(_sample(message)).acc_lon == 1.5

    def test_position_is_moved_to_the_geometric_center(self):
        """A position given at the rear axle is moved forward along the heading to the geometric center."""
        ego = ego_state(_sample(ego_data(x=10.0, y=0.0, yaw=math.pi / 2.0, translation_to_geometric_center=(1.5, 0.0))))

        assert (ego.x, ego.y) == pytest.approx((10.0, 1.5))

    def test_velocity_is_rotated_into_the_frame_of_the_position(self):
        """A velocity along the heading points into the direction of the yaw."""
        ego = ego_state(_sample(ego_data(yaw=math.pi / 2.0, vel_lon=10.0, vel_lat=1.0)))

        assert ego.velocity == pytest.approx((-1.0, 10.0))
        assert ego.speed == pytest.approx(math.hypot(10.0, 1.0))

    def test_unset_velocity_is_unavailable(self):
        """Without velocity, neither the velocity vector nor the speed is known."""
        ego = ego_state(_sample(ego_data()))

        assert ego.velocity is None
        assert ego.speed is None


class TestObjectStates:
    """Tests reading the states of the other road users."""

    def test_reads_the_footprints_and_velocities_of_the_road_users(self):
        """Each road user is an oriented rectangle moving over ground."""
        objects = object_states(_sample(ego_data(), object_list([road_user(7, 10.0, 2.0, yaw=math.pi / 2.0, vel_lon=3.0)])))

        assert objects.ids.tolist() == [7]
        assert (objects.x[0], objects.y[0], objects.yaw[0]) == (10.0, 2.0, math.pi / 2.0)
        assert (objects.length[0], objects.width[0]) == (4.0, 2.0)
        assert (objects.vx[0], objects.vy[0]) == pytest.approx((0.0, 3.0))

    def test_road_user_without_velocity_stands_still(self):
        """A velocity marked as not set is taken as zero."""
        objects = object_states(_sample(ego_data(), object_list([road_user(1, 10.0, 0.0, vel_lon=None, vel_lat=None)])))

        assert (objects.vx[0], objects.vy[0]) == (0.0, 0.0)

    def test_ego_vehicle_listed_among_the_objects_is_left_out(self):
        """A simulation that lists the ego vehicle among the objects identifies it by its ID."""
        objects = object_list([road_user(0, 0.0, 0.0), road_user(1, 10.0, 0.0)])

        assert object_states(_sample(ego_data(), objects, ego_object_id=0)).ids.tolist() == [1]
        assert object_states(_sample(ego_data(), objects)).ids.tolist() == [0, 1]

    def test_objects_are_transformed_into_the_frame_of_the_ego_vehicle(self):
        """Objects in the frame of a sensor are compared with the ego vehicle in its frame."""
        tf_buffer = static_tf_buffer("map", "lidar", x=100.0, yaw=math.pi / 2.0)
        sample = _sample(ego_data(), object_list([road_user(1, 10.0, 0.0, vel_lon=5.0)], frame_id="lidar"), tf_buffer)

        objects = object_states(sample)

        assert (objects.x[0], objects.y[0]) == pytest.approx((100.0, 10.0))
        assert (objects.vx[0], objects.vy[0]) == pytest.approx((0.0, 5.0))
        assert objects.header.frame_id == "map"

    def test_rejects_objects_that_cannot_be_transformed(self):
        """Objects in a frame without transform cannot be compared with the ego vehicle."""
        sample = _sample(ego_data(), object_list([road_user(1, 10.0, 0.0)], frame_id="lidar"))

        with pytest.raises(ValueError, match="'lidar' cannot be transformed into frame 'map'"):
            object_states(sample)

    def test_positions_in_the_frame_of_the_ego_vehicle(self):
        """A road user ahead of an ego vehicle heading north lies on its x axis."""
        ego = ego_state(_sample(ego_data(x=5.0, y=5.0, yaw=math.pi / 2.0)))

        assert to_ego_frame(ego, 5.0, 15.0) == pytest.approx((10.0, 0.0))
