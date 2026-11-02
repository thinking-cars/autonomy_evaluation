# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Client receiving the Lanelet2 map of the current scene from the parameters of a map server node.

``lanelet2_map_server`` distributes a map, and ``autonomy_datasets`` the map of
each scene it publishes, as the ROS parameters ``map_contents`` (OSM XML),
``origin_lat``, ``origin_lon`` and ``map_frame_id`` of their node. The client
requests these parameters once the map server is available, and follows every
change of them announced on ``/parameter_events``, e.g. when a dataset
continues with a scene recorded elsewhere. ``autonomy_datasets`` changes the map
before it publishes the first sample of a scene, so the map of a scene is loaded
before the samples of the scene are evaluated.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from autonomy_evaluation.utils.Lanelet2Map import Lanelet2Map
from rcl_interfaces.msg import ParameterEvent
from rclpy.clock import Clock, ClockType
from rclpy.parameter import parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient
from rclpy.task import Future

# Parameters of the map server holding the map
MAP_PARAMETERS = ("map_contents", "origin_lat", "origin_lon", "map_frame_id")

# Interval in seconds at which the client checks whether the map server is available to request the map from
_REQUEST_PERIOD_S = 1.0

# Interval in seconds at which waiting for the map server is logged
_WAIT_LOG_INTERVAL_S = 10.0


class Lanelet2MapClient:
    """Keeps the Lanelet2 map a map server node holds in its parameters.

    Attributes:
        server: Fully qualified name of the map server node, e.g. ``/datasets``.
        map: The current map, ``None`` while none has been received or the map
            server holds a map without lanelets.
    """

    def __init__(self, node: Any, server: str, load: Callable[..., Optional[Lanelet2Map]] = Lanelet2Map.from_osm):
        """Request the map from the map server once it is available, and follow its changes.

        Args:
            node (Any): node the client receives the map for
            server (str): name of the map server node, made fully qualified if relative
            load (Callable[..., Optional[Lanelet2Map]], optional): loads a map from the
                ``map_contents``, ``origin_lat``, ``origin_lon`` and ``map_frame_id`` parameters
        """
        self.node = node
        self.server = server if server.startswith("/") else f"/{server}"
        self.load = load
        self.map: Optional[Lanelet2Map] = None
        self.values: Dict[str, Any] = {}
        self.pending_request: Optional[Future] = None
        self.parameter_client = AsyncParameterClient(node, self.server)
        self.parameter_event_subscription = self.parameter_client.on_parameter_event(self.parameter_event_callback)
        # driven by a steady clock, so that the map is requested while the simulation clock stands still
        self.request_timer = node.create_timer(
            _REQUEST_PERIOD_S, self.request_parameters, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    def request_parameters(self):
        """Requests the map parameters once the map server is available, until they have been received"""
        if self.values or self.pending_request is not None:
            return
        if not self.parameter_client.services_are_ready():
            self.node.get_logger().warn(
                f"Waiting for map server '{self.server}' to receive the Lanelet2 map...",
                throttle_duration_sec=_WAIT_LOG_INTERVAL_S,
            )
            return
        self.pending_request = self.parameter_client.get_parameters(list(MAP_PARAMETERS))
        self.pending_request.add_done_callback(self.parameters_received)

    def parameters_received(self, future: Future):
        """Loads the map from the requested parameters

        Args:
            future (Future): future of the request, holding the parameter values of the map server
        """
        self.pending_request = None
        try:
            response = future.result()
        except Exception as exception:
            self.node.get_logger().warn(f"Requesting the Lanelet2 map from '{self.server}' failed, retrying: {exception}")
            return
        if len(response.values) != len(MAP_PARAMETERS):
            self.node.get_logger().warn(f"Map server '{self.server}' does not hold the parameters {list(MAP_PARAMETERS)}")
            return
        self.request_timer.cancel()
        self.update({name: parameter_value_to_python(value) for name, value in zip(MAP_PARAMETERS, response.values)})

    def parameter_event_callback(self, event: ParameterEvent):
        """Loads the map whenever the map server changes its parameters

        Args:
            event (ParameterEvent): parameters a node declared or changed
        """
        if event.node != self.server:
            return
        values = {
            parameter.name: parameter_value_to_python(parameter.value)
            for parameter in [*event.new_parameters, *event.changed_parameters]
            if parameter.name in MAP_PARAMETERS
        }
        if values:
            self.update(values)

    def update(self, values: Dict[str, Any]):
        """Loads the map anew if its parameters changed

        Args:
            values (Dict[str, Any]): new values of map parameters by name, None for an unset parameter
        """
        changed = any(self.values.get(name) != value for name, value in values.items()) or not self.values
        self.values.update(values)
        if self.request_timer is not None:
            self.request_timer.cancel()
        if not changed:
            return
        contents = self.values.get("map_contents")
        if not contents:
            self.map = None
            return
        try:
            self.map = self.load(
                contents,
                float(self.values.get("origin_lat") or 0.0),
                float(self.values.get("origin_lon") or 0.0),
                self.values.get("map_frame_id") or "map",
            )
        except ValueError as exception:
            self.map = None
            self.node.get_logger().error(f"Cannot load the Lanelet2 map of '{self.server}', evaluating without it: {exception}")
            return
        if self.map is None:
            self.node.get_logger().info(f"Map server '{self.server}' holds no lanelets, evaluating without map")
        else:
            self.node.get_logger().info(
                f"Loaded the Lanelet2 map of '{self.server}' with {self.map.num_lanelets} lanelets in frame '{self.map.frame_id}'"
            )
