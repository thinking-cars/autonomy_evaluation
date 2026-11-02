# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for receiving the Lanelet2 map of the current scene from the parameters of a map server node.

The client follows the parameters a map server, e.g. autonomy_datasets, holds the map of the current
scene in: it requests them once, and loads the map anew whenever they change. The ROS communication
itself is stubbed, so the tests cover which parameter values load which map.
"""

from __future__ import annotations

from types import SimpleNamespace

from autonomy_evaluation.lanelet2_map_client import Lanelet2MapClient, MAP_PARAMETERS
from rcl_interfaces.msg import Parameter, ParameterEvent
from rclpy.parameter import Parameter as RclpyParameter


class _FakeLogger:
    """Collect the messages the client logs."""

    def __init__(self):
        """Start with an empty log."""
        self.messages: list = []

    def info(self, message: str, **kwargs) -> None:
        """Record a logged message, whatever its severity."""
        self.messages.append(message)

    warn = error = info


def _client(load=None) -> SimpleNamespace:
    """Stub the state of a client of the map server '/datasets', recording the maps it loads."""
    loaded: list = []

    def fake_load(contents, origin_lat, origin_lon, frame_id):
        loaded.append((contents, origin_lat, origin_lon, frame_id))
        return SimpleNamespace(num_lanelets=1, frame_id=frame_id, contents=contents)

    client = SimpleNamespace(
        server="/datasets",
        load=load or fake_load,
        loaded=loaded,
        map=None,
        values={},
        loaded_values=None,
        pending_request=None,
        request_timer=SimpleNamespace(cancel=lambda: None),
        node=SimpleNamespace(get_logger=lambda logger=_FakeLogger(): logger),
    )
    client.update = lambda values: Lanelet2MapClient.update(client, values)
    client.known = lambda: Lanelet2MapClient.known(client)
    return client


def _parameter(name: str, value) -> Parameter:
    """Build the message of a parameter of the given value."""
    return Parameter(name=name, value=RclpyParameter(name, value=value).get_parameter_value())


def _event(node: str, **values) -> ParameterEvent:
    """Build the event of a node changing the given parameters."""
    return ParameterEvent(node=node, changed_parameters=[_parameter(name, value) for name, value in values.items()])


_MAP = {"map_contents": "<osm/>", "origin_lat": 50.78, "origin_lon": 6.06, "map_frame_id": "map"}


class TestLanelet2MapClient:
    """Tests loading the map whenever the parameters of the map server change."""

    def test_loads_the_map_of_the_requested_parameters(self):
        """The parameters requested once the map server is available load its map."""
        client = _client()
        response = SimpleNamespace(values=[_parameter(name, _MAP[name]).value for name in MAP_PARAMETERS])

        Lanelet2MapClient.parameters_received(client, SimpleNamespace(result=lambda: response))

        assert client.loaded == [("<osm/>", 50.78, 6.06, "map")]
        assert client.map.contents == "<osm/>"

    def test_ignores_a_response_without_the_map_parameters(self):
        """A node that is no map server does not hold the map parameters."""
        client = _client()

        Lanelet2MapClient.parameters_received(client, SimpleNamespace(result=lambda: SimpleNamespace(values=[])))

        assert client.loaded == []

    def test_loads_the_map_of_a_new_scene(self):
        """A dataset continuing with a scene recorded elsewhere changes map and origin at once."""
        client = _client()
        client.update(dict(_MAP))

        Lanelet2MapClient.parameter_event_callback(client, _event("/datasets", map_contents="<osm>next</osm>", origin_lat=48.1))

        assert client.loaded[-1] == ("<osm>next</osm>", 48.1, 6.06, "map")

    def test_loads_the_map_once_a_starting_map_server_declared_all_its_parameters(self):
        """A map server starting after the client declares its parameters one by one, which load the map once."""
        client = _client()

        for name in MAP_PARAMETERS:
            Lanelet2MapClient.parameter_event_callback(client, _event("/datasets", **{name: _MAP[name]}))

        assert client.loaded == [("<osm/>", 50.78, 6.06, "map")]

    def test_requests_the_parameters_a_map_server_has_not_announced(self):
        """A map server that announced only some map parameters is asked for all of them."""
        client = _client()
        requested = []
        client.parameter_client = SimpleNamespace(
            services_are_ready=lambda: True,
            get_parameters=lambda names: requested.append(names) or SimpleNamespace(add_done_callback=lambda callback: None),
        )
        client.parameters_received = None
        Lanelet2MapClient.parameter_event_callback(client, _event("/datasets", map_contents="<osm/>"))

        Lanelet2MapClient.request_parameters(client)

        assert requested == [list(MAP_PARAMETERS)]

    def test_ignores_the_parameters_of_other_nodes(self):
        """Only the map server holds the map."""
        client = _client()

        Lanelet2MapClient.parameter_event_callback(client, _event("/lanelet2_map_server", map_contents="<osm/>"))

        assert client.loaded == []

    def test_unchanged_parameters_do_not_reload_the_map(self):
        """Loading a map takes time, so it is only loaded when it changes."""
        client = _client()
        client.update(dict(_MAP))

        client.update({"map_contents": "<osm/>"})

        assert len(client.loaded) == 1

    def test_blank_map_is_no_map(self):
        """A map server without map, e.g. a dataset whose scenes have none, provides no map."""
        client = _client()
        client.update(dict(_MAP))

        client.update({"map_contents": ""})

        assert client.map is None

    def test_map_that_cannot_be_loaded_is_no_map(self):
        """The evaluation continues without map rather than stopping on a broken one."""

        def failing_load(*args):
            raise ValueError("broken map")

        client = _client(load=failing_load)

        client.update(dict(_MAP))

        assert client.map is None
        assert "broken map" in client.node.get_logger().messages[-1]
