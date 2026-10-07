# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the helpers of the autonomy_evaluation node.

The node itself drives the evaluation via the ``request_samples`` service of the dataset and is
covered by running it against a dataset. Tested here are the parsing of the samples to evaluate,
as an unparsable value stops the node, the topics the inputs of an evaluation are subscribed on,
the stamping of received messages, the matching of the received input messages into the samples
to evaluate, which has to hold up when the dataset continues with a scene that was recorded before
the scene played before it, has to match topics of a simulation within a tolerance and must not wait
for an optional topic that is not published, the evaluation of a sample, which requests the next
samples unless others publish them, advancing the
evaluation until it finishes, which it also has to once the dataset node has shut down after its
last sample, and the finalization of the results, which reports the samples of an interrupted
evaluation as incomplete.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
from autonomy_evaluation import autonomy_evaluation
from autonomy_evaluation.autonomy_evaluation import AutonomyEvaluation, parse_sample_ids, SampleSynchronizer

_TOPICS = ["prediction", "label", "label_meta_info"]

# stamps of the last sample of a scene and of the first samples of the scene the dataset continues
# with, which nuScenes recorded years earlier
_PREVIOUS_SCENE = (1537853053, 397270000)
_NEXT_SCENE = [(1531885320, 49418000), (1531885320, 548742000), (1531885321, 48634000)]


def _message(stamp: tuple[int, int]) -> SimpleNamespace:
    """Fake an input message stamped with the recording time of its sample."""
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp[0], nanosec=stamp[1])))


def _synchronizer(
    queue_size: int = 10, topics=None, tolerance: float = 0.0, optional_topics=(), published_topics=None
) -> tuple[SampleSynchronizer, list]:
    """Create a synchronizer of the evaluation inputs next to the list of the messages of the samples it matched.

    Of the optional topics, those in ``published_topics`` (all by default) have a publisher.
    """
    matched_samples: list = []
    synchronizer = SampleSynchronizer(
        topics or _TOPICS,
        callback=lambda stamp, messages: matched_samples.append(tuple(messages.values())),
        queue_size=queue_size,
        tolerance=tolerance,
        optional_topics=optional_topics,
        expects_message=lambda topic: published_topics is None or topic in published_topics,
    )
    return synchronizer, matched_samples


def _publish_sample(synchronizer: SampleSynchronizer, stamp: tuple[int, int], topics=None) -> dict:
    """Add one message per given input (all of them by default), all stamped with the same time."""
    messages = {topic: _message(stamp) for topic in topics or _TOPICS}
    for topic, message in messages.items():
        synchronizer.add(topic, message, stamp)
    return messages


def _later(stamp: tuple[int, int], milliseconds: int) -> tuple[int, int]:
    """Shift a stamp by the given milliseconds."""
    nanoseconds = stamp[0] * 1_000_000_000 + stamp[1] + milliseconds * 1_000_000
    return divmod(nanoseconds, 1_000_000_000)


class TestParseSampleIds:
    """Tests parsing the 'sample_ids' parameter into the sample IDs to request."""

    def test_parses_comma_separated_ids(self):
        """Sample IDs are parsed in the given order."""
        assert parse_sample_ids("0,10,20") == [0, 10, 20]

    def test_parses_single_id(self):
        """A single ID is a valid request."""
        assert parse_sample_ids("7") == [7]

    def test_ignores_surrounding_whitespace(self):
        """Sample IDs separated by ', ' are parsed like IDs separated by ','."""
        assert parse_sample_ids(" 1, 2 ,3 ") == [1, 2, 3]

    @pytest.mark.parametrize("sample_ids", ["", " ", ","])
    def test_no_ids_evaluate_the_whole_dataset(self, sample_ids):
        """An empty value requests no specific samples, so the whole dataset is evaluated."""
        assert parse_sample_ids(sample_ids) == []

    @pytest.mark.parametrize("sample_ids", ["1;2", "first", "1.5", "1-2"])
    def test_rejects_values_that_are_no_ids(self, sample_ids):
        """A value that is no comma-separated list of IDs is rejected."""
        with pytest.raises(ValueError):
            parse_sample_ids(sample_ids)


class TestSampleSynchronizer:
    """Tests matching the messages of the evaluation inputs into the samples to evaluate."""

    def test_matches_the_messages_of_a_sample_in_input_order(self):
        """A sample is reported once every input has been received, in the order of the inputs."""
        synchronizer, matched_samples = _synchronizer()

        messages = _publish_sample(synchronizer, _NEXT_SCENE[0], topics=list(reversed(_TOPICS)))

        assert matched_samples == [tuple(messages[topic] for topic in _TOPICS)]
        assert not synchronizer.incomplete_samples

    def test_waits_for_the_missing_inputs_of_a_sample(self):
        """A sample of which an input is missing is not reported yet."""
        synchronizer, matched_samples = _synchronizer()

        _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["label", "label_meta_info"])

        assert matched_samples == []

    def test_matches_samples_of_a_scene_recorded_before_the_previous_scene(self):
        """The dataset continues with an older scene, whose samples are matched all the same."""
        synchronizer, matched_samples = _synchronizer()

        _publish_sample(synchronizer, _PREVIOUS_SCENE)
        messages = _publish_sample(synchronizer, _NEXT_SCENE[0])

        assert len(matched_samples) == 2
        assert matched_samples[-1] == tuple(messages[topic] for topic in _TOPICS)

    def test_keeps_a_waiting_sample_older_than_a_matched_one(self):
        """A sample of a new, older scene is not dropped by a late sample of the previous scene."""
        synchronizer, matched_samples = _synchronizer()
        # the last sample of the previous scene still waits for the system under test, while the
        # first sample of the next scene, recorded years earlier, is published already
        pending = _publish_sample(synchronizer, _PREVIOUS_SCENE, topics=["label", "label_meta_info"])
        messages = _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["label", "label_meta_info"])

        synchronizer.add("prediction", _message(_PREVIOUS_SCENE), _PREVIOUS_SCENE)
        synchronizer.add("prediction", _message(_NEXT_SCENE[0]), _NEXT_SCENE[0])

        assert len(matched_samples) == 2
        assert matched_samples[0][1:] == (pending["label"], pending["label_meta_info"])
        assert matched_samples[1][1:] == (messages["label"], messages["label_meta_info"])

    def test_gives_up_on_the_sample_waiting_the_longest(self):
        """Samples are dropped in the order they arrived, not by their stamp."""
        synchronizer, matched_samples = _synchronizer(queue_size=2)

        # a sample of the previous scene waits first, followed by two samples of the older scene
        for stamp in [_PREVIOUS_SCENE, *_NEXT_SCENE[:2]]:
            _publish_sample(synchronizer, stamp, topics=["label"])

        # the queue size is exceeded, so the sample that has been waiting the longest is given up
        # on, even though the samples kept for evaluation were recorded years before it
        assert list(synchronizer.incomplete_samples) == _NEXT_SCENE[:2]

        _publish_sample(synchronizer, _NEXT_SCENE[2], topics=["label"])

        assert list(synchronizer.incomplete_samples) == _NEXT_SCENE[1:]
        assert matched_samples == []

    def test_reports_the_stamp_of_the_first_input_and_the_messages_by_input(self):
        """The sample is identified by its message of the first input, whichever message arrived first."""
        reported: list = []
        synchronizer = SampleSynchronizer(
            ["trajectory", "objects"], callback=lambda stamp, messages: reported.append((stamp, messages)), tolerance=0.05
        )
        objects, trajectory = _message(_NEXT_SCENE[0]), _message(_later(_NEXT_SCENE[0], 20))

        synchronizer.add("objects", objects, _NEXT_SCENE[0])
        synchronizer.add("trajectory", trajectory, _later(_NEXT_SCENE[0], 20))

        assert reported == [(_later(_NEXT_SCENE[0], 20), {"trajectory": trajectory, "objects": objects})]

    def test_a_single_input_reports_every_message_as_a_sample(self):
        """An evaluation of a single topic evaluates each of its messages on its own."""
        synchronizer, matched_samples = _synchronizer(topics=["ego_data"])

        for stamp in _NEXT_SCENE:
            _publish_sample(synchronizer, stamp, topics=["ego_data"])

        assert len(matched_samples) == len(_NEXT_SCENE)

    def test_matches_messages_of_different_stamps_within_the_tolerance(self):
        """Topics a simulation publishes at slightly different times are matched within the tolerance."""
        for tolerance, num_matched_samples in [(0.0, 0), (0.05, 1)]:
            synchronizer, matched_samples = _synchronizer(topics=["ego_data", "objects"], tolerance=tolerance)

            _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["ego_data"])
            _publish_sample(synchronizer, _later(_NEXT_SCENE[0], 30), topics=["objects"])

            assert len(matched_samples) == num_matched_samples

    def test_does_not_match_messages_beyond_the_tolerance(self):
        """Messages whose stamps differ by more than the tolerance belong to different samples."""
        synchronizer, matched_samples = _synchronizer(topics=["ego_data", "objects"], tolerance=0.05)

        _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["ego_data"])
        _publish_sample(synchronizer, _later(_NEXT_SCENE[0], 60), topics=["objects"])

        assert matched_samples == []
        assert len(synchronizer.incomplete_samples) == 2

    def test_evaluates_a_sample_without_an_optional_input_that_is_not_published(self):
        """A dataset that publishes no meta information is evaluated without it."""
        synchronizer, matched_samples = _synchronizer(optional_topics=["label_meta_info"], published_topics=[])

        messages = _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["prediction", "label"])

        assert matched_samples == [(messages["prediction"], messages["label"], None)]
        assert not synchronizer.incomplete_samples

    def test_waits_for_an_optional_input_that_is_published(self):
        """The meta information of a dataset that publishes it is evaluated with its sample."""
        synchronizer, matched_samples = _synchronizer(optional_topics=["label_meta_info"], published_topics=["label_meta_info"])

        messages = _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["prediction", "label"])

        assert matched_samples == []

        meta_info = _message(_NEXT_SCENE[0])
        synchronizer.add("label_meta_info", meta_info, _NEXT_SCENE[0])

        assert matched_samples == [(messages["prediction"], messages["label"], meta_info)]

    def test_keeps_an_optional_message_received_before_the_required_ones(self):
        """Meta information published with the labels joins the sample once the prediction completes it."""
        synchronizer, matched_samples = _synchronizer(optional_topics=["label_meta_info"], published_topics=[])

        messages = _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["label", "label_meta_info", "prediction"])

        assert matched_samples == [tuple(messages[topic] for topic in _TOPICS)]

    def test_optional_inputs_never_complete_a_sample_on_their_own(self):
        """A sample is only evaluated once all its required inputs have been received."""
        synchronizer, matched_samples = _synchronizer(optional_topics=["label_meta_info"], published_topics=[])

        _publish_sample(synchronizer, _NEXT_SCENE[0], topics=["label", "label_meta_info"])

        assert matched_samples == []

    def test_matches_the_closest_message_of_a_topic_published_at_a_higher_rate(self):
        """A message of a slower topic joins the message of a faster topic whose stamp is closest to its own."""
        synchronizer, matched_samples = _synchronizer(topics=["ego_data", "objects"], tolerance=0.05)
        ego_data = {
            offset: _publish_sample(synchronizer, _later(_NEXT_SCENE[0], offset), topics=["ego_data"])
            for offset in range(0, 100, 10)
        }

        objects = _publish_sample(synchronizer, _later(_NEXT_SCENE[0], 42), topics=["objects"])

        assert matched_samples == [(ego_data[40]["ego_data"], objects["objects"])]


class _FakeLogger:
    """Collect the messages the node logs instead of publishing them to ROS."""

    def __init__(self):
        """Start with an empty log."""
        self.messages: list = []

    def info(self, message: str, **kwargs) -> None:
        """Record a logged message, whatever its severity."""
        self.messages.append(message)

    debug = warn = error = info


class _FakeEvaluationHandler:
    """Stand in for the evaluation whose results the node aggregates and writes."""

    def __init__(self):
        """Start without finalized or written results."""
        self.finalized_complete = None
        self.written_results = None

    def finalize(self, complete: bool = True) -> dict:
        """Report results that are marked the way the node asked for."""
        self.finalized_complete = complete
        return {"num_samples": 2, "num_scenes": 1, "complete": complete, "metrics": {}}

    def save_results(self, output_path: str, results: dict = None) -> str:
        """Keep the results instead of writing them to a file."""
        self.written_results = results
        return output_path


class _RecordingEvaluationHandler:
    """Stand in for the evaluation that records the samples the node evaluates."""

    def __init__(self):
        """Start without recorded samples."""
        self.recorded_samples: list = []

    def record_sample(self, sample_id: str, **messages) -> dict:
        """Record a sample the way the evaluation stores it, still without a scene."""
        entry = {"sample_id": sample_id, "scene_id": None, "metrics": {}}
        self.recorded_samples.append(entry)
        self.recorded_messages = messages
        return entry


def _evaluating_node(requests_samples: bool) -> SimpleNamespace:
    """Stub the node state that evaluating a sample reads, counting the attempts to continue the evaluation."""
    node = SimpleNamespace(
        requests_samples=requests_samples,
        evaluation_handler=_RecordingEvaluationHandler(),
        num_evaluated_samples=0,
        scenes_awaiting_sample=deque(),
        samples_awaiting_scene=deque(),
        evaluation_timeout=60.0,
        evaluation_deadline=None,
        results_path="",
        visualization_publishers={},
        num_advances=0,
        get_logger=lambda logger=_FakeLogger(): logger,
    )
    node.advance_evaluation = lambda: setattr(node, "num_advances", node.num_advances + 1)
    return node


def _sample_messages(stamp: tuple[int, int]) -> dict:
    """Fake the synchronized input messages of one sample, by input name."""
    return {topic: _message(stamp) for topic in _TOPICS}


class TestEvaluateSample:
    """Tests evaluating a sample with the samples requested from the dataset or published by others."""

    def test_passes_the_messages_by_input_name_and_identifies_the_sample_by_its_stamp(self):
        """The evaluation receives each message by the name of its input."""
        node = _evaluating_node(requests_samples=True)
        messages = _sample_messages(_NEXT_SCENE[0])

        AutonomyEvaluation.evaluate_sample(node, _NEXT_SCENE[0], messages)

        assert node.evaluation_handler.recorded_messages == messages
        assert node.evaluation_handler.recorded_samples[0]["sample_id"] == "1531885320.049418000"

    def test_evaluation_requests_the_next_samples_after_evaluating_one(self):
        """With the dataset as sample source, the evaluation continues with the next samples on its own."""
        node = _evaluating_node(requests_samples=True)

        AutonomyEvaluation.evaluate_sample(node, _NEXT_SCENE[0], _sample_messages(_NEXT_SCENE[0]))

        assert node.num_advances == 1
        # the dataset reports the scene of the sample with the response to the request
        assert list(node.samples_awaiting_scene) == node.evaluation_handler.recorded_samples

    def test_external_sample_source_leaves_publishing_samples_to_others(self):
        """With an external sample source, samples are evaluated as they arrive, without requesting further ones."""
        node = _evaluating_node(requests_samples=False)

        for stamp in _NEXT_SCENE:
            AutonomyEvaluation.evaluate_sample(node, stamp, _sample_messages(stamp))

        assert node.num_evaluated_samples == len(_NEXT_SCENE)
        assert node.num_advances == 0
        # the scenes are only reported to whoever published the samples, so no sample waits for one
        assert not node.samples_awaiting_scene


class _FakeRequestClient:
    """Stand in for the client of the sample request service of the dataset."""

    def __init__(self, ready: bool):
        """Start with the service available or not."""
        self.ready = ready
        self.removed_requests: list = []

    def service_is_ready(self) -> bool:
        """Report whether the dataset node offers the service."""
        return self.ready

    def remove_pending_request(self, future) -> None:
        """Record a request that is given up on."""
        self.removed_requests.append(future)


def _advancing_node(service_ready: bool, dataset_available: bool = True, publishing_finished: bool = False):
    """Stub the node state that advancing the evaluation reads, recording what the evaluation does next."""
    node = SimpleNamespace(
        evaluation_finished=False,
        pending_request=None,
        publishing_finished=publishing_finished,
        dataset_available=dataset_available,
        dataset_unavailable_since=None,
        sample_request_client=_FakeRequestClient(ready=service_ready),
        sample_request_service="/datasets/request_samples",
        actions=[],
        get_logger=lambda logger=_FakeLogger(): logger,
    )
    node.track_dataset_availability = lambda: AutonomyEvaluation.track_dataset_availability(node)
    node.awaiting_evaluations = lambda: False
    node.request_samples = lambda: node.actions.append("request")
    node.finalize_evaluation = lambda: node.actions.append("finalize")
    node.shutdown = lambda: node.actions.append("shutdown")
    return node


class TestAdvanceEvaluation:
    """Tests requesting further samples and finishing the evaluation, with and without the dataset node."""

    @pytest.fixture
    def clock(self, monkeypatch) -> list:
        """Control the steady clock the node measures how long the dataset has been gone with."""
        now = [100.0]
        monkeypatch.setattr(autonomy_evaluation.time, "monotonic", lambda: now[0])
        return now

    def test_requests_samples_while_the_dataset_is_available(self):
        """The next samples are requested as soon as the previous ones have been evaluated."""
        node = _advancing_node(service_ready=True, dataset_available=False)

        AutonomyEvaluation.advance_evaluation(node)

        assert node.actions == ["request"]
        assert node.dataset_available

    def test_waits_for_a_dataset_that_has_not_started_yet(self, clock):
        """A dataset node that never offered its service is waited for, however long it takes."""
        node = _advancing_node(service_ready=False, dataset_available=False)

        clock[0] += 3600.0
        AutonomyEvaluation.advance_evaluation(node)

        assert node.actions == []
        assert not node.publishing_finished

    def test_finishes_after_the_dataset_reported_its_end_and_shut_down(self):
        """The service is only needed to request samples, not to finish once publishing has ended."""
        node = _advancing_node(service_ready=False, publishing_finished=True)

        AutonomyEvaluation.advance_evaluation(node)

        assert node.actions == ["finalize", "shutdown"]

    def test_finishes_once_the_dataset_shut_down_after_its_last_sample(self, clock):
        """A dataset gone for longer than the grace period publishes no further samples."""
        node = _advancing_node(service_ready=False)

        AutonomyEvaluation.advance_evaluation(node)
        clock[0] += autonomy_evaluation._DATASET_SHUTDOWN_GRACE_PERIOD_S / 2
        AutonomyEvaluation.advance_evaluation(node)

        # within the grace period, a response the dataset sent before shutting down may still arrive
        assert node.actions == []

        clock[0] += autonomy_evaluation._DATASET_SHUTDOWN_GRACE_PERIOD_S
        AutonomyEvaluation.advance_evaluation(node)

        assert node.actions == ["finalize", "shutdown"]

    def test_continues_with_a_dataset_that_is_back_within_the_grace_period(self, clock):
        """A service that is briefly unavailable does not end the evaluation."""
        node = _advancing_node(service_ready=False)
        AutonomyEvaluation.advance_evaluation(node)
        clock[0] += autonomy_evaluation._DATASET_SHUTDOWN_GRACE_PERIOD_S / 2

        node.sample_request_client.ready = True
        AutonomyEvaluation.advance_evaluation(node)
        clock[0] += autonomy_evaluation._DATASET_SHUTDOWN_GRACE_PERIOD_S
        AutonomyEvaluation.advance_evaluation(node)

        assert node.actions == ["request", "request"]
        assert not node.publishing_finished

    def test_gives_up_on_a_request_the_shut_down_dataset_did_not_answer(self, clock):
        """A request pending when the dataset shut down would never be answered."""
        node = _advancing_node(service_ready=False)
        request = node.pending_request = object()

        AutonomyEvaluation.advance_evaluation(node)
        clock[0] += 2 * autonomy_evaluation._DATASET_SHUTDOWN_GRACE_PERIOD_S
        AutonomyEvaluation.advance_evaluation(node)

        assert node.sample_request_client.removed_requests == [request]
        assert node.pending_request is None
        assert node.actions == ["finalize", "shutdown"]


class TestReceiveMessage:
    """Tests stamping the received messages before they are matched into samples."""

    @staticmethod
    def _receiving_node(now: tuple[int, int]) -> tuple[SimpleNamespace, list]:
        """Stub the node state that receiving a message reads, collecting the stamped messages."""
        received: list = []
        node = SimpleNamespace(
            message_synchronizer=SimpleNamespace(add=lambda name, message, stamp: received.append((name, message, stamp))),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(seconds_nanoseconds=lambda: now)),
        )
        return node, received

    def test_stamps_a_message_with_its_header_stamp(self):
        """A message is matched by the stamp its publisher gave it."""
        node, received = self._receiving_node(now=_PREVIOUS_SCENE)
        message = _message(_NEXT_SCENE[0])

        AutonomyEvaluation.receive_message(node, "prediction", message)

        assert received == [("prediction", message, _NEXT_SCENE[0])]

    def test_stamps_a_message_without_header_on_reception(self):
        """A message without header, e.g. a std_msgs/Bool, is stamped with the time it is received at."""
        node, received = self._receiving_node(now=_PREVIOUS_SCENE)
        message = SimpleNamespace(data=True)

        AutonomyEvaluation.receive_message(node, "collision", message)

        assert received == [("collision", message, _PREVIOUS_SCENE)]


class TestInputTopic:
    """Tests the topics the inputs of an evaluation are subscribed on."""

    _DERIVED_TOPICS = {"label_meta_info": ("label", "/meta_info")}

    @staticmethod
    def _node(remappings: dict) -> SimpleNamespace:
        """Stub the topic resolution of a node started with the given remappings of its relative names."""

        def resolve_topic_name(topic: str, only_expand: bool = False) -> str:
            return f"/{topic}" if only_expand or topic not in remappings else remappings[topic]

        return SimpleNamespace(resolve_topic_name=resolve_topic_name)

    def test_subscribes_an_input_on_its_name(self):
        """An input is subscribed on its node-relative name, which remappings redirect."""
        node = self._node({"label": "/object_list/lidar_01"})

        assert AutonomyEvaluation.input_topic(node, "label", self._DERIVED_TOPICS) == "label"

    def test_derived_input_follows_the_topic_of_its_source(self):
        """Meta information is subscribed next to the remapped topic of the object list it belongs to."""
        node = self._node({"label": "/object_list/lidar_01"})

        topic = AutonomyEvaluation.input_topic(node, "label_meta_info", self._DERIVED_TOPICS)

        assert topic == "/object_list/lidar_01/meta_info"

    def test_remapped_derived_input_keeps_its_own_topic(self):
        """A derived input that is remapped itself is subscribed where it is remapped to."""
        node = self._node({"label": "/object_list/lidar_01", "label_meta_info": "/meta_info"})

        assert AutonomyEvaluation.input_topic(node, "label_meta_info", self._DERIVED_TOPICS) == "label_meta_info"


class TestIsPublished:
    """Tests deciding whether a sample waits for the message of an optional input."""

    @staticmethod
    def _node(publishers: dict) -> SimpleNamespace:
        """Stub a node whose subscribed topics have the given numbers of publishers."""
        return SimpleNamespace(
            data_subscriptions={"label_meta_info": SimpleNamespace(topic_name="/object_list/lidar_01/meta_info")},
            count_publishers=lambda topic: publishers.get(topic, 0),
            unpublished_optional_topics=set(),
            get_logger=lambda logger=_FakeLogger(): logger,
        )

    def test_waits_for_an_optional_input_that_has_a_publisher(self):
        """The meta information of a dataset that publishes it is awaited."""
        node = self._node({"/object_list/lidar_01/meta_info": 1})

        assert AutonomyEvaluation.is_published(node, "label_meta_info") is True

    def test_logs_once_that_an_optional_input_is_not_published(self):
        """Evaluating without an unpublished optional input is reported once, not for every sample."""
        node = self._node({})

        assert AutonomyEvaluation.is_published(node, "label_meta_info") is False
        assert AutonomyEvaluation.is_published(node, "label_meta_info") is False
        assert len(node.get_logger().messages) == 1
        assert "label_meta_info" in node.get_logger().messages[0]


def _node(num_evaluated_samples: int = 2, results_path: str = "/results/evaluation.json") -> SimpleNamespace:
    """Stub the node state that finalizing an evaluation reads, without initializing ROS."""
    return SimpleNamespace(
        evaluation="counting",
        evaluation_finished=False,
        evaluation_handler=_FakeEvaluationHandler(),
        num_evaluated_samples=num_evaluated_samples,
        results_path=results_path,
        request_timer=SimpleNamespace(cancel=lambda: None),
        get_logger=lambda logger=_FakeLogger(): logger,
    )


class TestFinalizeEvaluation:
    """Tests reporting the results of an evaluation that ran to its end or was interrupted."""

    def test_finished_evaluation_writes_complete_results(self):
        """An evaluation that evaluated all its samples reports complete results."""
        node = _node()

        AutonomyEvaluation.finalize_evaluation(node)

        assert node.evaluation_handler.finalized_complete is True
        assert node.evaluation_handler.written_results["complete"] is True

    def test_interrupted_evaluation_writes_incomplete_results(self):
        """An evaluation stopped before its last sample, e.g. with Ctrl-C, still writes its results."""
        node = _node()

        AutonomyEvaluation.finalize_evaluation(node, complete=False)

        assert node.evaluation_handler.finalized_complete is False
        assert node.evaluation_handler.written_results["complete"] is False

    def test_finished_evaluation_is_not_finalized_again_on_shutdown(self):
        """Shutting down after the last sample must not overwrite the results with incomplete ones."""
        node = _node()
        AutonomyEvaluation.finalize_evaluation(node)

        AutonomyEvaluation.finalize_evaluation(node, complete=False)

        assert node.evaluation_handler.finalized_complete is True
        assert node.evaluation_handler.written_results["complete"] is True

    def test_manual_playback_writes_its_results_when_stopped(self):
        """With manual playback, which runs no request timer, the results are written once the node is stopped."""
        node = _node()
        node.request_timer = None

        AutonomyEvaluation.finalize_evaluation(node, complete=False)

        assert node.evaluation_handler.written_results["complete"] is False

    def test_interrupted_evaluation_without_samples_writes_nothing(self):
        """An evaluation interrupted before its first sample has no results to write."""
        node = _node(num_evaluated_samples=0)

        AutonomyEvaluation.finalize_evaluation(node, complete=False)

        assert node.evaluation_handler.written_results is None

    def test_results_are_only_logged_without_a_results_path(self):
        """Without 'results_path' the interrupted results are logged instead of written."""
        node = _node(results_path="")

        AutonomyEvaluation.finalize_evaluation(node, complete=False)

        assert node.evaluation_handler.finalized_complete is False
        assert node.evaluation_handler.written_results is None
