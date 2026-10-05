# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

import json
import signal
import time
from collections import deque, OrderedDict
from functools import partial
from typing import Any, Callable, Collection, Optional, Sequence, Union

import rclpy
import rclpy.exceptions
from autonomy_datasets_msgs.srv import RequestSamples
from autonomy_evaluation.evaluations import load_evaluation
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.subscription import Subscription
from rclpy.task import Future
from rclpy.timer import Timer
from tf2_ros import Buffer, TransformListener

# Interval in seconds at which the evaluation checks whether it can request further samples; the
# evaluation also advances whenever a request is answered or a sample has been evaluated
_REQUEST_TIMER_PERIOD_S = 0.5

# Interval in seconds at which waiting for the sample request service of the dataset is logged
_SERVICE_WAIT_LOG_INTERVAL_S = 10.0

# Seconds for which the sample request service of a dataset that has been available must be gone
# before the dataset node is considered to have shut down, which lets a response that the dataset
# sent just before shutting down still arrive
_DATASET_SHUTDOWN_GRACE_PERIOD_S = 2.0

# Number of samples whose messages are kept while they wait for the messages of their remaining
# evaluation inputs
_SYNCHRONIZER_QUEUE_SIZE = 10

# Values of the 'sample_source' parameter: the node requests the samples to evaluate from the
# dataset, or evaluates whatever samples reach it, e.g. from a simulation or the RViz playback panel
SAMPLE_SOURCE_DATASET = "dataset"
SAMPLE_SOURCE_EXTERNAL = "external"

# Stamp of a message as (seconds, nanoseconds)
Stamp = tuple[int, int]


def parse_sample_ids(sample_ids: str) -> list[int]:
    """Parses the IDs of the dataset samples to evaluate

    Args:
        sample_ids (str): comma-separated sample IDs, e.g. "0,10,20"

    Returns:
        list[int]: parsed sample IDs, empty if no ID is given

    Raises:
        ValueError: if the IDs are not a comma-separated list of integers
    """
    return [int(sample_id) for sample_id in sample_ids.split(",") if sample_id.strip()]


class SampleSynchronizer:
    """Matches the messages of the evaluation inputs that belong to the same sample.

    Messages are matched by their stamp. By default, only messages with exactly the same stamp are
    matched: the dataset stamps all messages of a sample with the recording time of that sample,
    and a system under test stamps its output with the stamp of the input it processed. With a
    tolerance, a message joins the waiting sample whose stamp is closest to its own, as long as
    both stamps differ by no more than the tolerance and the sample still misses a message of its
    input. This matches topics that a simulation or a live system publishes at slightly different
    times or at different rates, of which the first message within the tolerance is matched.

    Messages of a sample that never completes are dropped in the order they arrived, never by
    comparing their stamps: the dataset replays one scene after the other, and a scene can have
    been recorded days before the scene played before it, so the stamp of a message says nothing
    about how recently it was received. (``message_filters.TimeSynchronizer`` drops by stamp
    instead, and therefore discards the messages of a scene that starts before the end of the
    preceding one.)

    A sample is complete once the messages of all required inputs have been received. It waits
    for the message of an optional input only while that input is expected, i.e. while its topic
    has a publisher; otherwise, the sample is reported without it. Not every dataset publishes
    the meta information of its object lists, for example.

    Messages are added from subscription callbacks, which the node executor runs one after
    another, so no locking is needed.
    """

    def __init__(
        self,
        topics: Sequence[str],
        callback: Callable[[Stamp, dict[str, Any]], None],
        queue_size: int = _SYNCHRONIZER_QUEUE_SIZE,
        tolerance: float = 0.0,
        optional_topics: Collection[str] = (),
        expects_message: Callable[[str], bool] = lambda topic: True,
    ):
        """Constructor

        Args:
            topics (Sequence[str]): input names to match, in the order their messages are passed
                to the callback
            callback (Callable[[Stamp, dict[str, Any]], None]): called for every completed sample
                with the stamp of its message of the first required input and its messages by
                input name, which are None for the optional inputs it is reported without
            queue_size (int, optional): number of samples to keep while they wait for the messages
                of their remaining inputs
            tolerance (float, optional): seconds by which the stamps of the messages of a sample
                may differ; 0 only matches messages with exactly the same stamp
            optional_topics (Collection[str], optional): input names among the topics whose
                messages a sample is only waited for while they are expected
            expects_message (Callable[[str], bool], optional): reports whether a message of an
                optional input is expected, which is asked whenever a sample has received the
                messages of all required inputs but not of that optional input
        """
        self.topics = list(topics)
        self.optional_topics = [topic for topic in self.topics if topic in optional_topics]
        self.required_topics = [topic for topic in self.topics if topic not in optional_topics]
        self.callback = callback
        self.expects_message = expects_message
        self.queue_size = queue_size
        self.tolerance_ns = round(tolerance * 1e9)
        # stamped messages of the samples that are still missing inputs, by the stamp of the
        # message that opened the sample and in the order the samples were opened
        self.incomplete_samples: OrderedDict[Stamp, dict[str, tuple[Stamp, Any]]] = OrderedDict()

    def add(self, topic: str, message: Any, stamp: Stamp):
        """Adds a received message and reports the sample it completes to the callback

        Args:
            topic (str): input name the message was received on
            message (Any): received message
            stamp (Stamp): stamp of the message, i.e. of the sample it belongs to
        """
        sample_stamp = self.find_sample(topic, stamp)
        messages = self.incomplete_samples.setdefault(sample_stamp, {})
        messages[topic] = (stamp, message)

        # complete once all required inputs have been received, and those optional inputs that are expected
        if all(topic in messages for topic in self.required_topics) and not any(
            topic not in messages and self.expects_message(topic) for topic in self.optional_topics
        ):
            del self.incomplete_samples[sample_stamp]
            self.callback(
                messages[self.required_topics[0]][0],
                {topic: messages[topic][1] if topic in messages else None for topic in self.topics},
            )
            return

        # give up on the sample that has been waiting for its remaining inputs the longest
        while len(self.incomplete_samples) > self.queue_size:
            self.incomplete_samples.popitem(last=False)

    def find_sample(self, topic: str, stamp: Stamp) -> Stamp:
        """Finds the waiting sample a message belongs to

        A message joins the sample of exactly its stamp, replacing a message the sample already
        holds for its input, or else the waiting sample within the tolerance whose stamp is
        closest to its own and that still misses a message of its input.

        Args:
            topic (str): input name the message was received on
            stamp (Stamp): stamp of the message

        Returns:
            Stamp: stamp of the sample the message belongs to, which is its own stamp if it opens
                a new sample
        """
        if stamp in self.incomplete_samples or not self.tolerance_ns:
            return stamp
        nanoseconds = _to_nanoseconds(stamp)
        candidates = [
            (abs(_to_nanoseconds(sample_stamp) - nanoseconds), sample_stamp)
            for sample_stamp, messages in self.incomplete_samples.items()
            if topic not in messages
        ]
        candidates = [candidate for candidate in candidates if candidate[0] <= self.tolerance_ns]
        return min(candidates)[1] if candidates else stamp


def _to_nanoseconds(stamp: Stamp) -> int:
    """Converts a stamp into nanoseconds

    Args:
        stamp (Stamp): stamp as (seconds, nanoseconds)

    Returns:
        int: nanoseconds
    """
    return stamp[0] * 1_000_000_000 + stamp[1]


class AutonomyEvaluation(Node):
    """ROS 2 node evaluating automated driving tasks, generating metrics-based evidence for benchmarking them."""

    def __init__(self):
        """Constructor"""
        super().__init__("autonomy_evaluation")

        self.auto_reconfigurable_params: list[str] = []
        self.evaluation = self.declare_and_load_parameter(
            name="evaluation",
            param_type=rclpy.Parameter.Type.STRING,
            description="name of an evaluation of this package, or '<module>:<class>' of an evaluation implemented in "
            "another package",
            default="object_detection_3d",
            add_to_auto_reconfigurable_params=False,
            read_only=True,
        )

        self.visualize = self.declare_and_load_parameter(
            name="visualize",
            param_type=rclpy.Parameter.Type.BOOL,
            description="publish the per-sample visualization of the evaluation for RViz, e.g. the true positives, false "
            "positives and false negatives of an object detection",
            default=False,
        )

        self.sample_source = self.declare_and_load_parameter(
            name="sample_source",
            param_type=rclpy.Parameter.Type.STRING,
            description="'dataset' requests the samples to evaluate from the dataset one after another; 'external' "
            "evaluates the samples published by others, e.g. by a simulation, a live system or the playback panel in RViz, "
            "and reports the results once the node is stopped",
            default="dataset",
            add_to_auto_reconfigurable_params=False,
            read_only=True,
        )
        if self.sample_source not in (SAMPLE_SOURCE_DATASET, SAMPLE_SOURCE_EXTERNAL):
            self.get_logger().fatal(
                f"Parameter 'sample_source' is neither '{SAMPLE_SOURCE_DATASET}' nor '{SAMPLE_SOURCE_EXTERNAL}': "
                f"'{self.sample_source}'"
            )
            raise SystemExit(1)
        self.requests_samples = self.sample_source == SAMPLE_SOURCE_DATASET

        self.sync_tolerance = self.declare_and_load_parameter(
            name="sync_tolerance",
            param_type=rclpy.Parameter.Type.DOUBLE,
            description="seconds by which the stamps of the messages of a sample may differ; 0 only matches messages "
            "with exactly the same stamp, as the dataset and a system under test echoing its stamps publish them",
            default=0.0,
            add_to_auto_reconfigurable_params=False,
            read_only=True,
            from_value=0.0,
            to_value=60.0,
        )

        self.samples_per_request = self.declare_and_load_parameter(
            name="samples_per_request",
            param_type=rclpy.Parameter.Type.INTEGER,
            description="number of samples to request from the dataset at a time; 0 requests all remaining "
            "samples at once, 1 evaluates every sample before the next one is published",
            default=1,
            from_value=0,
            to_value=100000,
        )

        self.sample_ids = self.declare_and_load_parameter(
            name="sample_ids",
            param_type=rclpy.Parameter.Type.STRING,
            description="comma-separated IDs of the dataset samples to evaluate (e.g. '0,10,20'); "
            "if empty, all samples of the dataset are evaluated",
            default="",
            add_to_auto_reconfigurable_params=False,
            read_only=True,
        )
        try:
            self.requested_sample_ids = parse_sample_ids(self.sample_ids)
        except ValueError:
            self.get_logger().fatal(f"Parameter 'sample_ids' is not a comma-separated list of sample IDs: '{self.sample_ids}'")
            raise SystemExit(1)

        self.evaluation_timeout = self.declare_and_load_parameter(
            name="evaluation_timeout",
            param_type=rclpy.Parameter.Type.DOUBLE,
            description="seconds to wait for a published sample to be evaluated before continuing without it",
            default=60.0,
            from_value=0.0,
            to_value=3600.0,
        )

        self.results_path = self.declare_and_load_parameter(
            name="results_path",
            param_type=rclpy.Parameter.Type.STRING,
            description="path of the JSON file the evaluation results are written to; results are only logged if empty",
            default="",
        )

        self.setup()

    def declare_and_load_parameter(
        self,
        name: str,
        param_type: rclpy.Parameter.Type,
        description: str,
        default: Optional[Any] = None,
        add_to_auto_reconfigurable_params: bool = True,
        is_required: bool = False,
        read_only: bool = False,
        from_value: Optional[Union[int, float]] = None,
        to_value: Optional[Union[int, float]] = None,
        step_value: Optional[Union[int, float]] = None,
        additional_constraints: str = "",
    ) -> Any:
        """Declares and loads a ROS parameter

        Args:
            name (str): name
            param_type (rclpy.Parameter.Type): parameter type
            description (str): description
            default (Optional[Any], optional): default value
            add_to_auto_reconfigurable_params (bool, optional): enable reconfiguration of parameter
            is_required (bool, optional): whether failure to load parameter will stop node
            read_only (bool, optional): set parameter to read-only
            from_value (Optional[Union[int, float]], optional): parameter range minimum
            to_value (Optional[Union[int, float]], optional): parameter range maximum
            step_value (Optional[Union[int, float]], optional): parameter range step
            additional_constraints (str, optional): additional constraints description

        Returns:
            Any: parameter value
        """

        # declare parameter
        param_desc = ParameterDescriptor()
        param_desc.description = description
        param_desc.additional_constraints = additional_constraints
        param_desc.read_only = read_only
        if from_value is not None and to_value is not None:
            if param_type == rclpy.Parameter.Type.INTEGER:
                value_range = IntegerRange(from_value=from_value, to_value=to_value)
                if step_value is not None:
                    value_range.step = step_value
                param_desc.integer_range = [value_range]
            elif param_type == rclpy.Parameter.Type.DOUBLE:
                value_range = FloatingPointRange(from_value=from_value, to_value=to_value)
                if step_value is not None:
                    value_range.step = step_value
                param_desc.floating_point_range = [value_range]
            else:
                self.get_logger().warn(f"Parameter type of parameter '{name}' does not support specifying a range")
        self.declare_parameter(name, param_type, param_desc)

        # load parameter
        try:
            param = self.get_parameter(name).value
            self.get_logger().info(f"Loaded parameter '{name}': {param}")
        except rclpy.exceptions.ParameterUninitializedException:
            if is_required:
                self.get_logger().fatal(f"Missing required parameter '{name}', exiting")
                raise SystemExit(1)
            else:
                self.get_logger().warn(f"Missing parameter '{name}', using default value: {default}")
                param = default
                self.set_parameters([rclpy.Parameter(name=name, value=param)])

        # add parameter to auto-reconfigurable parameters
        if add_to_auto_reconfigurable_params:
            self.auto_reconfigurable_params.append(name)

        return param

    def parameters_callback(self, parameters: list[rclpy.Parameter]) -> SetParametersResult:
        """Handles reconfiguration when a parameter value is changed

        Args:
            parameters (list[rclpy.Parameter]): parameters

        Returns:
            SetParametersResult: parameter change result
        """

        for param in parameters:
            if param.name in self.auto_reconfigurable_params:
                setattr(self, param.name, param.value)
                self.get_logger().info(f"Reconfigured parameter '{param.name}' to: {param.value}")

        result = SetParametersResult()
        result.successful = True

        return result

    def setup(self):
        """Sets up subscribers, publishers, etc. to configure the node"""

        # callback for dynamic parameter configuration
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.data_subscriptions: dict[str, Subscription] = {}

        # load the selected evaluation and the topics it reads
        try:
            evaluation_handler = load_evaluation(self.evaluation)
            inputs = evaluation_handler.all_inputs()
        except ValueError as exception:
            self.get_logger().fatal(f"{exception}, exiting")
            raise SystemExit(1)

        # provide the evaluation with the transforms between the frames of its messages, e.g. to
        # compare objects given in different frames; the buffer follows the node clock, so that it
        # is cleared when the simulation clock jumps back to a scene recorded earlier
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        evaluation_handler.tf_buffer = self.tf_buffer

        # create subscriptions for the topics of the evaluation, whose messages are matched into
        # the samples to evaluate by their stamp; a sample only waits for the message of an
        # optional topic while that topic is published
        self.unpublished_optional_topics: set[str] = set()
        self.message_synchronizer = SampleSynchronizer(
            topics=list(inputs),
            callback=self.evaluate_sample,
            queue_size=_SYNCHRONIZER_QUEUE_SIZE,
            tolerance=self.sync_tolerance,
            optional_topics=list(evaluation_handler.optional_ground_truth()),
            expects_message=self.is_published,
        )
        derived_topics = evaluation_handler.derived_topics()
        for name, msg_type in inputs.items():
            self.data_subscriptions[name] = self.create_subscription(
                msg_type,
                self.input_topic(name, derived_topics),
                partial(self.receive_message, name),
                qos_profile=QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                ),
            )
        roles = {
            "input": evaluation_handler.required_inputs(),
            "ground truth": evaluation_handler.required_ground_truth(),
            "optional ground truth": evaluation_handler.optional_ground_truth(),
        }
        for role, names in roles.items():
            if names:
                topics = ", ".join(f"'{name}' from '{self.data_subscriptions[name].topic_name}'" for name in names)
                self.get_logger().info(f"Evaluating {role} {topics}")

        # Messages without a header are stamped when they are received, which the messages of
        # other inputs can only match within a tolerance
        unstamped_inputs = [name for name, msg_type in inputs.items() if "header" not in msg_type.get_fields_and_field_types()]
        if unstamped_inputs:
            self.get_logger().info(f"Stamping the messages of {unstamped_inputs} on reception, as they carry no header")
            if len(inputs) > 1 and not self.sync_tolerance:
                self.get_logger().warn(
                    f"Messages of {unstamped_inputs} can only be matched with those of other inputs if they are "
                    "received at exactly their stamp, set 'sync_tolerance' to match them within a tolerance"
                )

        # create publishers visualizing the evaluation's per-sample matching outcome
        self.visualization_publishers: dict[str, Publisher] = {}
        if self.visualize:
            for msg_topic, msg_type in evaluation_handler.visualization_outputs().items():
                self.visualization_publishers[msg_topic] = self.create_publisher(
                    msg_type,
                    f"~/{msg_topic}",
                    qos_profile=QoSProfile(
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=10,
                    ),
                )
            self.get_logger().info(f"Visualizing evaluation results on: {sorted(self.visualization_publishers)}")

        self.evaluation_handler = evaluation_handler

        self.published_sample_ids: list[int] = []
        # A sample may already be evaluated before the dataset node answers the request that
        # published it, so published samples and evaluated samples are matched in publishing
        # order: whichever of the two arrives first waits here for its counterpart, which makes
        # at most one of both queues non-empty at a time.
        self.scenes_awaiting_sample: deque = deque()
        self.samples_awaiting_scene: deque = deque()
        self.pending_request: Optional[Future] = None
        self.evaluation_deadline: Optional[float] = None
        self.publishing_finished = False
        # whether the sample request service of the dataset has been available, and since when it
        # is gone, to tell a dataset node that has not started yet from one that has shut down
        self.dataset_available = False
        self.dataset_unavailable_since: Optional[float] = None
        self.evaluation_finished = False
        self.num_evaluated_samples = 0

        # With an external sample source, the samples are evaluated as they arrive. Whoever
        # publishes them receives the responses of the dataset, if any, so the evaluation neither
        # learns the scenes of the samples nor when publishing has ended, and reports its results
        # once the node is stopped.
        if not self.requests_samples:
            self.request_timer: Optional[Timer] = None
            if self.requested_sample_ids:
                self.get_logger().warn(
                    f"Parameter 'sample_ids' is ignored, as only sample source '{SAMPLE_SOURCE_DATASET}' requests samples"
                )
            self.get_logger().info("Evaluating the samples published by others, stop the node to report the results")
            return

        self.sample_request_client = self.create_client(RequestSamples, "~/request_samples")
        # name of the service with remappings applied, for the log messages
        self.sample_request_service = self.resolve_service_name(self.sample_request_client.srv_name)
        # driven by a steady clock, so that the evaluation also advances while the simulation clock
        # of the dataset stands still, i.e. while no sample is being published
        self.request_timer = self.create_timer(
            _REQUEST_TIMER_PERIOD_S,
            self.advance_evaluation,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(f"Requesting samples to evaluate from '{self.sample_request_service}'")

    def input_topic(self, name: str, derived_topics: dict[str, tuple[str, str]]) -> str:
        """Determines the topic an input of the evaluation is subscribed on

        An input is subscribed on its node-relative name, which is remapped onto the topic of the
        system under test or of the ground truth. An input published next to another one follows
        the topic of that input, unless it is remapped itself.

        Args:
            name (str): input name
            derived_topics (dict[str, tuple[str, str]]): inputs published next to another input,
                as input name -> (name of the other input, suffix of its topic)

        Returns:
            str: topic to subscribe on
        """
        if name not in derived_topics or self.resolve_topic_name(name) != self.resolve_topic_name(name, only_expand=True):
            return name
        source, suffix = derived_topics[name]
        return self.resolve_topic_name(source) + suffix

    def is_published(self, name: str) -> bool:
        """Reports whether the topic of optional ground truth has a publisher, so that its messages are waited for

        A sample whose required inputs have all been received is evaluated without the message of
        an optional input whose topic has no publisher, e.g. without the meta information of the
        labels of a dataset that publishes none. That is logged once per input.

        Args:
            name (str): name of the optional input

        Returns:
            bool: whether the topic of the input has a publisher
        """
        topic = self.data_subscriptions[name].topic_name
        if self.count_publishers(topic) > 0:
            return True
        if name not in self.unpublished_optional_topics:
            self.unpublished_optional_topics.add(name)
            self.get_logger().info(f"Evaluating the samples without optional '{name}', as '{topic}' has no publisher")
        return False

    def advance_evaluation(self):
        """Requests the next samples to evaluate, or finalizes the evaluation once all were published

        Called periodically as well as whenever a request has been answered or a sample has been
        evaluated, and does nothing while the evaluation is waiting for one of those. The service
        of the dataset is only needed to request further samples, so the evaluation finishes even
        after the dataset node has shut down.
        """
        if self.evaluation_finished:
            return
        self.track_dataset_availability()
        if self.pending_request is not None or self.awaiting_evaluations():
            return
        if self.publishing_finished:
            self.finalize_evaluation()
            self.shutdown()
            return
        if not self.dataset_available:
            self.get_logger().warn(
                f"Waiting for service '{self.sample_request_service}' to request samples of the dataset...",
                throttle_duration_sec=_SERVICE_WAIT_LOG_INTERVAL_S,
            )
            return
        if self.dataset_unavailable_since is None:
            self.request_samples()

    def track_dataset_availability(self):
        """Ends publishing once the dataset node has shut down

        The dataset node shuts down after it has published its last sample, without necessarily
        reporting the end of the dataset: a request that publishes the last sample is answered
        before the dataset notices that no sample follows. Once the service of a dataset that has
        been available is gone for longer than a grace period, no further samples can be
        published, so the samples published so far are the ones to evaluate. A request that the
        dataset did not answer before it went away is given up on.
        """
        if self.sample_request_client.service_is_ready():
            self.dataset_available = True
            self.dataset_unavailable_since = None
            return
        if not self.dataset_available or self.publishing_finished:
            return
        now = time.monotonic()
        if self.dataset_unavailable_since is None:
            self.dataset_unavailable_since = now
        if now - self.dataset_unavailable_since < _DATASET_SHUTDOWN_GRACE_PERIOD_S:
            return

        if self.pending_request is not None:
            self.sample_request_client.remove_pending_request(self.pending_request)
            self.pending_request = None
            self.get_logger().warn("The dataset node shut down before answering the last request for samples")
        self.get_logger().info(
            f"The dataset node providing '{self.sample_request_service}' has shut down, finishing with the "
            "samples it published"
        )
        self.publishing_finished = True

    def awaiting_evaluations(self) -> bool:
        """Reports whether published samples are still waiting to be evaluated

        A sample is evaluated once all evaluation inputs have been received for it, which happens
        once the system under test has processed the sample the dataset published. Samples that
        are not evaluated within 'evaluation_timeout' seconds are given up on, so that a system
        under test which skips samples does not stall the evaluation.

        Returns:
            bool: whether the evaluation waits for published samples to be evaluated
        """
        outstanding_evaluations = len(self.scenes_awaiting_sample) - len(self.samples_awaiting_scene)
        if outstanding_evaluations <= 0:
            return False
        if self.evaluation_deadline is not None and time.monotonic() < self.evaluation_deadline:
            return True
        missing_samples = ", ".join(str(sample_id) for sample_id in self.published_sample_ids[-outstanding_evaluations:])
        self.get_logger().warn(
            f"Sample(s) {missing_samples} were not evaluated within {self.evaluation_timeout} s, continuing without them"
        )
        # Only samples waiting to be evaluated are left in the queue, as evaluated samples are
        # matched with a scene as soon as one is published. Dropping their scenes keeps the
        # following samples matched with the scene they were published from; only the evaluation
        # of a given up sample that still arrives later shifts the matching by one sample.
        self.scenes_awaiting_sample.clear()
        return False

    def request_samples(self):
        """Requests the next samples to evaluate from the dataset node"""
        request = RequestSamples.Request()
        if self.requested_sample_ids:
            request.mode = RequestSamples.Request.MODE_SAMPLE_IDS
            request.sample_ids = self.requested_sample_ids
            requested_samples = f"the samples {self.sample_ids}"
        elif self.samples_per_request > 0:
            request.mode = RequestSamples.Request.MODE_NEXT_SAMPLES
            request.num_samples = self.samples_per_request
            requested_samples = f"the next {self.samples_per_request} sample(s)"
        else:
            request.mode = RequestSamples.Request.MODE_ALL_SAMPLES
            requested_samples = "all remaining samples"

        self.get_logger().debug(f"Requesting {requested_samples} of the dataset for evaluation")
        self.pending_request = self.sample_request_client.call_async(request)
        self.pending_request.add_done_callback(self.samples_published_callback)

    def samples_published_callback(self, future: Future):
        """Records the samples the dataset node has published and continues the evaluation

        Args:
            future (Future): future of the request, holding the response of the dataset node
        """
        self.pending_request = None
        try:
            response = future.result()
        except Exception as exception:
            self.get_logger().error(f"Requesting samples of the dataset failed: {exception}")
            self.publishing_finished = True
            self.advance_evaluation()
            return

        published_sample_ids = [int(sample_id) for sample_id in response.published_sample_ids]
        self.published_sample_ids.extend(published_sample_ids)
        self.assign_scenes(response.published_scene_ids)
        self.evaluation_deadline = time.monotonic() + self.evaluation_timeout

        published_samples = ", ".join(str(sample_id) for sample_id in published_sample_ids)
        if response.success:
            self.get_logger().debug(f"Dataset published sample(s) {published_samples}: {response.message}")
        else:
            self.get_logger().warn(f"Dataset did not publish all requested samples: {response.message}")

        # A request for a fixed set of samples is answered once all of them have been published,
        # any other request is repeated until the dataset has published its last sample.
        self.publishing_finished = (
            response.end_of_dataset or not response.success or bool(self.requested_sample_ids) or self.samples_per_request <= 0
        )
        self.advance_evaluation()

    def assign_scenes(self, published_scene_ids: Sequence[str]):
        """Attributes the scenes of published samples to the samples that are evaluated for them

        The evaluation aggregates the metrics of the samples of a scene, so every evaluated sample
        needs the scene the dataset published it from. Samples are evaluated in the order the
        dataset published them, so both are matched in that order; a scene whose sample has not
        been evaluated yet waits for it, and vice versa.

        Args:
            published_scene_ids (Sequence[str]): scenes of the published samples, in the order
                the samples were published
        """
        for scene_id in published_scene_ids:
            if self.samples_awaiting_scene:
                self.samples_awaiting_scene.popleft()["scene_id"] = str(scene_id)
            else:
                self.scenes_awaiting_sample.append(str(scene_id))

    def finalize_evaluation(self, complete: bool = True):
        """Aggregates the metrics of the evaluated samples per scene and over the whole evaluation

        The results hold the metrics of every single sample, of the samples of each scene, and of
        all evaluated samples, of which the metrics over all samples are logged. Calling this on an
        evaluation that has already been finalized does nothing, so that an evaluation which ran to
        its end is not finalized a second time when the node shuts down.

        Args:
            complete (bool, optional): whether all samples to evaluate have been processed; the
                results of an evaluation that was interrupted before its last sample, e.g. with
                Ctrl-C, are marked as incomplete via '"complete": false'
        """
        if self.evaluation_finished:
            return
        self.evaluation_finished = True
        if self.request_timer is not None:
            self.request_timer.cancel()

        if not self.num_evaluated_samples:
            self.get_logger().warn(f"Evaluation '{self.evaluation}' evaluated no sample, no metrics are aggregated")
            return

        results = self.evaluation_handler.finalize(complete=complete)
        aggregated_metrics = json.dumps(results["metrics"], indent=2, default=str)
        evaluated_samples = f"{results['num_samples']} evaluated sample(s) of {results['num_scenes']} scene(s)"
        if complete:
            self.get_logger().info(f"Evaluation '{self.evaluation}' finished after {evaluated_samples}.")
        else:
            self.get_logger().warn(
                f"Evaluation '{self.evaluation}' was interrupted after {evaluated_samples}, "
                "its results are marked as incomplete."
            )

        if self.results_path:
            reported_results = "evaluation results" if complete else "incomplete evaluation results"
            try:
                results_path = self.evaluation_handler.save_results(self.results_path, results=results)
                self.get_logger().info(f"Wrote {reported_results} to '{results_path}'")
            except OSError as exception:
                self.get_logger().error(f"Failed to write {reported_results} to '{self.results_path}': {exception}")
        else:
            self.get_logger().info(f"Aggregated dataset metrics:\n{aggregated_metrics}")

    def shutdown(self):
        """Stops the node, as there is nothing left to evaluate once the evaluation has finished

        Shutting down the ROS context ends the spinning of the node in 'main', which lets the
        process exit with the evaluation results reported.
        """
        self.get_logger().info("Nothing left to evaluate, shutting down")
        rclpy.try_shutdown()

    def receive_message(self, name: str, message: Any):
        """Stamps a received message of an input and hands it to the matching of the samples

        A message is stamped with the stamp of its header, which the dataset sets to the
        recording time of its sample. A message without a header is stamped with the time it is
        received at.

        Args:
            name (str): input name the message was received on
            message (Any): received message
        """
        header = getattr(message, "header", None)
        if header is not None:
            stamp = (header.stamp.sec, header.stamp.nanosec)
        else:
            stamp = self.get_clock().now().seconds_nanoseconds()
        self.message_synchronizer.add(name, message, stamp)

    def evaluate_sample(self, stamp: Stamp, messages: dict[str, Any]):
        """Evaluates a single sample once the messages of all its inputs have been received

        The messages are passed to the evaluation by the names of their inputs, which match the
        parameters of its ``compute_sample_metrics``. The message of optional ground truth whose
        topic is not published is None.

        Samples are identified by the stamp of their message of the first input, which is the
        stamp the dataset recorded them with, and are attributed to the scene the dataset
        reported for them, which can arrive after the sample has been evaluated.

        Args:
            stamp (Stamp): stamp of the sample
            messages (dict[str, Any]): messages of the sample by input name
        """
        sample_id = f"{stamp[0]}.{stamp[1]:09d}"
        self.get_logger().debug(f"Evaluating sample '{sample_id}'")

        result = self.evaluation_handler.record_sample(sample_id=sample_id, **messages)
        self.num_evaluated_samples += 1
        # attribute the sample to the scene the dataset published it from, which the dataset may
        # only report after the sample has been evaluated; with an external sample source, the
        # scene is only reported to whoever requested the sample, if at all
        if self.requests_samples:
            if self.scenes_awaiting_sample:
                result["scene_id"] = self.scenes_awaiting_sample.popleft()
            else:
                self.samples_awaiting_scene.append(result)
        # a system under test that needs longer for some samples must not run into the timeout,
        # which therefore restarts with every evaluated sample
        self.evaluation_deadline = time.monotonic() + self.evaluation_timeout
        if not self.results_path:
            self.get_logger().debug(f"Sample '{sample_id}' result: {result}")

        # publish the sample's visualization for inspection in RViz
        if self.visualization_publishers:
            visualization = self.evaluation_handler.visualize_sample(sample_id=sample_id, **messages)
            for msg_topic, publisher in self.visualization_publishers.items():
                publisher.publish(visualization[msg_topic])

        # request the next samples, or aggregate the dataset metrics if this was the last one; with
        # an external sample source, others publish the next samples instead
        if self.requests_samples:
            self.advance_evaluation()


def main():
    """Initializes ROS, runs the node event loop, and performs shutdown cleanup."""

    rclpy.init()
    node = AutonomyEvaluation()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # An evaluation that is stopped before its last sample, e.g. with Ctrl-C, still reports the
        # samples it did evaluate, marked as incomplete results. An evaluation that ran to its end
        # has been finalized already and is left untouched. Ctrl-C reaches the node from the
        # terminal and once more from the launch system, so further interrupts are ignored while
        # the results are written, which they would otherwise abort.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.finalize_evaluation(complete=False)
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
