# autonomy_evaluation

<p align="center">
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/thinking-cars/autonomy_evaluation/releases/latest"><img src="https://img.shields.io/github/v/release/thinking-cars/autonomy_evaluation"/></a>
  <a href="https://github.com/thinking-cars/autonomy_evaluation/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thinking-cars/autonomy_evaluation"/></a>
  <br>
  <a href="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/docker-ros.yml"><img src="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/docker-ros.yml/badge.svg"/></a>
  <a href="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/compose-oci.yml"><img src="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/compose-oci.yml/badge.svg"/></a>
  <a href="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/helm-oci.yml"><img src="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/helm-oci.yml/badge.svg"/></a>
  <a href="https://thinking-cars.github.io/autonomy_evaluation"><img src="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/docs.yml/badge.svg"/></a>
  <a href="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/consistency.yml"><img src="https://github.com/thinking-cars/autonomy_evaluation/actions/workflows/consistency.yml/badge.svg"/></a>
</p>

> This repository is part of the **Autonomy.Benchmarks** suite of the **Autonomy.Hub Ecosystem**

Within the Autonomy.Benchmarks suite, **Autonomy.Evaluation** generates the metrics-based evidence for benchmarking automated driving deployments. It evaluates arbitrary ROS systems under test, either from their own topics alone, e.g. a closed-loop planner by its time to collision, or against ground truth, e.g. a perception algorithm against the labels of a dataset replayed by [Autonomy.Datasets](https://github.com/thinking-cars/autonomy_datasets), and reports the resulting metrics per scene and over all evaluated samples:

- 🔄 **Unified ROS 2 Interface**: Evaluate any ROS system under test, on datasets, in simulation or live, using the benefits of the ROS 2 ecosystem
- 🧩 **Pluggable Evaluations**: Select an evaluation by name, or bring your own from another package, reading any topics as inputs and, where needed, ground truth
- 📊 **Established Metrics**: Use the provided evaluations, which follow the metrics of established benchmarks, with [Autonomy.Datasets](https://github.com/thinking-cars/autonomy_datasets) across different datasets and automated driving tasks
- ⚡ **Efficient Data Pipeline**: Works seamlessly with preprocessed Rosbag files from [Autonomy.Datasets](https://github.com/thinking-cars/autonomy_datasets) for fast execution during development
- 🐳 **Dockerized Environment**: Reproducible setup with all dependencies included
- 🔌 **Modular Architecture**: Easy integration with other ROS 2 packages

## Supported Evaluations

This repository supports evaluations of various automated driving tasks.

Detailed metric definitions and computation notes are documented in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

> [**Contributions**](docs/IMPLEMENTATION.md#adding-more-evaluations) adding more evaluations are welcome

| Evaluation | Datasets | Task | Preview |
| ---------- | -------- | ---- | ------- |
| [**3D Object Detection**](docs/IMPLEMENTATION.md#3d-object-detection) | All [Autonomy.Datasets](https://github.com/thinking-cars/autonomy_datasets) with 3D object labels | 3D bounding box detection on the classes of `perception_msgs/ObjectClassification` | ![Rviz Screenshot 3D Object Detection evaluation](./docs/assets/3d-object-detection.png)

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


## 🚀 Quick Start

Clone [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) and follow its setup instructions to prepare your dataset.

Use the provided [docker-compose.yml](docker-compose.yml) to start the full pipeline — dataset publisher, system under test, and evaluation node:

```bash
# enable GUI output from Docker container
xhost +local:

# pull and start Docker containers
cp .env.template .env
# configure evaluated module and dataset in the '.env' file
docker compose up -d
# stop containers once finished
docker compose down
```

Configure the evaluation and dataset via ROS launch arguments in [docker-compose.yml](docker-compose.yml):

```yaml
command: ros2 launch autonomy_evaluation autonomy_evaluation.launch.py evaluation:=object_detection_3d prediction:=$your_prediction_topic label:=$your_label_topic request_samples:=/datasets/request_samples visualize:=true
```

The evaluation node requests the samples it evaluates from the dataset node via its `request_samples` service, which publishes them and responds once they have been published. The dataset therefore publishes the next sample only once the system under test has processed the current one. As soon as all samples have been published, the node aggregates its metrics per scene of the dataset and over all evaluated samples. To evaluate samples published by others instead, e.g. by a closed-loop simulation, set `sample_source:=external`. See the [node documentation](autonomy_evaluation/README.md#autonomy_evaluation) for the topics of the evaluations, the sample settings and the results.

## 💻 Development

### Set up Development Environment

1. Clone the repository.
    ```bash
    git clone https://github.com/thinking-cars/autonomy_evaluation.git
    ```
1. Initialize the [`.openads-dev-environment`](https://github.com/openads-project/openads-dev-environment) submodule containing development environment configuration.
    ```bash
    cd autonomy_evaluation
    git submodule update --init --recursive
    ```
1. Open the repository in [Visual Studio Code](https://code.visualstudio.com).
    ```bash
    code .
    ```
1. Install the recommended VS Code extensions.
    > *Ctrl+Shift+P / Extensions: Show Recommended Extensions / Install Workspace Recommended Extensions (Cloud Download Icon)*
1. Reopen the repository in a [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers).
    > *Ctrl+Shift+P / Dev Containers: Rebuild and Reopen in Container*

### Build

> *Ctrl+Shift+B*

```bash
colcon build
```

### Run Tests

> *Ctrl+Shift+P / Tasks: Run Test Task*

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=1
colcon test
colcon test-result --verbose
```


## 📝 Documentation

Package and node interfaces are documented in the respective package READMEs listed below. Implementation details are found in the [Source Code Documentation](https://thinking-cars.github.io/autonomy_evaluation).

| Package | Description |
| --- | --- |
| [autonomy_evaluation](autonomy_evaluation/README.md) | Metrics-based evaluation of automated driving modules, generating the evidence for benchmarking automated driving deployments |

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## 🙏 Acknowledgements

This project is maintained by [Thinking Cars](https://thinking-cars.de). We appreciate contributions and are happy to discuss potential collaborations.
