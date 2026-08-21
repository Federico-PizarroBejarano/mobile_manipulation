# Mobile Manipulation
A ROS-based framework for mobile manipulation research, featuring MPC-based control, robot simulation, planning utilities, and N²M²-style RL for base motion.

## Package Overview
- **mm_assets**: Robot and scene URDF/mesh files
- **mm_control**: MPC controller implementation using Acados
- **mm_plan**: Planning base classes and simple planners
- **mm_rl**: SAC training for N²M²-style base control given a scripted EE planner
- **mm_run**: Launch files, configurations, and ROS nodes/scripts
- **mm_simulator**: PyBullet simulation interface
- **mm_utils**: Utility functions for math, parsing, logging, etc.

Configuration parameters are documented in [configuration.md](./mm_run/config/configuration.md).

## Installation
### Prerequisites
Ensure you have ROS Noetic installed on your system. Follow the [ROS Noetic installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu) if it's not already set up.

### Installation of `mobile_manipulation_central`
To install `mobile_manipulation_central`, clone this repository into your ROS workspace and compile it using `catkin`.

```bash
cd ~/catkin_ws/src
git clone https://github.com/utiasDSL/mobile_manipulation_central
git checkout mm_dev
cd ~/catkin_ws
catkin build mobile_manipulation_central
source devel/setup.bash
```

### Pinocchio
```bash
sudo apt install libeigen3-dev ros-noetic-eigenpy ros-noetic-hpp-fcl ros-noetic-pinocchio
```

Make sure to source your ROS environment:
```bash
source /opt/ros/noetic/setup.bash
```

### Acados
Follow the instructions on the [Acados website](https://docs.acados.org/installation/). Don't forget to install the Python interface.

If Python fails to load the solver (e.g. an undefined symbol from `libhpipm`), prepend your Acados `lib` directory before running experiments or `roslaunch`:

```bash
export LD_LIBRARY_PATH=<your_acados_install>/lib:$LD_LIBRARY_PATH
```

### Installing this repo
```bash
cd ~/catkin_ws/src
git clone https://github.com/utiasDSL/mobile_manipulation
cd ~/catkin_ws
catkin build mm_assets mm_utils mm_simulator mm_plan mm_control mm_run mm_rl
source devel/setup.bash
python3 -m pip install -r src/mobile_manipulation/requirements.txt
```

## Usage
Commands below assume the workspace is sourced (`source ~/catkin_ws/devel/setup.bash`). Scripts live under `mm_run/src/mm_run/` and are invoked with `rosrun` after a catkin build.

### Compile MPC Controller
```bash
rosrun mm_control generate_acados_code.py --config $(rospack find mm_run)/config/simple_experiment.yaml
```

### Run Controller with PyBullet Simulation (Synchronous)
Single-process MPC + PyBullet loop (`experiment.py`):

```bash
rosrun mm_run experiment.py --config $(rospack find mm_run)/config/simple_experiment.yaml --GUI
```

### Run Controller and Simulation Asynchronously (ROS Nodes)
Launches `sim_ros`, the plan node (`mpc_ros` by default), `low_level_cmd_node` (publishes `cmd_vel` from `/mpc_plan`), and TF helpers:

```bash
roslaunch mm_run run_pybullet_sim.launch config:=$(rospack find mm_run)/config/simple_experiment.yaml gui:=True
```

To use the MPSF plan node instead of standard MPC:

```bash
roslaunch mm_run run_pybullet_sim.launch config:=$(rospack find mm_run)/config/mpsf_experiment.yaml gui:=True mpsf:=True
```

### Visualize Results
Logs are written under `mm_run/results/<log_dir>/<TIMESTAMP>/`:

- Synchronous `experiment.py`: `combined/`
- ROS runs: `sim/` (simulator) and `control/` (controller)

```bash
roscd mm_utils/scripts
python3 plot_logs.py --folder ../../mm_run/results/[EXPERIMENT_NAME]/[TIMESTAMP] --tracking
```

### RL Training and Evaluation
Train a SAC policy that commands only the mobile base while a scripted EE planner provides desired end-effector velocities:

```bash
cd ~/catkin_ws/src/mobile_manipulation
python3 -m mm_rl.train -c mm_rl/config/train_config.yaml --checkpoint_dir checkpoints --log_dir logs
```

Evaluate a checkpoint (optionally against MPSF on the same goals):

```bash
python3 -m mm_rl.evaluate_model -c mm_rl/config/train_config.yaml \
  --checkpoint checkpoints/final_model.pth --n-episodes 10 --seed 3

python3 -m mm_rl.evaluate_model -c mm_rl/config/train_config.yaml \
  --checkpoint checkpoints/final_model.pth --compare-mpsf --seed 3 --n-episodes 5
```

## Tests
From the repository root, with the workspace sourced.

EE planner unit tests (no simulation):

```bash
python3 mm_rl/test_ee_planner_path.py
```

IK solver used by the RL environment:

```bash
python3 mm_rl/test_ik_solver.py -c mm_rl/config/train_config.yaml
```

EE planner as faux teleoperator with MPSF tracking:

```bash
python3 mm_rl/test_ee_planner_mpsf.py -c mm_rl/config/train_config.yaml --n-episodes 5
python3 mm_rl/test_ee_planner_mpsf.py -c mm_rl/config/train_config.yaml --n-episodes 5 --use-ik-solver
```

MPC / upright scenario suite (sync PyBullet or ROS):

```bash
rosrun mm_run run_scenario_tests.py --mode pybullet
rosrun mm_run run_scenario_tests.py
```

Inherited MPC scenario configs (run individually):

```bash
rosrun mm_run experiment.py --config $(rospack find mm_run)/config/tests/test_position_only.yaml
rosrun mm_run experiment.py --config $(rospack find mm_run)/config/tests/test_orientation_only.yaml
rosrun mm_run experiment.py --config $(rospack find mm_run)/config/tests/test_complex_waypoints.yaml
rosrun mm_run experiment.py --config $(rospack find mm_run)/config/tests/test_path_planning.yaml
rosrun mm_run experiment.py --config $(rospack find mm_run)/config/tests/test_upright_payload.yaml
```

## Configuration
Configuration files are located in `mm_run/config/`. Key configuration options include:

- **Robot**: Robot model parameters (`config/robot/`)
- **Scene**: Environment and obstacle definitions (`config/scene/`)
- **Controller**: MPC parameters (`config/controller/`)
- **Simulation**: Simulation settings (`config/sim/`)
- **Tests / experiments**: Example and integration configs (`config/tests/`, `config/mpsf_experiments/`, `simple_experiment.yaml`)
- **RL**: Training and evaluation configs (`mm_rl/config/`)
