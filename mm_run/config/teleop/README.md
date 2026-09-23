# Joystick Teleop Configuration
This directory contains configuration files for different joystick controllers.

## Supported Controllers

- **PS4 Controller**: `ps4.yaml`
- **XBOX Controller (USB)**: `xbox.yaml`

Button indices are the same for sim and hardware; see comments in each YAML file.

## Robot / tool collision

- Gripper-only: experiment includes `config/robot/thing.yaml` (shaft + palm).
- Tray: include `config/robot/thing_tray.yaml` instead (shaft + palm + tray spheres).

See `mm_run/config/configuration.md` (tool collision modes).

## Installation
Before using the joystick teleop, you need to install the required ROS packages:

```bash
# Install ROS joy package (required for joystick support)
sudo apt-get update
sudo apt-get install ros-noetic-joy

# Optional: Install joystick testing utilities
sudo apt-get install joystick
```

## Usage
### Launching Teleop
For XBOX controller (USB):
```bash
roslaunch mm_run teleop.launch controller_type:=xbox
```

For PS4 controller (default):
```bash
roslaunch mm_run teleop.launch controller_type:=ps4
```

### Specifying Device
If your controller is on a different device (e.g., `/dev/input/js1`):
```bash
roslaunch mm_run teleop.launch controller_type:=xbox joy_dev:=/dev/input/js1
```

### Finding Your Controller Device
1. List available joystick devices:
   ```bash
   ls -la /dev/input/js*
   ```

2. Test a specific device:
   ```bash
   jstest /dev/input/js0
   ```

3. If using Bluetooth, make sure your controller is paired:
   ```bash
   bluetoothctl
   # Then use: scan on, pair <MAC_ADDRESS>, connect <MAC_ADDRESS>
   ```

## Testing Your Controller
Use the test script to verify your controller is working:

```bash
# Make sure teleop.launch is running first, then:
rosrun mm_run test_joystick.py xbox
```

This will display real-time button and axis values from your controller.

## Troubleshooting
### Controller Not Detected
1. Check if the device exists:
   ```bash
   ls -la /dev/input/js*
   ```

2. Check permissions (you may need to add your user to the `input` group):
   ```bash
   sudo usermod -a -G input $USER
   # Then log out and back in
   ```

3. For Bluetooth controllers, ensure they're connected:
   ```bash
   bluetoothctl devices
   ```

### Wrong Button Mappings
If buttons don't work as expected, you can test the raw input:
```bash
jstest /dev/input/js0
```

Press buttons and note which button number corresponds to which physical button. The button numbers in the code are:
- Button 2: Start/End experiment (Square / X)

## Joystick Teleop (sim and real robot)

Stick commands are **chassis-relative**: stick forward means robot-forward, strafe left means robot-left (same for EE linear/angular). Nodes rotate those twists into the world frame using the current base yaw before MPC, diff-IK, or RL EE integration.

All joystick consumers use **`/bluetooth_teleop/joy`** (deadman relay, Square/Triangle, sticks). Only one process can open `/dev/input/js0` at a time.

- **Real robot:** use the existing lab / Clearpath joy publisher (already `/bluetooth_teleop/joy`).
- **Sim / no joy yet:** `roslaunch mm_run teleop.launch` starts a `joy_node` under `bluetooth_teleop`.

Launch arg **`teleop:=mpsf|direct|rl|none`** selects the plan publisher (`mpsf_ros`, `direct_teleop_ros`, `rl_teleop_ros`, or `mpc_ros`).

Experiment configs (task, not mode):

| Config | Task |
|--------|------|
| `teleop_pick_place.yaml` | Pick-and-place: `thing.yaml`, upright off, includes RL checkpoint |
| `teleop_tray.yaml` | Tray: `thing_tray.yaml`, upright on, no RL |

| Mode | Behavior |
|------|----------|
| `mpsf` | Sticks → MPC desired velocity (collision-aware) |
| `direct` | Sticks → synthetic `MpcPlan` (base twist or arm diff-IK) |
| `rl` | Sticks → EE (MoMa integrate); SAC policy → base; diff-IK → `MpcPlan` (pick-place config only) |
| `none` | Standard MPC / waypoint experiments, no stick teleop |

### Launch checklist (real robot)

**Terminal 1 — robot + hardware deadman relay:**
```bash
roslaunch mobile_manipulation_central thing.launch
```
(`thing.launch` only enables the relay node via `use_joy_stick_relay` default `true`. The joy topic `/bluetooth_teleop/joy` and enable button index `13` (d-pad up) are **hardcoded** in `mobile_manipulation_central/joy_stick_relay.py`)

Gripper bringup: `thing.launch` / `gripper.launch` default `activate_gripper:=true` runs a one-shot that activates the Robotiq **only if** it is not already ready (safe if a tray is already gripped). Pass `activate_gripper:=false` to skip. Rebuild `mobile_manipulation_central` after pulling so the one-shot is installed.

**Hardware check:** after `thing.launch`, the gripper should finish its init dance once (or log "already ... ready"). Relaunching with the gripper still powered/ready should not re-dance. Mid-experiment Triangle should only open/close.

**Terminal 2 — pick-and-place (same config; switch mode with `teleop:=`):**
```bash
roslaunch mm_run hardware_teleop.launch teleop:=mpsf \
  config:=$(rospack find mm_run)/config/mpsf_experiments/teleop_pick_place.yaml
# teleop:=direct or teleop:=rl with the same config
```

**Tray balancing (mpsf/direct; RL not intended):**
```bash
roslaunch mm_run hardware_teleop.launch teleop:=mpsf \
  config:=$(rospack find mm_run)/config/mpsf_experiments/teleop_tray.yaml
```

For RL, retrain if needed then set `rl.checkpoint` in `teleop_pick_place.yaml` or pass `--checkpoint /path/to/final_model.pth` on the node:
```bash
cd ~/catkin_ws/src/mobile_manipulation
python3 -m mm_rl.train -c mm_rl/config/train_config.yaml --checkpoint_dir checkpoints
```

(`joystick` defaults to `false` so a second `joy_node` is not started.) If joy is missing, add `joystick:=true controller_type:=ps4`.

Or launch controller only:
```bash
roslaunch mm_run run.launch teleop:=mpsf config:=.../teleop_pick_place.yaml
roslaunch mm_run run.launch teleop:=direct config:=.../teleop_pick_place.yaml
roslaunch mm_run run.launch teleop:=rl config:=.../teleop_pick_place.yaml
roslaunch mm_run run.launch teleop:=mpsf config:=.../teleop_tray.yaml
```

### Sim RL / pick-place teleop
```bash
roslaunch mm_run teleop.launch
roslaunch mm_run run_pybullet_sim.launch teleop:=rl gui:=True \
  config:=$(rospack find mm_run)/config/mpsf_experiments/teleop_pick_place.yaml
```

### Button map (PS4 / Xbox)

| Button | PS4 | Xbox | Role |
|--------|-----|------|------|
| D-pad up | 13 | 13 | Hardware deadman and stick enable — hold to drive |
| L1 / LB | 4 | 4 | Clearpath teleop — **do not hold during mm teleop** |
| R1 / RB | 5 | 5 | Clearpath teleop — **do not hold during mm teleop** |
| Circle / B | 1 | 1 | Unused |
| Triangle / Y | 3 | 3 | Toggle gripper open / close (normal mode) |
| Cross / A | 0 | 0 | Toggle base ↔ EE teleop mode (`mpsf`/`direct` only; unused in `rl`) |
| Square / X | 2 | 2 | Start / stop experiment |
| D-pad L / R | 12 / 11 | 12 / 11 | EE yaw |
| D-pad down | 14 | 14 | Unused |

### Verify d-pad indices

Before first hardware run, confirm d-pad left/right button numbers (joy already up on the robot):
```bash
rosrun mm_run test_joystick.py ps4
```
Or in sim / without lab joy: `roslaunch mm_run teleop.launch controller_type:=ps4` first. Press d-pad left and right; update `controller.teleop.ee_yaw_buttons` in the experiment YAML if indices differ from `[12, 11]` (left, right).

### During an experiment

1. Position robot with Clearpath L1/LB + sticks (experiment not running).
2. Press **Square** or **Enter** to start.
3. Hold **d-pad up** and move the sticks. Releasing it stops the motors and ignores the sticks.
4. Press Cross to toggle base / EE mode (direct EE uses arm-only differential IK; base stays still).
5. Press **Triangle** to toggle the gripper open / close (normal mode; ignored in sim if the driver is not running).
6. Press Square again to stop. Release d-pad up for an immediate hardware stop.
7. Do not hold L1/R1 (Clearpath).

Stick teleop requires `controller.teleop.enabled: true`. Gate param: `/teleop_sticks_active`. Direct EE IK params live under top-level `ik:` in the experiment YAML (defaults match RL).

If sticks move but nothing happens, check the warn log for `pressed buttons=[...]`.
You need `13` (d-pad up) in that list.

## Logging and rosbags

Each trial writes under `mm_run/results/<log_dir>/<timestamp>/`:
- `control/data.npz` — states, joy, desired twists, `u_cmd`, (MPSF) `mpc_*`
- `metrics/metrics.npz` + `summary.txt`
- `bag/trial.bag` (or `trial_all.bag` if `bag_all:=true`)

Hardware: `record_bag` defaults true. Sim: false unless passed.
Bag starts on Square/Enter and stops on Square/shutdown.

Launch overrides:
```bash
roslaunch mm_run hardware_teleop.launch record_bag:=false
roslaunch mm_run run_pybullet_sim.launch config:=... record_bag:=true bag_all:=true
```

## Configuration Parameters
All config files support:
- `deadzone`: Minimum value before joystick input is registered (default: 0.1)
- `autorepeat_rate`: Rate at which button presses are repeated (default: 20 Hz)
