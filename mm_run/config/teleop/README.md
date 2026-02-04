# Joystick Teleop Configuration
This directory contains configuration files for different joystick controllers.

## Supported Controllers

- **PS4 Controller**: `ps4.yaml`
- **XBOX Controller (USB)**: `xbox.yaml`

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
- Button 2: Start/End tasks
- Button 1: Task switching (if enabled)

## Configuration Parameters
All config files support:
- `deadzone`: Minimum value before joystick input is registered (default: 0.1)
- `autorepeat_rate`: Rate at which button presses are repeated (default: 20 Hz)
