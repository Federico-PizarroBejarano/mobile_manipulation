#!/usr/bin/env python3
"""Simple script to test joystick input and button mappings.

Usage:
    rosrun mm_run test_joystick.py [controller_type - ps4 or xbox (default: xbox)]
"""

import sys

import rospy
from sensor_msgs.msg import Joy


def test_joystick(controller_type="xbox"):
    """Test joystick input and display button/axis values."""
    rospy.init_node("test_joystick", anonymous=True)

    # Determine topic based on namespace
    joy_topic = "/teleop/joy"

    print(f"Testing {controller_type} controller...")
    print(f"Listening to topic: {joy_topic}")
    print("Press Ctrl+C to exit\n")

    if controller_type == "ps4":
        print("PS4 Button mappings:")
        print("  Button 0: X")
        print("  Button 1: Circle (task switching)")
        print("  Button 2: Square (start/end)")
        print("  Button 3: Triangle")
    else:  # xbox
        print("XBOX Button mappings:")
        print("  Button 0: A")
        print("  Button 1: B (task switching)")
        print("  Button 2: X (start/end)")
        print("  Button 3: Y")

    print("\nWaiting for joystick messages...")
    print("Press buttons or move sticks to see values\n")

    def joy_callback(msg):
        """Callback to display joystick data."""
        print("\033[2J\033[H")  # Clear screen
        print(f"Controller: {controller_type}")
        print(f"Topic: {joy_topic}")
        print("\nButtons (pressed=1, released=0):")
        for i, button in enumerate(msg.buttons):
            status = "✓ PRESSED" if button == 1 else " "
            print(f"  Button {i}: {button} {status}")

        print("\nAxes (sticks/triggers):")
        for i, axis in enumerate(msg.axes):
            print(f"  Axis {i}: {axis:.3f}")

        print("\nPress Ctrl+C to exit")

    rospy.Subscriber(joy_topic, Joy, joy_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("\n\nTest complete!")


if __name__ == "__main__":
    controller_type = sys.argv[1] if len(sys.argv) > 1 else "xbox"
    test_joystick(controller_type)
