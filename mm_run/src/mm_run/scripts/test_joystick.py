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
    joy_topic = "/bluetooth_teleop/joy"

    print(f"Testing {controller_type} controller...")
    print(f"Listening to topic: {joy_topic}")
    print("Button map: mm_run/config/teleop/README.md")
    print("Press Ctrl+C to exit\n")

    print("Waiting for joystick messages...")
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
