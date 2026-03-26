cmake_minimum_required(VERSION 3.0.2)
project(mm_run)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
   src/mm_run/scripts/experiment.py
   src/mm_run/scripts/mpsf_experiment.py
   src/mm_run/scripts/test_joystick.py
   src/mm_run/nodes/sim_ros.py
   src/mm_run/nodes/isaac_sim_ros.py
   src/mm_run/nodes/mpc_ros.py
   src/mm_run/nodes/mpsf_ros.py
   src/mm_run/nodes/planner_test_ros.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
