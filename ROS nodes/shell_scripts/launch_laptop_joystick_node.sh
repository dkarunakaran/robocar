#!/bin/bash

rosparam set joy_node/dev "/dev/input/js2"
rosparam set joy_node/autorepeat_rate 1.0
rosrun joy joy_node
