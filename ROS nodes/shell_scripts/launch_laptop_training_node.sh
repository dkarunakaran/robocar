#!/bin/bash
export PATH=/home/dhanoop/anaconda3/bin:$PATH
unset PYTHONPATH
source activate training_node
export PYTHONPATH="/home/dhanoop/anaconda3/envs/training_node/lib/python3.5/site-packages:/opt/ros/kinetic/lib/python2.7/dist-packages:/usr/lib/python2.7/site-packages"
cd /home/dhanoop/catkin_ws/src/training_node/scripts
rosrun training_node training_node.py
